# Copyright © 2023-2025 Apple Inc.

import time
from pathlib import Path

import numpy as np
import mlx.core as mx

from .loading import _find_switch_mlp, _find_moe_block
from .modules import (
    PredictiveCachedSwitchLinear,
    SyncPredictiveCachedSwitchLinear,
    CachedQuantizedSwitchLinear,
    PredictiveExpertCache,
)


def _run_gate(moe_block, x) -> mx.array:
    """Run the MoE block's gate and return top-k expert indices.

    Handles two gate styles:
      - Standard (Qwen, Mixtral): gate returns raw logits, we softmax + argpartition
      - GLM-style: gate returns (inds, scores) tuple directly
    """
    result = moe_block.gate(x)
    if isinstance(result, tuple):
        inds, _ = result
        return inds

    gates = result
    k = getattr(moe_block, "top_k", None) or getattr(moe_block, "num_experts_per_tok", 2)
    gates = mx.softmax(gates, axis=-1, precise=True)
    return mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]


def _shared_expert_output(moe_block, x) -> mx.array:
    """Compute the shared expert contribution, or zeros if none."""
    if hasattr(moe_block, "shared_expert") and hasattr(moe_block, "shared_expert_gate"):
        se = moe_block.shared_expert(x)
        return mx.sigmoid(moe_block.shared_expert_gate(x)) * se
    if hasattr(moe_block, "shared_expert"):
        return moe_block.shared_expert(x)
    if hasattr(moe_block, "shared_experts"):
        return moe_block.shared_experts(x)
    return mx.zeros_like(x)


def router_only_forward(model, tokenizer, prompt: str,
                        max_tokens: int = 10) -> dict[int, set[int]]:
    """Run the model with MoE expert computation skipped, collecting router selections.

    Monkey-patches MoE blocks to run the gate (router) and shared expert but
    skip switch_mlp. Hidden states drift without MoE output, but routers still
    produce plausible expert selections. Works with any model architecture that
    uses _find_moe_block-detectable MoE layers.

    Args:
        model: The loaded MLX model.
        tokenizer: The model's tokenizer.
        prompt: Text prompt for generation.
        max_tokens: Number of tokens to generate.

    Returns:
        Dict mapping layer_idx to set of expert IDs selected across all tokens.
    """
    import mlx_lm as _mlx_lm

    collected: dict[int, set[int]] = {}
    moe_blocks: dict[int, int] = {}  # id(block) -> layer_idx

    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_blocks[id(block)] = i
            collected[i] = set()

    if not moe_blocks:
        return collected

    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_call(block_map, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)

            inds = _run_gate(self, x)
            mx.eval(inds)
            flat = np.asarray(inds.reshape(-1))
            collected[layer_idx].update(int(e) for e in flat)
            return _shared_expert_output(self, x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_call(moe_blocks, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    return collected


def router_only_discovery(model, tokenizer, prompt: str,
                          max_tokens: int = 10) -> dict[int, set[int]]:
    """Fast cold-start discovery: run routers only, populate Phase 2 caches.

    Like router_only_forward() but batches all mx.eval to the end instead of
    sync-ing per layer per token (480 sync points -> 1). Populates the
    CachedQuantizedSwitchLinear caches so upgrade_to_predictive() can use them.

    Args:
        model: The loaded MLX model with Phase 2 cached modules.
        tokenizer: The model's tokenizer.
        prompt: Text prompt for generation.
        max_tokens: Number of tokens to generate.

    Returns:
        Dict mapping layer_idx to set of expert IDs discovered.
    """
    import mlx_lm as _mlx_lm

    collected: dict[int, list[mx.array]] = {}
    moe_blocks: dict[int, int] = {}

    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_blocks[id(block)] = i
            collected[i] = []

    if not moe_blocks:
        return {}

    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_call(block_map, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)

            inds = _run_gate(self, x)
            collected[layer_idx].append(inds)
            return _shared_expert_output(self, x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_call(moe_blocks, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    # One bulk eval for all collected indices
    all_tensors = []
    for inds_list in collected.values():
        all_tensors.extend(inds_list)
    if all_tensors:
        mx.eval(*all_tensors)

    # Flatten into sets and populate Phase 2 caches
    result: dict[int, set[int]] = {}
    for i, inds_list in collected.items():
        expert_counts: dict[int, int] = {}
        for inds in inds_list:
            flat = np.asarray(inds.reshape(-1))
            for eid in flat:
                eid = int(eid)
                expert_counts[eid] = expert_counts.get(eid, 0) + 1
        result[i] = set(expert_counts.keys())

        layer = model.layers[i]
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        cache = proj._cache
        for eid, count in expert_counts.items():
            cache.entries[eid] = {}
            cache.frequency[eid] = count
            cache.last_active[eid] = max_tokens
            cache.all_seen.add(eid)
        cache.step = max_tokens

    return result


def speculative_router_probe(model, tokenizer, prompt: str,
                             max_tokens: int = 10) -> dict[int, set[int]]:
    """Skip-MoE forward, then probe each router on ALL layers' hidden states.

    Runs a skip-MoE forward pass (same as router_only_forward) to capture
    per-layer hidden states cheaply. Then probes each MoE layer's router on
    the UNION of all layers' captured hidden states, not just its own.

    This tests whether hidden states from other layers help discover experts
    that the layer's own drifted state misses.

    Args:
        model: The loaded MLX model.
        tokenizer: The model's tokenizer.
        prompt: Text prompt for generation.
        max_tokens: Number of tokens to generate.

    Returns:
        Dict mapping layer_idx to set of expert IDs discovered.
    """
    import mlx_lm as _mlx_lm

    moe_layer_indices: list[int] = []
    moe_blocks_map: dict[int, int] = {}  # id(block) -> layer_idx
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_layer_indices.append(i)
            moe_blocks_map[id(block)] = i

    hidden_states_per_layer: dict[int, list[mx.array]] = {i: [] for i in moe_layer_indices}

    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks_map:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_and_capture(block_map, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)

            mx.eval(x)
            hidden_states_per_layer[layer_idx].append(x)
            return _shared_expert_output(self, x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_and_capture(moe_blocks_map, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    # Collect ALL hidden states from ALL layers into one pool
    all_states: list[mx.array] = []
    for layer_idx in moe_layer_indices:
        all_states.extend(hidden_states_per_layer[layer_idx])

    # Probe each router on the full pool of hidden states
    collected: dict[int, set[int]] = {}
    for layer_idx in moe_layer_indices:
        moe_block = _find_moe_block(model.layers[layer_idx])
        all_experts = set()
        for h in all_states:
            inds = _run_gate(moe_block, h)
            mx.eval(inds)
            flat = np.asarray(inds.reshape(-1))
            all_experts.update(int(e) for e in flat)
        collected[layer_idx] = all_experts

    return collected


def speculative_router_cross_layer(model, tokenizer, prompt: str,
                                   max_tokens: int = 10) -> dict[int, set[int]]:
    """Probe all routers using only the FIRST MoE layer's hidden states.

    Uses the skip-MoE forward pass. Captures hidden states only at the first
    MoE layer, then feeds those same states through every other MoE layer's
    router. This tests the strongest form of the MoEpic hypothesis: that a
    single layer's hidden states predict all other layers' routing.

    Args:
        model: The loaded MLX model.
        tokenizer: The model's tokenizer.
        prompt: Text prompt for generation.
        max_tokens: Number of tokens to generate.

    Returns:
        Dict mapping layer_idx to set of expert IDs discovered.
    """
    import mlx_lm as _mlx_lm

    moe_layer_indices: list[int] = []
    moe_blocks_map: dict[int, int] = {}  # id(block) -> layer_idx
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_layer_indices.append(i)
            moe_blocks_map[id(block)] = i

    first_moe = moe_layer_indices[0]
    captured_states: list[mx.array] = []

    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks_map:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_and_capture_first(block_map, first_idx, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)
            if layer_idx == first_idx:
                mx.eval(x)
                captured_states.append(x)
            return _shared_expert_output(self, x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_and_capture_first(
            moe_blocks_map, first_moe, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    # Probe every router with first layer's hidden states
    collected: dict[int, set[int]] = {}
    for layer_idx in moe_layer_indices:
        moe_block = _find_moe_block(model.layers[layer_idx])
        all_experts = set()
        for h in captured_states:
            inds = _run_gate(moe_block, h)
            mx.eval(inds)
            flat = np.asarray(inds.reshape(-1))
            all_experts.update(int(e) for e in flat)
        collected[layer_idx] = all_experts

    return collected
