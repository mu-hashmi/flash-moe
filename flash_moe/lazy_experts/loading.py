# Copyright © 2023-2025 Apple Inc.

import contextlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def _find_switch_mlp(layer, layer_idx=None):
    """Find the SwitchGLU module in a model layer, supporting multiple architectures.

    Returns (switch_mlp, key_prefix_base) or (None, None) if not an MoE layer.

    Supported paths:
      - layer.mlp.switch_mlp (Qwen, DeepSeek, GLM, Hunyuan, Jamba, OLMoE)
      - layer.block_sparse_moe.switch_mlp (Mixtral, PhiMoE, MiniMax, GraniteMoE)
    """
    prefix = f"model.layers.{layer_idx}" if layer_idx is not None else None

    if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
        switch = layer.mlp.switch_mlp
        key_base = f"{prefix}.mlp.switch_mlp" if prefix else "mlp.switch_mlp"
        return switch, key_base

    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
        switch = layer.block_sparse_moe.switch_mlp
        key_base = f"{prefix}.block_sparse_moe.switch_mlp" if prefix else "block_sparse_moe.switch_mlp"
        return switch, key_base

    return None, None


def _find_moe_block(layer):
    """Find the MoE block in a layer (the parent of switch_mlp).

    Returns the MoE block or None. Works for both Qwen (layer.mlp) and
    Mixtral (layer.block_sparse_moe) families.
    """
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
        return layer.mlp
    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
        return layer.block_sparse_moe
    return None


def _detect_num_experts(switch_mlp) -> int:
    """Detect number of experts from a SwitchGLU module."""
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(switch_mlp, name, None)
        if proj is not None and hasattr(proj, "num_experts"):
            return proj.num_experts
    return 512


def _build_shard_map(model_path: Path) -> dict[str, str]:
    """Read model.safetensors.index.json and return {key: absolute_shard_path}.

    For models with per-expert safetensors keys (e.g. Mixtral, GLM), adds
    synthetic stacked-format keys so callers can look up
    ``{prefix}.switch_mlp.gate_proj.weight`` even though the actual file
    stores ``{prefix}.experts.0.w1.weight``.  The shard path returned is
    expert 0's shard (all experts for a layer share a shard).
    """
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_map = {key: str(model_path / shard) for key, shard in weight_map.items()}

    # Detect per-expert format and add synthetic stacked keys.
    # Two naming conventions:
    #   Mixtral:  {prefix}.experts.0.w1.weight  (w1->gate, w2->down, w3->up)
    #   GLM/DS:   {prefix}.experts.0.gate_proj.weight
    _w_to_proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    seen_expert_prefixes: set[str] = set()
    for key in weight_map:
        if ".experts.0." not in key:
            continue
        idx = key.index(".experts.0.")
        moe_prefix = key[:idx]
        remainder = key[idx + len(".experts.0."):]
        parts = remainder.split(".")
        if len(parts) != 2:
            continue
        sub_name, wt = parts
        proj_name = _w_to_proj.get(sub_name, sub_name)
        synth_key = f"{moe_prefix}.switch_mlp.{proj_name}.{wt}"
        if synth_key not in shard_map:
            shard_map[synth_key] = str(model_path / weight_map[key])
            seen_expert_prefixes.add(moe_prefix)

    return shard_map


_PROJ_TO_EXPERT_NAMES = {
    "gate_proj": ("gate_proj", "w1"),
    "up_proj": ("up_proj", "w3"),
    "down_proj": ("down_proj", "w2"),
}


def _load_proj_experts(shard: dict, key_prefix: str, expert_ids,
                       shard_map: dict[str, str] | None = None,
                       ) -> tuple[mx.array, mx.array, mx.array | None]:
    """Load weight/scales/biases for ``expert_ids`` from a safetensors shard.

    Handles both:
      - **Stacked format** (Qwen): ``{key_prefix}.weight`` is a (E, ...) tensor.
      - **Per-expert format** (Mixtral, GLM): individual keys like
        ``{moe_base}.experts.{e}.{sub}.weight``.

    For per-expert format, some experts may live in a different shard file.
    Pass ``shard_map`` (from ``_build_shard_map``) to enable cross-shard
    loading.  Extra shards are loaded on demand and freed immediately.
    """
    stacked_key = f"{key_prefix}.weight"
    if stacked_key in shard:
        w = shard[stacked_key][expert_ids]
        s = shard[f"{key_prefix}.scales"][expert_ids]
        biases_key = f"{key_prefix}.biases"
        b = shard[biases_key][expert_ids] if biases_key in shard else None
        return w, s, b

    # Per-expert format: key_prefix is e.g.
    #   "model.layers.0.block_sparse_moe.switch_mlp.gate_proj"
    # We need to map back to "model.layers.0.block_sparse_moe.experts.{e}.w1"
    parts = key_prefix.rsplit(".", 1)
    switch_prefix = parts[0]
    proj_name = parts[1]
    moe_base = switch_prefix.rsplit(".switch_mlp", 1)[0]

    candidates = _PROJ_TO_EXPERT_NAMES.get(proj_name, (proj_name,))

    ids = np.asarray(expert_ids).reshape(-1) if not isinstance(expert_ids, np.ndarray) else expert_ids.reshape(-1)

    _extra_shards: dict[str, dict] = {}

    def _resolve_shard(expert_key: str) -> dict:
        """Return the shard dict containing ``expert_key``."""
        if expert_key in shard:
            return shard
        if shard_map is None:
            return shard
        alt_path = shard_map.get(expert_key)
        if alt_path is None:
            return shard
        if alt_path not in _extra_shards:
            _extra_shards[alt_path] = mx.load(alt_path)
        return _extra_shards[alt_path]

    ws, ss, bs = [], [], []
    has_bias = None
    for eid in ids:
        eid = int(eid)
        loaded = False
        for sub in candidates:
            expert_key = f"{moe_base}.experts.{eid}.{sub}.weight"
            s_dict = _resolve_shard(expert_key)
            if expert_key in s_dict:
                ws.append(s_dict[expert_key])
                ss.append(s_dict[f"{moe_base}.experts.{eid}.{sub}.scales"])
                b_key = f"{moe_base}.experts.{eid}.{sub}.biases"
                if has_bias is None:
                    has_bias = b_key in s_dict
                if has_bias:
                    bs.append(s_dict[b_key])
                loaded = True
                break
        if not loaded:
            raise KeyError(
                f"No expert key found for expert {eid}, "
                f"tried: {[f'{moe_base}.experts.{eid}.{s}.weight' for s in candidates]}"
            )

    w = mx.stack(ws)
    s = mx.stack(ss)
    b = mx.stack(bs) if has_bias else None
    return w, s, b


def select_capacity(target_model_memory_gb: float, system_memory_gb: float,
                    num_moe_layers: int = 48,
                    expert_slot_mb: float = 1.69) -> int:
    """Select expert cache capacity to stay under the Metal memory pressure cliff.

    Uses a 56% of system RAM heuristic for the Metal headroom budget, then
    divides available space by per-slot expert memory.

    Args:
        target_model_memory_gb: Size of non-expert model weights in GB.
        system_memory_gb: Total device memory in GB.
        num_moe_layers: Number of MoE layers in the model.
        expert_slot_mb: Memory per expert slot per layer in MB.

    Returns:
        Optimal capacity per layer (multiple of 8, clamped to 0..512).
    """
    # Apple Silicon reports ~34 GB for a "32 GB" Mac (unified memory includes
    # system overhead). 0.53 accounts for this to stay at cap 208 on 32 GB.
    budget_gb = system_memory_gb * 0.53 - target_model_memory_gb
    expert_memory_per_slot_gb = num_moe_layers * expert_slot_mb / 1024
    capacity = int(budget_gb / expert_memory_per_slot_gb)
    capacity = (capacity // 8) * 8
    return max(0, min(512, capacity))


def _with_cache_limit_zero(cache_bytes: int = 0):
    """Context manager to temporarily reduce Metal cache limit.

    Reclaims MLX buffer cache headroom, giving 1.3-2x speedup for operations
    above the 20 GB Metal pressure cliff. A small non-zero value (e.g. 256 MB)
    allows intermediate buffer reuse during forward passes while still
    reclaiming most cache memory.
    """
    @contextlib.contextmanager
    def _ctx():
        default_limit = mx.device_info()["memory_size"] // 4
        mx.set_cache_limit(cache_bytes)
        if cache_bytes == 0:
            mx.clear_cache()
        try:
            yield
        finally:
            mx.set_cache_limit(default_limit)

    return _ctx()


def compute_adaptive_allocations(
    layer_profiles: dict[int, dict],
    total_budget: int,
    min_per_layer: int = 32,
) -> dict:
    """Compute optimal per-layer expert cache allocations using MoEpic greedy.

    Iteratively transfers one slot from the layer with lowest marginal cost
    to the layer with highest marginal utility.

    Args:
        layer_profiles: Dict mapping layer_idx to profile dict with:
            - ``"working_set"``: list of (expert_id, activation_count) sorted descending
            - ``"entropy"``: float (routing entropy)
            - ``"unique_count"``: int
        total_budget: Total expert slots to allocate across all layers.
        min_per_layer: Minimum slots per layer.

    Returns:
        Dict with ``"allocations"``, ``"miss_rates"``, and ``"iterations"``.
    """
    layers = sorted(layer_profiles.keys())
    n_layers = len(layers)

    base = max(min_per_layer, total_budget // n_layers)
    allocs = {li: min(base, 512) for li in layers}

    current_total = sum(allocs.values())
    if current_total < total_budget:
        deficit = total_budget - current_total
        for li in layers:
            if deficit <= 0:
                break
            add = min(deficit, 512 - allocs[li])
            allocs[li] += add
            deficit -= add
    elif current_total > total_budget:
        surplus = current_total - total_budget
        for li in reversed(layers):
            if surplus <= 0:
                break
            remove = min(surplus, allocs[li] - min_per_layer)
            allocs[li] -= remove
            surplus -= remove

    def _miss_rate(layer_idx, cap):
        ws = layer_profiles[layer_idx]["working_set"]
        if not ws or cap >= len(ws):
            return 0.0
        total_activations = sum(cnt for _, cnt in ws)
        if total_activations == 0:
            return 0.0
        covered = sum(cnt for _, cnt in ws[:cap])
        return 1.0 - covered / total_activations

    def _marginal_cost(layer_idx):
        cap = allocs[layer_idx]
        if cap <= min_per_layer:
            return float('inf')
        return _miss_rate(layer_idx, cap - 1) - _miss_rate(layer_idx, cap)

    def _marginal_utility(layer_idx):
        cap = allocs[layer_idx]
        if cap >= 512:
            return 0.0
        return _miss_rate(layer_idx, cap) - _miss_rate(layer_idx, cap + 1)

    max_iterations = total_budget * 2
    iterations = 0
    for _ in range(max_iterations):
        donor = min(layers, key=_marginal_cost)
        recipient = max(layers, key=_marginal_utility)

        if donor == recipient:
            break
        cost = _marginal_cost(donor)
        utility = _marginal_utility(recipient)
        if utility <= cost or utility <= 1e-9:
            break

        allocs[donor] -= 1
        allocs[recipient] += 1
        iterations += 1

    miss_rates = {li: _miss_rate(li, allocs[li]) for li in layers}

    return {
        "allocations": allocs,
        "miss_rates": miss_rates,
        "iterations": iterations,
    }
