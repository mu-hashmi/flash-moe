# Copyright © 2023-2025 Apple Inc.

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mlx.core as mx

from .loading import (
    _find_switch_mlp,
    _find_moe_block,
    _build_shard_map,
    _load_proj_experts,
    _load_experts,
    _with_cache_limit_zero,
    SafetensorsMap,
)
from .modules import (
    PredictiveCachedSwitchLinear,
    SyncPredictiveCachedSwitchLinear,
    CachedQuantizedSwitchLinear,
    PredictiveExpertCache,
)


def delta_warmup(model, tokenizer, model_path, new_prompt,
                 discovery_tokens: int = 10) -> dict:
    """Fast cache update for multi-turn: discover missing experts, swap them in.

    Instead of the slow reset -> re-warmup -> upgrade cycle (~70s), keeps the
    existing predictive cache and does a fast "delta discovery" pass:
    1. Run discovery_tokens through predictive cache (~0.5s at 20 tok/s)
    2. Collect which experts were requested but not cached
    3. Evict cold experts, load missing ones from disk
    4. One-time tensor rebuild per affected layer

    Args:
        model: The loaded MLX model with predictive modules.
        tokenizer: The model's tokenizer.
        model_path: Path to the model directory containing safetensors shards.
        new_prompt: Prompt text for discovery.
        discovery_tokens: Number of tokens to generate for expert discovery.

    Returns:
        Dict with timing breakdown and per-layer swap stats.
    """
    import mlx_lm as _mlx_lm

    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache._indices_buffer.clear()

    # Step 1: Discovery pass at full speed through existing predictive cache
    t0 = time.perf_counter()
    _mlx_lm.generate(model, tokenizer, prompt=new_prompt,
                     max_tokens=discovery_tokens, verbose=False)
    t_discovery = time.perf_counter() - t0

    # Step 2: Collect requested expert IDs and identify misses per layer
    t1 = time.perf_counter()
    layer_info = {}

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache

        all_requested = set()
        for indices in cache._indices_buffer:
            flat = np.asarray(indices.reshape(-1))
            all_requested |= set(int(x) for x in np.unique(flat))
        cache._indices_buffer.clear()

        missing = all_requested - cache.cached_set

        cold = sorted(
            [(cache._lcp_priority(eid), slot, eid)
             for slot, eid in enumerate(cache.cached_ids)
             if eid not in all_requested],
        )

        layer_info[i] = {
            "requested": all_requested,
            "missing": missing,
            "cold": cold,
            "cache": cache,
        }

    # Step 3: Compute per-layer swaps and group shard loads across all layers
    total_swaps = 0
    total_missing = 0
    per_layer_stats = []
    layer_swaps: dict[int, list[tuple]] = {}

    for i, info in layer_info.items():
        missing = info["missing"]
        cold = list(info["cold"])
        total_missing += len(missing)

        if not missing:
            per_layer_stats.append({"layer": i, "missing": 0, "swapped": 0})
            continue

        swaps = []
        for new_eid in sorted(missing):
            if not cold:
                break
            _, slot, old_eid = cold.pop(0)
            swaps.append((slot, old_eid, new_eid))

        if not swaps:
            per_layer_stats.append({"layer": i, "missing": len(missing), "swapped": 0})
            continue

        layer_swaps[i] = swaps
        total_swaps += len(swaps)
        per_layer_stats.append({
            "layer": i, "missing": len(missing), "swapped": len(swaps),
        })

    # Group all shard loads across layers
    shard_groups: dict[str, list[tuple]] = {}
    for i, swaps in layer_swaps.items():
        cache = layer_info[i]["cache"]
        new_eids = mx.array([new_eid for _, _, new_eid in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            sp = cache._shard_paths[proj_name]
            kp = cache._key_prefixes[proj_name]
            shard_groups.setdefault(sp, []).append((i, proj_name, kp, new_eids))

    loaded_experts: dict[int, dict[str, tuple]] = {}
    st_map = getattr(model, "_st_map", None)

    for shard_path, group in shard_groups.items():
        shard = None if st_map is not None else mx.load(shard_path)

        layers_in_batch = sorted(set(li for li, _, _, _ in group))
        for layer_i in layers_in_batch:
            layer_entries = [(pn, kp, eids) for li, pn, kp, eids in group if li == layer_i]
            to_eval = []
            for proj_name, key_prefix, new_eids in layer_entries:
                new_w, new_s, new_b = _load_experts(key_prefix, new_eids,
                                                     shard=shard, shard_map=shard_map,
                                                     st_map=st_map)
                loaded_experts.setdefault(layer_i, {})[proj_name] = (new_w, new_s, new_b)
                to_eval.extend([new_w, new_s])
                if new_b is not None:
                    to_eval.append(new_b)
            mx.eval(*to_eval)

        del shard

    # Step 4: Scatter-update per layer and rebuild lookups
    for i, swaps in layer_swaps.items():
        cache = layer_info[i]["cache"]
        slot_indices = mx.array([slot for slot, _, _ in swaps])

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            new_w, new_s, new_b = loaded_experts[i][proj_name]

            w = cache.weights.pop(proj_name)
            w[slot_indices] = new_w
            cache.weights[proj_name] = w

            s = cache.scales.pop(proj_name)
            s[slot_indices] = new_s
            cache.scales[proj_name] = s

            if cache.biases[proj_name] is not None and new_b is not None:
                b = cache.biases.pop(proj_name)
                b[slot_indices] = new_b
                cache.biases[proj_name] = b

        del loaded_experts[i]

        for slot, old_eid, new_eid in swaps:
            cache.cached_set.discard(old_eid)
            cache.cached_set.add(new_eid)
            cache.cached_ids[slot] = new_eid
            cache.frequency.pop(old_eid, None)
            cache.last_active.pop(old_eid, None)

        cache.rebuild_lookup()

        to_eval = [cache.lookup, cache.hit_mask]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            to_eval.append(cache.weights[proj_name])
            to_eval.append(cache.scales[proj_name])
            if cache.biases[proj_name] is not None:
                to_eval.append(cache.biases[proj_name])
        mx.eval(*to_eval)

    t_rebuild = time.perf_counter() - t1

    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache.total_requests = 0
            proj._cache.total_fallbacks = 0
            proj._cache._indices_buffer.clear()

    mx.clear_cache()

    return {
        "discovery_time": t_discovery,
        "rebuild_time": t_rebuild,
        "total_time": t_discovery + t_rebuild,
        "total_swaps": total_swaps,
        "total_missing": total_missing,
        "per_layer": per_layer_stats,
    }


def fast_delta_warmup(model, tokenizer, model_path, new_prompt,
                      discovery_tokens: int = 10, discovery_method: str = "predictive",
                      min_swaps_threshold: int = 0) -> dict:
    """Optimized delta warmup with pluggable discovery and fused load+scatter.

    Two discovery methods:
      "predictive" -- generate through existing predictive cache (~1.3s, high fidelity)
      "router-only" -- skip MoE expert computation, run routers only (~0.5s, lower fidelity)

    Args:
        model: The loaded MLX model with predictive modules.
        tokenizer: The model's tokenizer.
        model_path: Path to the model directory containing safetensors shards.
        new_prompt: Prompt text for discovery.
        discovery_tokens: Number of tokens to generate for expert discovery.
        discovery_method: Discovery strategy ("predictive" or "router-only").
        min_swaps_threshold: Skip layers with fewer missing experts than this.
            Reduces Metal buffer traffic at cost of slightly higher fallback rate.

    Returns:
        Dict with timing breakdown and per-layer swap stats.
    """
    from .discovery import router_only_forward

    model_path = Path(model_path)

    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache._indices_buffer.clear()

    # Step 1: Discovery (under cache_limit(0) to reclaim Metal headroom)
    t0 = time.perf_counter()

    with _with_cache_limit_zero():
        if discovery_method == "router-only":
            discovered = router_only_forward(model, tokenizer, new_prompt,
                                             max_tokens=discovery_tokens)
        else:
            import mlx_lm as _mlx_lm
            _mlx_lm.generate(model, tokenizer, prompt=new_prompt,
                             max_tokens=discovery_tokens, verbose=False)
            discovered = {}
            for i, layer in enumerate(model.layers):
                switch, _ = _find_switch_mlp(layer, i)
                if switch is None:
                    continue
                proj = getattr(switch, "up_proj", None)
                if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                         SyncPredictiveCachedSwitchLinear)):
                    continue
                cache = proj._cache
                requested = set()
                for indices in cache._indices_buffer:
                    flat = np.asarray(indices.reshape(-1))
                    requested |= set(int(x) for x in np.unique(flat))
                cache._indices_buffer.clear()
                discovered[i] = requested

    t_discovery = time.perf_counter() - t0

    # Memory pressure check
    device_mem = mx.device_info()["memory_size"]
    active_mem = mx.get_active_memory()
    memory_pressure = active_mem > 0.85 * device_mem
    MAX_SWAPS_PER_LAYER = 3 if memory_pressure else 10
    if memory_pressure:
        print(f"  [memory pressure: {active_mem / 1e9:.1f}/{device_mem / 1e9:.0f} GB — "
              f"limiting to {MAX_SWAPS_PER_LAYER} swaps/layer]")

    # Step 2: Compute delta
    t1 = time.perf_counter()
    total_swaps = 0
    total_missing = 0
    per_layer_stats = []
    layer_swaps: dict[int, list[tuple]] = {}
    layer_caches: dict[int, PredictiveExpertCache] = {}

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                 SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache
        layer_caches[i] = cache
        requested = discovered.get(i, set())
        missing = requested - cache.cached_set
        total_missing += len(missing)

        if not missing or len(missing) < min_swaps_threshold:
            per_layer_stats.append({"layer": i, "missing": len(missing), "swapped": 0})
            continue

        cold = sorted(
            [(cache._lcp_priority(eid), slot, eid)
             for slot, eid in enumerate(cache.cached_ids)
             if eid not in requested],
        )

        swaps = []
        for new_eid in sorted(missing):
            if not cold:
                break
            _, slot, old_eid = cold.pop(0)
            swaps.append((slot, old_eid, new_eid))

        swaps = swaps[:MAX_SWAPS_PER_LAYER]

        if not swaps:
            per_layer_stats.append({"layer": i, "missing": len(missing), "swapped": 0})
            continue

        layer_swaps[i] = swaps
        total_swaps += len(swaps)
        per_layer_stats.append({
            "layer": i, "missing": len(missing), "swapped": len(swaps),
        })

    # Step 3: Group shard loads, then fused load+scatter per layer within each shard
    # Reduce cache limit during shard loads to prevent GPU timeout under memory pressure.
    shard_layers: dict[str, set[int]] = {}
    for i, swaps in layer_swaps.items():
        cache = layer_caches[i]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            sp = cache._shard_paths[proj_name]
            shard_layers.setdefault(sp, set()).add(i)

    t_shard_load = 0.0
    t_scatter = 0.0
    st_map = getattr(model, "_st_map", None)

    _default_cache = mx.device_info()["memory_size"] // 4
    mx.set_cache_limit(256 * 1024 * 1024)

    for shard_path, layer_set in shard_layers.items():
        t_load_start = time.perf_counter()
        shard = None if st_map is not None else mx.load(shard_path)
        t_shard_load += time.perf_counter() - t_load_start

        for layer_i in sorted(layer_set):
            swaps = layer_swaps[layer_i]
            cache = layer_caches[layer_i]
            new_eids = mx.array([new_eid for _, _, new_eid in swaps])
            slot_indices = mx.array([slot for slot, _, _ in swaps])

            t_load_start = time.perf_counter()
            loaded = {}
            to_eval = []
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                key_prefix = cache._key_prefixes[proj_name]
                if cache._shard_paths[proj_name] != shard_path:
                    continue
                new_w, new_s, new_b = _load_experts(key_prefix, new_eids,
                                                     shard=shard, shard_map=cache._shard_map,
                                                     st_map=st_map)
                loaded[proj_name] = (new_w, new_s, new_b)
                to_eval.extend([new_w, new_s])
                if new_b is not None:
                    to_eval.append(new_b)
            mx.eval(*to_eval)
            t_shard_load += time.perf_counter() - t_load_start

            t_scatter_start = time.perf_counter()
            _use_scatter_slots = hasattr(mx.fast, "scatter_slots")
            if _use_scatter_slots:
                targets = []
                values_list = []
                proj_order = []
                for proj_name, (new_w, new_s, new_b) in loaded.items():
                    targets.append(cache.weights.pop(proj_name))
                    values_list.append(new_w)
                    proj_order.append((proj_name, "weights"))

                    targets.append(cache.scales.pop(proj_name))
                    values_list.append(new_s)
                    proj_order.append((proj_name, "scales"))

                    if cache.biases[proj_name] is not None and new_b is not None:
                        targets.append(cache.biases.pop(proj_name))
                        values_list.append(new_b)
                        proj_order.append((proj_name, "biases"))

                results = mx.fast.scatter_slots(targets, slot_indices, values_list)
                for (proj_name, tensor_type), result in zip(proj_order, results):
                    getattr(cache, tensor_type)[proj_name] = result
            else:
                for proj_name, (new_w, new_s, new_b) in loaded.items():
                    w = cache.weights.pop(proj_name)
                    w[slot_indices] = new_w
                    cache.weights[proj_name] = w

                    s = cache.scales.pop(proj_name)
                    s[slot_indices] = new_s
                    cache.scales[proj_name] = s

                    if cache.biases[proj_name] is not None and new_b is not None:
                        b = cache.biases.pop(proj_name)
                        b[slot_indices] = new_b
                        cache.biases[proj_name] = b
            t_scatter += time.perf_counter() - t_scatter_start

        del shard

    # Rebuild lookup tables and eval scatter results in batches
    t_lookup_start = time.perf_counter()
    EVAL_BATCH = 10
    batch_eval = []
    batch_count = 0

    for layer_i in layer_swaps:
        cache = layer_caches[layer_i]
        swaps = layer_swaps[layer_i]

        for slot, old_eid, new_eid in swaps:
            cache.cached_set.discard(old_eid)
            cache.cached_set.add(new_eid)
            cache.cached_ids[slot] = new_eid
            cache.frequency.pop(old_eid, None)
            cache.last_active.pop(old_eid, None)

        cache.rebuild_lookup()

        batch_eval.append(cache.lookup)
        batch_eval.append(cache.hit_mask)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            batch_eval.append(cache.weights[proj_name])
            batch_eval.append(cache.scales[proj_name])
            if cache.biases[proj_name] is not None:
                batch_eval.append(cache.biases[proj_name])
        batch_count += 1

        if batch_count >= EVAL_BATCH:
            mx.eval(*batch_eval)
            batch_eval = []
            batch_count = 0

    if batch_eval:
        mx.eval(*batch_eval)

    mx.set_cache_limit(_default_cache)

    t_lookup_rebuild = time.perf_counter() - t_lookup_start
    t_rebuild = time.perf_counter() - t1

    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache.total_requests = 0
            proj._cache.total_fallbacks = 0
            proj._cache._indices_buffer.clear()

    mx.clear_cache()

    layers_swapped = len(layer_swaps)
    layers_skipped = sum(1 for s in per_layer_stats
                         if s["missing"] > 0 and s["swapped"] == 0)

    return {
        "discovery_time": t_discovery,
        "rebuild_time": t_rebuild,
        "shard_load_time": t_shard_load,
        "scatter_time": t_scatter,
        "lookup_rebuild_time": t_lookup_rebuild,
        "total_time": t_discovery + t_rebuild,
        "total_swaps": total_swaps,
        "total_missing": total_missing,
        "layers_swapped": layers_swapped,
        "layers_skipped": layers_skipped,
        "discovery_method": discovery_method,
        "per_layer": per_layer_stats,
    }


@dataclass
class LayerSwapPlan:
    layer_idx: int
    cache: PredictiveExpertCache
    swaps: list  # [(slot, old_eid, new_eid), ...]
    miss_count: int

    def __post_init__(self):
        self.new_eids = mx.array([new_eid for _, _, new_eid in self.swaps])
        self.slot_indices = mx.array([slot for slot, _, _ in self.swaps])


class IncrementalDeltaWarmup:
    """Progressive expert cache updates between tokens.

    After discover(), call step() between generated tokens to incrementally
    swap experts. Each step() builds lazy scatter graphs for N layers --
    no mx.eval(), the forward pass evaluates them naturally.

    Usage::

        warmup = IncrementalDeltaWarmup(model, tokenizer, model_path)
        stats = warmup.discover(new_prompt)

        for response in mlx_lm.stream_generate(model, tokenizer, new_prompt, ...):
            print(response.text, end='', flush=True)
            if not warmup.is_complete:
                warmup.step()
    """

    def __init__(self, model, tokenizer, model_path):
        self._model = model
        self._tokenizer = tokenizer
        self._model_path = Path(model_path)
        self._shard_map = _build_shard_map(self._model_path)
        self._swap_queue: list[LayerSwapPlan] = []
        self._layers_done = 0
        self._swaps_done = 0
        self._total_layers = 0
        self._total_swaps = 0
        self._memory_pressure = False

    def discover(self, prompt: str, tokens: int = 10) -> dict:
        """Run discovery pass and compute swap plans.

        Generates tokens through the existing predictive cache to discover
        which experts the new prompt needs, then computes per-layer swap
        plans sorted by miss count (highest first).

        Args:
            prompt: Prompt text for discovery.
            tokens: Number of tokens to generate for discovery.

        Returns:
            Dict with discovery_time, total_layers, total_swaps, total_missing.
        """
        import mlx_lm as _mlx_lm

        for layer in self._model.layers:
            switch, _ = _find_switch_mlp(layer)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if isinstance(proj, (PredictiveCachedSwitchLinear,
                                 SyncPredictiveCachedSwitchLinear)):
                proj._cache._indices_buffer.clear()

        t0 = time.perf_counter()
        with _with_cache_limit_zero():
            _mlx_lm.generate(self._model, self._tokenizer, prompt=prompt,
                             max_tokens=tokens, verbose=False)

        discovered = {}
        for i, layer in enumerate(self._model.layers):
            switch, _ = _find_switch_mlp(layer, i)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                     SyncPredictiveCachedSwitchLinear)):
                continue
            cache = proj._cache
            requested = set()
            for indices in cache._indices_buffer:
                flat = np.asarray(indices.reshape(-1))
                requested |= set(int(x) for x in np.unique(flat))
            cache._indices_buffer.clear()
            discovered[i] = requested

        t_discovery = time.perf_counter() - t0

        # Memory pressure check
        device_mem = mx.device_info()["memory_size"]
        active_mem = mx.get_active_memory()
        self._memory_pressure = active_mem > 0.85 * device_mem
        if self._memory_pressure:
            print(f"  [memory pressure: {active_mem / 1e9:.1f}/{device_mem / 1e9:.0f} GB — "
                  f"swap plans will be trimmed]")

        # Compute swap plans
        self._swap_queue = []
        total_missing = 0
        max_swaps_per_plan = 3 if self._memory_pressure else 999

        for i, layer in enumerate(self._model.layers):
            switch, _ = _find_switch_mlp(layer, i)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                     SyncPredictiveCachedSwitchLinear)):
                continue

            cache = proj._cache
            requested = discovered.get(i, set())
            missing = requested - cache.cached_set
            total_missing += len(missing)

            if not missing:
                continue

            cold = sorted(
                [(cache._lcp_priority(eid), slot, eid)
                 for slot, eid in enumerate(cache.cached_ids)
                 if eid not in requested],
            )

            swaps = []
            for new_eid in sorted(missing):
                if not cold:
                    break
                _, slot, old_eid = cold.pop(0)
                swaps.append((slot, old_eid, new_eid))

            swaps = swaps[:max_swaps_per_plan]

            if swaps:
                self._swap_queue.append(LayerSwapPlan(
                    layer_idx=i, cache=cache,
                    swaps=swaps, miss_count=len(missing),
                ))

        self._swap_queue.sort(key=lambda p: p.miss_count, reverse=True)
        self._total_layers = len(self._swap_queue)
        self._total_swaps = sum(p.miss_count for p in self._swap_queue)
        self._layers_done = 0
        self._swaps_done = 0

        for layer in self._model.layers:
            switch, _ = _find_switch_mlp(layer)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if isinstance(proj, (PredictiveCachedSwitchLinear,
                                 SyncPredictiveCachedSwitchLinear)):
                proj._cache.total_requests = 0
                proj._cache.total_fallbacks = 0
                proj._cache._indices_buffer.clear()

        return {
            "discovery_time": t_discovery,
            "total_layers": self._total_layers,
            "total_swaps": self._total_swaps,
            "total_missing": total_missing,
        }

    def step(self, layers_per_step: int = 2) -> int:
        """Swap experts for the next N layers. All lazy -- no mx.eval().

        Constructs scatter graphs that get evaluated naturally by the next
        forward pass. Call between tokens in the generation loop.

        Args:
            layers_per_step: Number of layers to process per step.

        Returns:
            Number of layers processed in this step.
        """
        processed = 0
        for _ in range(layers_per_step):
            if not self._swap_queue:
                break

            plan = self._swap_queue.pop(0)
            cache = plan.cache

            st_map = getattr(self._model, "_st_map", None)
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                shard_path = cache._shard_paths[proj_name]
                key_prefix = cache._key_prefixes[proj_name]
                shard = mx.load(shard_path) if st_map is None else None
                new_w, new_s, new_b = _load_experts(key_prefix, plan.new_eids,
                                                     shard=shard, shard_map=cache._shard_map,
                                                     st_map=st_map)
                del shard

                w = cache.weights.pop(proj_name)
                w[plan.slot_indices] = new_w
                cache.weights[proj_name] = w

                s = cache.scales.pop(proj_name)
                s[plan.slot_indices] = new_s
                cache.scales[proj_name] = s

                if cache.biases[proj_name] is not None and new_b is not None:
                    b = cache.biases.pop(proj_name)
                    b[plan.slot_indices] = new_b
                    cache.biases[proj_name] = b

            for slot, old_eid, new_eid in plan.swaps:
                cache.cached_set.discard(old_eid)
                cache.cached_set.add(new_eid)
                cache.cached_ids[slot] = new_eid
                cache.frequency.pop(old_eid, None)
                cache.last_active.pop(old_eid, None)

            cache.rebuild_lookup()

            self._layers_done += 1
            self._swaps_done += len(plan.swaps)
            processed += 1

        return processed

    @property
    def is_complete(self) -> bool:
        return not self._swap_queue

    @property
    def remaining_layers(self) -> int:
        return len(self._swap_queue)

    @property
    def total_layers(self) -> int:
        return self._total_layers

    @property
    def progress(self) -> dict:
        return {
            "layers_done": self._layers_done,
            "layers_total": self._total_layers,
            "swaps_done": self._swaps_done,
            "swaps_total": self._total_swaps,
        }


def incremental_delta_warmup(model, tokenizer, model_path, new_prompt,
                              discovery_tokens: int = 10) -> tuple:
    """Create an IncrementalDeltaWarmup and run discovery.

    Args:
        model: The loaded MLX model with predictive modules.
        tokenizer: The model's tokenizer.
        model_path: Path to the model directory containing safetensors shards.
        new_prompt: Prompt text for discovery.
        discovery_tokens: Number of tokens to generate for discovery.

    Returns:
        Tuple of (warmup_object, discovery_stats).
    """
    warmup = IncrementalDeltaWarmup(model, tokenizer, model_path)
    stats = warmup.discover(new_prompt, tokens=discovery_tokens)
    return warmup, stats
