# Copyright © 2023-2025 Apple Inc.

import json
import datetime
from pathlib import Path

import numpy as np
import mlx.core as mx

from .loading import (
    _find_switch_mlp,
    _detect_num_experts,
    _build_shard_map,
    _load_proj_experts,
)
from .modules import (
    PredictiveCachedSwitchLinear,
    SyncPredictiveCachedSwitchLinear,
    CachedQuantizedSwitchLinear,
    PredictiveExpertCache,
)
from .core import upgrade_to_predictive


def save_cache_state(model, path: str | Path, metadata: dict | None = None) -> None:
    """Save discovered expert routing state to JSON for fast cold start.

    Captures per-layer expert IDs, frequencies, and LCP priorities from
    the current cache state (Phase 2 ExpertCache or Phase 3 PredictiveExpertCache).

    Args:
        model: The model with lazy expert modules installed.
        path: Output JSON file path.
        metadata: Optional metadata dict to include (e.g. prompt, capacity).
    """
    layers = {}
    capacity = None

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        proj = getattr(switch, "gate_proj", None)
        if proj is None:
            continue

        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            cache = proj._cache
            layers[str(i)] = {
                "cached_ids": list(cache.cached_ids),
                "frequency": {str(k): v for k, v in cache.frequency.items()},
                "last_active": {str(k): v for k, v in cache.last_active.items()},
                "step": cache.step,
                "all_seen": list(cache.cached_set),
            }
            if capacity is None:
                capacity = cache.capacity

        elif isinstance(proj, CachedQuantizedSwitchLinear):
            cache = proj._cache
            layers[str(i)] = {
                "cached_ids": sorted(cache.entries.keys()),
                "frequency": {str(k): v for k, v in cache.frequency.items()},
                "last_active": {str(k): v for k, v in cache.last_active.items()},
                "step": cache.step,
                "all_seen": sorted(cache.all_seen),
            }
            if capacity is None:
                capacity = cache.capacity

    state = {
        "version": 1,
        "capacity": capacity,
        "timestamp": datetime.datetime.now().isoformat(),
        "metadata": metadata or {},
        "layers": layers,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)


def load_cache_state(path: str | Path) -> dict:
    """Load saved expert routing state from JSON.

    Args:
        path: Path to the saved cache state JSON file.

    Returns:
        Dict with version, capacity, timestamp, metadata, and per-layer state.
    """
    with open(path) as f:
        state = json.load(f)
    if state.get("version") != 1:
        raise ValueError(f"Unsupported cache state version: {state.get('version')}")
    return state


def save_prepacked_weights(model, path: str | Path) -> None:
    """Save pre-stacked predictive cache weights to safetensors for fast warm start.

    After upgrade_to_predictive(), the PredictiveExpertCache objects have stacked
    weight/scale/bias tensors in Metal memory. Save them to disk so the next warm
    start can load_prepacked_weights() and skip the entire upgrade_to_predictive().

    Convention: cache at "foo.json" -> weights at "foo.weights.safetensors",
    metadata at "foo.weights.meta.json".

    Args:
        model: The model with predictive expert modules installed.
        path: Output safetensors file path.
    """
    path = Path(path)
    tensors: dict[str, mx.array] = {}
    meta_layers: dict[str, dict] = {}

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"layer.{i}.{proj_name}.weight"] = cache.weights[proj_name]
            tensors[f"layer.{i}.{proj_name}.scales"] = cache.scales[proj_name]
            if cache.biases[proj_name] is not None:
                tensors[f"layer.{i}.{proj_name}.biases"] = cache.biases[proj_name]

        meta_layers[str(i)] = {
            "cached_ids": list(cache.cached_ids),
            "num_experts": cache.num_experts,
            "pinned_set": sorted(cache.pinned_set),
            "frequency": {str(k): v for k, v in cache.frequency.items()},
            "last_active": {str(k): v for k, v in cache.last_active.items()},
            "step": cache.step,
        }

    mx.save_safetensors(str(path), tensors)

    meta_path = Path(str(path) + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump({"version": 1, "layers": meta_layers}, f)

    print(f"  Saved prepacked weights: {path} ({len(tensors)} tensors, "
          f"{len(meta_layers)} layers)")


def load_prepacked_weights(model, prepacked_path: str | Path,
                           model_path: str | Path | None = None) -> int:
    """Load pre-stacked predictive cache from safetensors, skipping upgrade_to_predictive().

    The model must already have Phase 2 modules installed (enable_lazy_experts with
    predictive=True). This replaces them with PredictiveCachedSwitchLinear using the
    pre-packed tensors directly.

    Args:
        model: The model with Phase 2 modules installed.
        prepacked_path: Path to the prepacked safetensors file.
        model_path: Optional model directory for shard map (enables dynamic updates).

    Returns:
        Number of modules upgraded.
    """
    prepacked_path = Path(prepacked_path)
    meta_path = Path(str(prepacked_path) + ".meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    packed = mx.load(str(prepacked_path))

    shard_map = None
    if model_path is not None:
        shard_map = _build_shard_map(Path(model_path))

    upgraded = 0
    layers_processed = 0
    batch_eval: list[mx.array] = []

    for layer_str, layer_meta in meta["layers"].items():
        i = int(layer_str)
        layer = model.layers[i]
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        cached_ids = layer_meta["cached_ids"]
        num_experts = layer_meta["num_experts"]
        capacity = len(cached_ids)

        pred_cache = PredictiveExpertCache(capacity, num_experts)

        phase2_mod = getattr(switch, "gate_proj")

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            w_key = f"layer.{i}.{proj_name}.weight"
            s_key = f"layer.{i}.{proj_name}.scales"
            b_key = f"layer.{i}.{proj_name}.biases"

            pred_cache.weights[proj_name] = packed[w_key]
            pred_cache.scales[proj_name] = packed[s_key]
            pred_cache.biases[proj_name] = packed[b_key] if b_key in packed else None

            batch_eval.append(pred_cache.weights[proj_name])
            batch_eval.append(pred_cache.scales[proj_name])
            if pred_cache.biases[proj_name] is not None:
                batch_eval.append(pred_cache.biases[proj_name])

        pred_cache.build_lookup(cached_ids)
        batch_eval.append(pred_cache.lookup)

        pred_cache.pinned_set = set(layer_meta.get("pinned_set", []))
        pred_cache.frequency = {int(k): v for k, v in layer_meta.get("frequency", {}).items()}
        pred_cache.last_active = {int(k): v for k, v in layer_meta.get("last_active", {}).items()}
        pred_cache.step = layer_meta.get("step", 0)

        if shard_map is not None and key_base is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                key_prefix = f"{key_base}.{proj_name}"
                pred_cache._shard_paths[proj_name] = shard_map[f"{key_prefix}.weight"]
                pred_cache._key_prefixes[proj_name] = key_prefix
            pred_cache._shard_map = shard_map
            pred_cache._st_map = getattr(model, "_st_map", None)

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            p2 = getattr(switch, proj_name)
            replacement = PredictiveCachedSwitchLinear(
                group_size=p2.group_size,
                bits=p2.bits,
                mode=p2.mode,
                proj_name=proj_name,
                cache=pred_cache,
            )
            setattr(switch, proj_name, replacement)
            upgraded += 1

        layers_processed += 1

        if layers_processed % 8 == 0:
            mx.eval(*batch_eval)
            batch_eval = []
            print(f"  Prepacked load: {layers_processed} layers "
                  f"({mx.get_active_memory() / 1e9:.1f} GB)")

    if batch_eval:
        mx.eval(*batch_eval)

    del packed

    print(f"  Prepacked load complete: {layers_processed} layers, {upgraded} modules "
          f"({mx.get_active_memory() / 1e9:.1f} GB)")
    return upgraded


def upgrade_from_saved_state(model, model_path: str | Path, cache_state: dict,
                             capacity: int, sync: bool = False) -> int:
    """Skip warmup: build predictive cache directly from saved state.

    Instead of: enable_lazy -> warmup gen -> upgrade_to_predictive
    Does: enable_lazy -> load saved state -> upgrade_to_predictive

    The model must already have Phase 2 modules installed via
    enable_lazy_experts(predictive=True). This populates the Phase 2 caches
    with frequency/last_active/all_seen from the saved state, then calls
    upgrade_to_predictive() which loads weights from disk.

    Args:
        model: The model with Phase 2 modules installed.
        model_path: Path to the model directory containing safetensors shards.
        cache_state: Dict from load_cache_state().
        capacity: Number of experts per layer.
        sync: If True, use SyncPredictiveCachedSwitchLinear.

    Returns:
        Number of modules upgraded.
    """
    num_moe_layers = sum(
        1 for layer in model.layers
        if _find_switch_mlp(layer)[0] is not None
    )
    expert_slot_mb = 1.69
    base_memory_gb = mx.get_active_memory() / 1e9
    projected_gb = base_memory_gb + capacity * num_moe_layers * expert_slot_mb / 1024
    device_gb = mx.device_info()["memory_size"] / 1e9
    limit_gb = 0.85 * device_gb

    if projected_gb > limit_gb:
        max_capacity = int((limit_gb - base_memory_gb) * 1024 / (num_moe_layers * expert_slot_mb))
        max_capacity = (max_capacity // 8) * 8
        max_capacity = max(max_capacity, 0)
        print(f"  [memory guard: {projected_gb:.1f} GB projected > {limit_gb:.1f} GB limit — "
              f"reducing capacity {capacity} -> {max_capacity}]")
        capacity = max_capacity

    layers_data = cache_state["layers"]

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue

        layer_key = str(i)
        if layer_key not in layers_data:
            continue

        saved = layers_data[layer_key]
        cache = proj._cache

        cache.frequency = {int(k): v for k, v in saved["frequency"].items()}
        cache.last_active = {int(k): v for k, v in saved["last_active"].items()}
        cache.step = saved["step"]
        cache.all_seen = set(saved["all_seen"])

        for eid in saved["all_seen"]:
            cache.entries[eid] = {}

    return upgrade_to_predictive(model, model_path, capacity, sync=sync)


def load_universal_profile(path: str | Path) -> dict:
    """Load universal expert profile from JSON.

    Args:
        path: Path to the profile JSON file (from profile_experts.py).

    Returns:
        Dict with num_prompts, layers, and per-layer activation_counts.
    """
    with open(path) as f:
        return json.load(f)


def upgrade_from_profile(
    model,
    model_path: str | Path,
    capacity: int,
    profile: dict,
    pin_threshold: float = 0.5,
    pin_top_k: int | None = None,
) -> int:
    """Profile-based cold start: skip discovery, use profile's top experts directly.

    When a universal expert profile exists, populates Phase 2 caches from the
    profile's activation counts and calls upgrade_to_predictive(). Experts above
    pin_threshold are marked pinned after upgrade.

    The model must already have Phase 2 modules installed (enable_lazy_experts
    with predictive=True).

    Args:
        model: The model with Phase 2 modules installed.
        model_path: Path to the model directory containing safetensors shards.
        capacity: Number of experts per layer.
        profile: Dict from load_universal_profile().
        pin_threshold: Minimum activation fraction to consider an expert universal
            when pin_top_k is None.
        pin_top_k: If set, pin exactly top-K activation-count experts per layer
            (after intersecting with cached experts). Use 0 for no pinning.

    Returns:
        Number of modules upgraded.
    """
    if pin_top_k is not None and pin_top_k < 0:
        raise ValueError(f"pin_top_k must be >= 0, got {pin_top_k}")

    model_path = Path(model_path)
    num_prompts = profile["num_prompts"]
    min_count = int(pin_threshold * num_prompts)

    moe_idx = 0
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue

        layer_data = profile["layers"].get(str(i))
        if layer_data is None:
            moe_idx += 1
            continue

        counts = layer_data.get("activation_counts", {})
        sorted_experts = sorted(
            ((int(eid), int(cnt)) for eid, cnt in counts.items()),
            key=lambda x: (-x[1], x[0]),
        )

        cache = proj._cache
        for eid, count in sorted_experts:
            cache.entries[eid] = {}
            cache.frequency[eid] = count
            cache.last_active[eid] = 1
            cache.all_seen.add(eid)
        cache.step = 1
        moe_idx += 1

    upgraded = upgrade_to_predictive(model, model_path, capacity)

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        layer_data = profile["layers"].get(str(i))
        if layer_data is None:
            continue

        counts = layer_data.get("activation_counts", {})
        if pin_top_k is None:
            pinned = set(
                int(eid) for eid, cnt in counts.items()
                if int(cnt) >= min_count
            )
        elif pin_top_k == 0:
            pinned = set()
        else:
            sorted_experts = sorted(
                ((int(eid), int(cnt)) for eid, cnt in counts.items()),
                key=lambda x: (-x[1], x[0]),
            )
            pinned = {eid for eid, _ in sorted_experts[:pin_top_k]}
        proj._cache.pinned_set = pinned & proj._cache.cached_set

    return upgraded
