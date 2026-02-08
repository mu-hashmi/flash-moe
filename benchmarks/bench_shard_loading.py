"""Benchmark the shard loading phase of upgrade_to_predictive().

Instruments the shard loading (Pass 2) and tensor assembly (Pass 3) phases
separately to identify where time is spent during the upgrade.

Usage:
    PATH_REMOVED bench_shard_loading.py [capacity] [warmup_tokens]
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, _build_shard_map, PredictiveExpertCache,
    CachedQuantizedSwitchLinear, PredictiveCachedSwitchLinear,
    SyncPredictiveCachedSwitchLinear, get_cache_stats,
)

import numpy as np

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"


def timed_upgrade(model, model_path, capacity):
    """Replicate upgrade_to_predictive() with per-phase timing."""
    shard_map = _build_shard_map(model_path)

    # --- Pass 1: harvest LCP caches ---
    t0 = time.perf_counter()
    layer_meta = {}
    moe_idx = 0

    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
        first_proj = getattr(switch, "gate_proj")
        if not isinstance(first_proj, CachedQuantizedSwitchLinear):
            continue

        lcp_cache = first_proj._cache
        num_experts = 512
        C = min(capacity, num_experts)
        moe_idx += 1

        discovered = sorted(
            lcp_cache.entries.keys(),
            key=lambda eid: lcp_cache._priority(eid),
            reverse=True,
        )[:C]
        discovered_set = set(discovered)

        filler = []
        for eid in range(num_experts):
            if len(discovered) + len(filler) >= C:
                break
            if eid not in discovered_set:
                filler.append(eid)
        cached_ids = list(discovered) + filler

        pred_cache = PredictiveExpertCache(C, num_experts)
        harvested = {}
        to_load = {}
        has_bias = None
        phase2_mods = {}

        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mods[name] = getattr(switch, name)
            h_list = []
            load_list = []
            for slot, eid in enumerate(cached_ids):
                cached = lcp_cache.lookup(eid, name)
                if cached is not None:
                    w, s, b = cached
                    if has_bias is None:
                        has_bias = b is not None
                    h_list.append((slot, w, s, b))
                else:
                    load_list.append((slot, eid))
            harvested[name] = h_list
            to_load[name] = load_list

        layer_meta[i] = {
            "cached_ids": cached_ids,
            "pred_cache": pred_cache,
            "harvested": harvested,
            "to_load": to_load,
            "has_bias": has_bias if has_bias is not None else True,
            "phase2_mods": phase2_mods,
            "disc_count": len(discovered),
            "filler_count": len(filler),
            "C": C,
            "lcp_cache": lcp_cache,
        }

    t_harvest = time.perf_counter() - t0

    # --- Pass 2: shard loading ---
    t1 = time.perf_counter()
    shard_groups: dict[str, list[tuple]] = {}
    for i, meta in layer_meta.items():
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not meta["to_load"][name]:
                continue
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            shard_groups.setdefault(shard_path, []).append(
                (i, name, key_prefix, meta["to_load"][name]))

    loaded: dict[int, dict[str, dict[int, tuple]]] = {}
    shard_times = {}
    total_load_calls = 0
    total_eval_calls = 0

    for shard_path, group in shard_groups.items():
        ts = time.perf_counter()
        shard = mx.load(shard_path)
        t_load = time.perf_counter() - ts
        total_load_calls += 1

        layers_in_batch = sorted(set(layer_i for layer_i, _, _, _ in group))
        layer_eval_times = []

        for layer_i in layers_in_batch:
            layer_entries = [(n, kp, slots) for li, n, kp, slots in group if li == layer_i]
            to_eval = []
            for name, key_prefix, slot_eids in layer_entries:
                load_ids = mx.array([eid for _, eid in slot_eids])
                w_batch = shard[f"{key_prefix}.weight"][load_ids]
                s_batch = shard[f"{key_prefix}.scales"][load_ids]
                biases_key = f"{key_prefix}.biases"
                b_batch = shard[biases_key][load_ids] if biases_key in shard else None
                to_eval.extend([w_batch, s_batch])
                if b_batch is not None:
                    to_eval.append(b_batch)

                slot_map = {}
                for j, (slot, _) in enumerate(slot_eids):
                    slot_map[slot] = (w_batch[j], s_batch[j],
                                      b_batch[j] if b_batch is not None else None)
                loaded.setdefault(layer_i, {})[name] = slot_map

            te = time.perf_counter()
            mx.eval(*to_eval)
            layer_eval_times.append(time.perf_counter() - te)
            total_eval_calls += 1

        del shard

        shard_name = Path(shard_path).name
        shard_times[shard_name] = {
            "mx_load": t_load,
            "layers": len(layers_in_batch),
            "eval_times": layer_eval_times,
            "total_eval": sum(layer_eval_times),
        }

    t_shard_loading = time.perf_counter() - t1

    # --- Pass 3: tensor assembly ---
    t2 = time.perf_counter()
    upgraded = 0
    cls = PredictiveCachedSwitchLinear
    assembly_times = []

    for i, meta in layer_meta.items():
        ta = time.perf_counter()
        pred_cache = meta["pred_cache"]
        cached_ids = meta["cached_ids"]
        has_bias = meta["has_bias"]
        C = meta["C"]

        for name in ("gate_proj", "up_proj", "down_proj"):
            ws, ss, bs = [], [], []
            harvested_map = {slot: (w, s, b) for slot, w, s, b in meta["harvested"][name]}
            loaded_map = loaded.get(i, {}).get(name, {})

            for slot in range(C):
                if slot in harvested_map:
                    w, s, b = harvested_map[slot]
                else:
                    w, s, b = loaded_map[slot]
                ws.append(w)
                ss.append(s)
                if has_bias:
                    bs.append(b)

            pred_cache.weights[name] = mx.stack(ws)
            pred_cache.scales[name] = mx.stack(ss)
            pred_cache.biases[name] = mx.stack(bs) if has_bias else None

        for name in ("gate_proj", "up_proj", "down_proj"):
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            pred_cache._shard_paths[name] = shard_map[f"{key_prefix}.weight"]
            pred_cache._key_prefixes[name] = key_prefix

        pred_cache.build_lookup(cached_ids)
        mx.eval(pred_cache.lookup)

        switch = model.layers[i].mlp.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mod = meta["phase2_mods"][name]
            replacement = cls(
                group_size=phase2_mod.group_size,
                bits=phase2_mod.bits,
                mode=phase2_mod.mode,
                proj_name=name,
                cache=pred_cache,
            )
            setattr(switch, name, replacement)
            upgraded += 1

        meta["lcp_cache"].entries.clear()
        meta["lcp_cache"].frequency.clear()
        meta["lcp_cache"].last_active.clear()

        assembly_times.append(time.perf_counter() - ta)

    t_assembly = time.perf_counter() - t2

    # Count experts needing disk load vs harvested from LCP
    total_harvested = sum(
        len(meta["harvested"][n])
        for meta in layer_meta.values()
        for n in ("gate_proj", "up_proj", "down_proj")
    )
    total_disk = sum(
        len(meta["to_load"][n])
        for meta in layer_meta.values()
        for n in ("gate_proj", "up_proj", "down_proj")
    )

    return {
        "upgraded": upgraded,
        "harvest_time": t_harvest,
        "shard_loading_time": t_shard_loading,
        "assembly_time": t_assembly,
        "total_time": t_harvest + t_shard_loading + t_assembly,
        "shard_times": shard_times,
        "assembly_per_layer": assembly_times,
        "mx_load_calls": total_load_calls,
        "eval_calls": total_eval_calls,
        "unique_shards": len(shard_groups),
        "harvested_slots": total_harvested // 3,  # per-expert (3 projs each)
        "disk_loaded_slots": total_disk // 3,
    }


def main():
    capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    warmup_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")

    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)

    print(f"Enabling cached expert loading (capacity={capacity})...")
    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=capacity,
                                   predictive=True)
    print(f"Replaced {replaced} modules")

    print("Evaluating non-expert parameters...")
    mx.eval(model.parameters())
    mem_base = mx.get_active_memory() / 1e9
    print(f"Base memory: {mem_base:.1f} GB")

    prompt = "Write a hello world program in Python"
    print(f"\nWarmup: generating {warmup_tokens} tokens to discover experts...")
    mlx_lm.generate(model, tokenizer, prompt=prompt,
                    max_tokens=warmup_tokens, verbose=False)

    warmup_stats = get_cache_stats(model)
    print(f"Warmup hit rate: {warmup_stats['total_hit_rate']:.1%}")
    discovered_per_layer = [l["cached_experts"] for l in warmup_stats["layers"]]
    print(f"Discovered experts/layer: min={min(discovered_per_layer)} "
          f"median={sorted(discovered_per_layer)[len(discovered_per_layer)//2]} "
          f"max={max(discovered_per_layer)}")

    print(f"\n{'='*60}")
    print(f"BENCHMARKING: upgrade_to_predictive (capacity={capacity})")
    print(f"{'='*60}")

    results = timed_upgrade(model, model_path, capacity)

    print(f"\n--- Phase Timing ---")
    print(f"  Pass 1 (harvest LCP):     {results['harvest_time']:.3f}s")
    print(f"  Pass 2 (shard loading):   {results['shard_loading_time']:.3f}s")
    print(f"  Pass 3 (tensor assembly): {results['assembly_time']:.3f}s")
    print(f"  TOTAL:                    {results['total_time']:.3f}s")

    print(f"\n--- Shard Loading Detail ---")
    print(f"  Unique shards loaded: {results['unique_shards']}")
    print(f"  mx.load() calls: {results['mx_load_calls']}")
    print(f"  mx.eval() calls: {results['eval_calls']}")
    print(f"  Experts from LCP cache: {results['harvested_slots']}")
    print(f"  Experts from disk: {results['disk_loaded_slots']}")

    print(f"\n--- Per-Shard Timing ---")
    for shard_name, info in sorted(results["shard_times"].items()):
        print(f"  {shard_name}:")
        print(f"    mx.load(): {info['mx_load']:.3f}s")
        print(f"    layers: {info['layers']}")
        print(f"    eval total: {info['total_eval']:.3f}s")
        eval_strs = [f"{t:.3f}" for t in info["eval_times"]]
        print(f"    eval per-layer: [{', '.join(eval_strs)}]s")

    print(f"\n--- Per-Layer Assembly (Pass 3) ---")
    assembly = results["assembly_per_layer"]
    print(f"  Total layers: {len(assembly)}")
    print(f"  Min: {min(assembly):.3f}s  Median: {sorted(assembly)[len(assembly)//2]:.3f}s  "
          f"Max: {max(assembly):.3f}s")
    print(f"  Sum: {sum(assembly):.3f}s")

    mem_final = mx.get_active_memory() / 1e9
    print(f"\n--- Memory ---")
    print(f"  Base: {mem_base:.1f} GB")
    print(f"  After upgrade: {mem_final:.1f} GB")
    print(f"  Expert cache: {mem_final - mem_base:.1f} GB")

    print(f"\nUpgraded {results['upgraded']} modules")


if __name__ == "__main__":
    main()
