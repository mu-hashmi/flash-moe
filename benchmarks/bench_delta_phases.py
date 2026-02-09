"""Benchmark delta_warmup phase breakdown: shard I/O vs scatter updates.

Runs a full warmup cycle, then triggers a delta warmup on a different prompt,
timing each phase separately: discovery, shard loading, scatter updates, lookup rebuild.

Usage:
    PATH_REMOVED bench_delta_phases.py [capacity]
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, _build_shard_map,
    PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear,
    get_cache_stats, measure_fallback,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"


def timed_delta_warmup(model, tokenizer, model_path, new_prompt, discovery_tokens=10):
    """Delta warmup with per-phase timing breakdown."""
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    for layer in model.layers:
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache._indices_buffer.clear()

    # Phase 1: Discovery pass
    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=new_prompt,
                    max_tokens=discovery_tokens, verbose=False)
    t_discovery = time.perf_counter() - t0

    # Phase 2: Collect misses and plan swaps
    t1 = time.perf_counter()
    layer_info = {}

    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
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

    layer_swaps: dict[int, list[tuple]] = {}
    total_swaps = 0
    total_missing = 0

    for i, info in layer_info.items():
        missing = info["missing"]
        cold = list(info["cold"])
        total_missing += len(missing)
        if not missing:
            continue
        swaps = []
        for new_eid in sorted(missing):
            if not cold:
                break
            _, slot, old_eid = cold.pop(0)
            swaps.append((slot, old_eid, new_eid))
        if swaps:
            layer_swaps[i] = swaps
            total_swaps += len(swaps)

    t_planning = time.perf_counter() - t1

    # Phase 3: Shard loading
    t2 = time.perf_counter()
    shard_groups: dict[str, list[tuple]] = {}
    for i, swaps in layer_swaps.items():
        cache = layer_info[i]["cache"]
        new_eids = mx.array([new_eid for _, _, new_eid in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            sp = cache._shard_paths[proj_name]
            kp = cache._key_prefixes[proj_name]
            shard_groups.setdefault(sp, []).append((i, proj_name, kp, new_eids))

    loaded_experts: dict[int, dict[str, tuple]] = {}
    shard_load_times = {}

    for shard_path, group in shard_groups.items():
        ts = time.perf_counter()
        shard = mx.load(shard_path)
        t_load = time.perf_counter() - ts

        layers_in_batch = sorted(set(li for li, _, _, _ in group))
        eval_times = []

        for layer_i in layers_in_batch:
            layer_entries = [(pn, kp, eids) for li, pn, kp, eids in group if li == layer_i]
            to_eval = []
            for proj_name, key_prefix, new_eids in layer_entries:
                new_w = shard[f"{key_prefix}.weight"][new_eids]
                new_s = shard[f"{key_prefix}.scales"][new_eids]
                biases_key = f"{key_prefix}.biases"
                new_b = shard[biases_key][new_eids] if biases_key in shard else None
                loaded_experts.setdefault(layer_i, {})[proj_name] = (new_w, new_s, new_b)
                to_eval.extend([new_w, new_s])
                if new_b is not None:
                    to_eval.append(new_b)
            te = time.perf_counter()
            mx.eval(*to_eval)
            eval_times.append(time.perf_counter() - te)

        del shard
        shard_load_times[Path(shard_path).name] = {
            "mx_load": t_load,
            "layers": len(layers_in_batch),
            "eval_times": eval_times,
        }

    t_shard_io = time.perf_counter() - t2

    # Phase 4: Scatter updates
    t3 = time.perf_counter()
    scatter_times = []

    for i, swaps in layer_swaps.items():
        ts = time.perf_counter()
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

        lookup_np = np.zeros(cache.num_experts, dtype=np.int32)
        for slot, eid in enumerate(cache.cached_ids):
            lookup_np[eid] = slot
        cache.lookup = mx.array(lookup_np)

        to_eval = [cache.lookup]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            to_eval.append(cache.weights[proj_name])
            to_eval.append(cache.scales[proj_name])
            if cache.biases[proj_name] is not None:
                to_eval.append(cache.biases[proj_name])
        mx.eval(*to_eval)

        scatter_times.append(time.perf_counter() - ts)

    t_scatter = time.perf_counter() - t3

    # Cleanup
    for layer in model.layers:
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache.total_requests = 0
            proj._cache.total_fallbacks = 0
            proj._cache._indices_buffer.clear()
    mx.clear_cache()

    return {
        "discovery_time": t_discovery,
        "planning_time": t_planning,
        "shard_io_time": t_shard_io,
        "scatter_time": t_scatter,
        "total_time": t_discovery + t_planning + t_shard_io + t_scatter,
        "total_swaps": total_swaps,
        "total_missing": total_missing,
        "unique_shards": len(shard_groups),
        "shard_load_times": shard_load_times,
        "scatter_per_layer": scatter_times,
        "layers_with_swaps": len(layer_swaps),
    }


def main():
    capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 256

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
    print(f"Base memory: {mx.get_active_memory() / 1e9:.1f} GB")

    # Initial warmup + upgrade
    warmup_prompt = "Write a hello world program in Python"
    print(f"\nPhase 1: Warmup on default prompt ({10} tokens)...")
    mlx_lm.generate(model, tokenizer, prompt=warmup_prompt,
                    max_tokens=10, verbose=False)

    print(f"Upgrading to predictive cache (capacity={capacity})...")
    upgraded = upgrade_to_predictive(model, model_path, capacity)
    print(f"Upgraded {upgraded} modules")
    print(f"Memory after upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

    # Delta warmup on a different prompt
    delta_prompt = "Implement binary search in Rust with generics"
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: delta_warmup phases")
    print(f"{'='*60}")
    print(f"Delta prompt: {delta_prompt!r}")

    results = timed_delta_warmup(model, tokenizer, model_path, delta_prompt)

    print(f"\n--- Phase Timing ---")
    print(f"  Discovery ({10} tokens): {results['discovery_time']:.3f}s")
    print(f"  Planning:               {results['planning_time']:.3f}s")
    print(f"  Shard I/O:              {results['shard_io_time']:.3f}s")
    print(f"  Scatter updates:        {results['scatter_time']:.3f}s")
    print(f"  TOTAL:                  {results['total_time']:.3f}s")

    print(f"\n--- Swap Stats ---")
    print(f"  Total missing experts: {results['total_missing']}")
    print(f"  Total swaps: {results['total_swaps']}")
    print(f"  Layers with swaps: {results['layers_with_swaps']}")
    print(f"  Unique shards loaded: {results['unique_shards']}")

    if results["shard_load_times"]:
        print(f"\n--- Per-Shard I/O ---")
        for shard_name, info in sorted(results["shard_load_times"].items()):
            eval_strs = [f"{t:.3f}" for t in info["eval_times"]]
            print(f"  {shard_name}: mx.load={info['mx_load']:.3f}s, "
                  f"layers={info['layers']}, eval=[{', '.join(eval_strs)}]s")

    if results["scatter_per_layer"]:
        scatter = results["scatter_per_layer"]
        print(f"\n--- Scatter Updates ---")
        print(f"  Layers updated: {len(scatter)}")
        print(f"  Min: {min(scatter):.3f}s  Median: {sorted(scatter)[len(scatter)//2]:.3f}s  "
              f"Max: {max(scatter):.3f}s")
        print(f"  Sum: {sum(scatter):.3f}s")

    # Verify generation works after delta warmup
    print(f"\nVerifying generation after delta warmup...")
    response = mlx_lm.generate(model, tokenizer, prompt=delta_prompt,
                                max_tokens=50, verbose=True)
    fb = measure_fallback(model)
    print(f"\nFallback rate: {fb['fallback_rate']:.1%}")
    print(f"Final memory: {mx.get_active_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
