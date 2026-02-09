"""Task 1: Test whether mx.metal.set_cache_limit(0) before delta warmup helps performance.

Compares fast_delta_warmup with and without cache limit at capacity 224 and 256.
"""

import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, fast_delta_warmup,
    get_fallback_stats,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
PROMPT_EN = "Write a Python function for binary search"
PROMPT_CN = "用中文写一首关于春天的诗"


def run_trial(capacity, use_cache_limit):
    label = f"capacity={capacity}, cache_limit={'0' if use_cache_limit else 'default'}"
    print(f"\n{'='*60}")
    print(f"TRIAL: {label}")
    print(f"{'='*60}")

    model_path = hf_repo_to_path(MODEL)
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())
    base_mem = mx.get_active_memory() / 1e9
    print(f"Base memory: {base_mem:.2f} GB")

    # Warmup + upgrade
    print("Warmup generation...")
    mlx_lm.generate(model, tokenizer, prompt=PROMPT_EN, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)
    mem_after_upgrade = mx.get_active_memory() / 1e9
    print(f"Memory after upgrade: {mem_after_upgrade:.2f} GB")

    # Optionally set cache limit to 0
    if use_cache_limit:
        print("Setting mx.metal.set_cache_limit(0)...")
        mx.metal.set_cache_limit(0)
        mx.metal.clear_cache()
        mem_after_clear = mx.get_active_memory() / 1e9
        print(f"Memory after cache clear: {mem_after_clear:.2f} GB")

    # Delta warmup
    print(f"Running fast_delta_warmup (English -> Chinese)...")
    stats = fast_delta_warmup(model, tokenizer, model_path, PROMPT_CN)
    mem_after_delta = mx.get_active_memory() / 1e9

    print(f"  Discovery: {stats['discovery_time']:.2f}s")
    print(f"  Shard I/O: {stats['shard_load_time']:.2f}s")
    print(f"  Scatter: {stats['scatter_time']:.3f}s")
    print(f"  Lookup rebuild (eval): {stats['lookup_rebuild_time']:.2f}s")
    print(f"  Total: {stats['total_time']:.2f}s")
    print(f"  Swaps: {stats['total_swaps']} ({stats['total_missing']} missing)")
    print(f"  Memory after delta: {mem_after_delta:.2f} GB")

    # Restore cache limit if we changed it
    if use_cache_limit:
        mx.metal.set_cache_limit(mx.metal.device_info()["memory_size"] // 4)

    # Generate to verify quality
    print("Generating 50 tokens after delta...")
    t0 = time.perf_counter()
    out = mlx_lm.generate(model, tokenizer, prompt=PROMPT_CN, max_tokens=50, verbose=False)
    t_gen = time.perf_counter() - t0
    fb = get_fallback_stats(model)
    print(f"  Output: {out[:150]}...")
    print(f"  Speed: {50/t_gen:.1f} tok/s")
    print(f"  Fallback: {fb['fallback_rate']:.1%}")

    return {
        "capacity": capacity,
        "cache_limit": use_cache_limit,
        "mem_after_upgrade": mem_after_upgrade,
        "mem_after_delta": mem_after_delta,
        "discovery_time": stats["discovery_time"],
        "shard_load_time": stats["shard_load_time"],
        "scatter_time": stats["scatter_time"],
        "lookup_rebuild_time": stats["lookup_rebuild_time"],
        "total_time": stats["total_time"],
        "total_swaps": stats["total_swaps"],
        "gen_speed": 50 / t_gen,
        "fallback_rate": fb["fallback_rate"],
    }


def main():
    results = []
    for capacity in [224, 256]:
        for use_cache_limit in [False, True]:
            r = run_trial(capacity, use_cache_limit)
            results.append(r)
            # Force cleanup between trials
            mx.metal.clear_cache()

    print(f"\n{'='*60}")
    print("SUMMARY: mx.metal.set_cache_limit(0) experiment")
    print(f"{'='*60}")
    print(f"{'Capacity':>8} | {'Cache Limit':>11} | {'Mem Upgrade':>10} | {'Mem Delta':>9} | {'Discovery':>9} | {'Lookup/Eval':>11} | {'Delta Total':>11} | {'Gen tok/s':>9} | {'Fallback':>8}")
    print("-" * 110)
    for r in results:
        cl = "0" if r["cache_limit"] else "default"
        print(f"{r['capacity']:>8} | {cl:>11} | {r['mem_after_upgrade']:>9.2f}G | {r['mem_after_delta']:>8.2f}G | {r['discovery_time']:>8.2f}s | {r['lookup_rebuild_time']:>10.2f}s | {r['total_time']:>10.2f}s | {r['gen_speed']:>8.1f} | {r['fallback_rate']:>7.1%}")

    # Write results to final_validation.md
    with open("PATH_REMOVED", "w") as f:
        f.write("# MLX-MoE Final Validation Results\n\n")
        f.write("## 1. mx.metal.set_cache_limit(0) Experiment\n\n")
        f.write("**Question:** Does clearing the MLX buffer cache before delta warmup reduce memory pressure and speed up scatter eval?\n\n")
        f.write("| Capacity | Cache Limit | Mem After Upgrade (GB) | Mem After Delta (GB) | Discovery (s) | Shard I/O (s) | Scatter (s) | Lookup/Eval (s) | Delta Total (s) | Gen (tok/s) | Fallback |\n")
        f.write("|----------|-------------|----------------------|---------------------|--------------|--------------|------------|----------------|----------------|------------|----------|\n")
        for r in results:
            cl = "0" if r["cache_limit"] else "default"
            f.write(f"| {r['capacity']} | {cl} | {r['mem_after_upgrade']:.2f} | {r['mem_after_delta']:.2f} | {r['discovery_time']:.2f} | {r['shard_load_time']:.2f} | {r['scatter_time']:.3f} | {r['lookup_rebuild_time']:.2f} | {r['total_time']:.2f} | {r['gen_speed']:.1f} | {r['fallback_rate']:.1%} |\n")
        f.write("\n")

        # Analysis
        for cap in [224, 256]:
            default = next(r for r in results if r["capacity"] == cap and not r["cache_limit"])
            cleared = next(r for r in results if r["capacity"] == cap and r["cache_limit"])
            speedup = default["lookup_rebuild_time"] / cleared["lookup_rebuild_time"] if cleared["lookup_rebuild_time"] > 0 else float("inf")
            total_speedup = default["total_time"] / cleared["total_time"] if cleared["total_time"] > 0 else float("inf")
            mem_saved = default["mem_after_delta"] - cleared["mem_after_delta"]
            f.write(f"**Capacity {cap}:** cache_limit(0) {'helped' if total_speedup > 1.1 else 'did not help significantly'}.\n")
            f.write(f"- Lookup/eval: {default['lookup_rebuild_time']:.2f}s → {cleared['lookup_rebuild_time']:.2f}s ({speedup:.2f}x)\n")
            f.write(f"- Total delta: {default['total_time']:.2f}s → {cleared['total_time']:.2f}s ({total_speedup:.2f}x)\n")
            f.write(f"- Memory delta: {mem_saved:+.2f} GB\n\n")

    print("\nResults written to PATH_REMOVED")


if __name__ == "__main__":
    main()
