"""Task 2: Map the Metal memory pressure cliff across capacities.

Tests [160, 176, 192, 208, 224, 240, 256] with and without cache_limit(0).
For each: memory, delta warmup timing, generation speed, output quality.
"""

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
CAPACITIES = [160, 176, 192, 208, 224, 240, 256]


def run_trial(capacity, use_cache_limit):
    label = f"cap={capacity}, cache_limit={'0' if use_cache_limit else 'default'}"
    print(f"\n{'='*60}")
    print(f"TRIAL: {label}")
    print(f"{'='*60}")

    model_path = hf_repo_to_path(MODEL)
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())

    # Warmup + upgrade
    mlx_lm.generate(model, tokenizer, prompt=PROMPT_EN, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)
    mem_after = mx.get_active_memory() / 1e9
    print(f"Memory after upgrade: {mem_after:.2f} GB")

    if use_cache_limit:
        mx.metal.set_cache_limit(0)
        mx.metal.clear_cache()

    # Delta warmup
    stats = fast_delta_warmup(model, tokenizer, model_path, PROMPT_CN)
    mem_delta = mx.get_active_memory() / 1e9
    print(f"Delta: discovery={stats['discovery_time']:.1f}s, "
          f"lookup/eval={stats['lookup_rebuild_time']:.1f}s, "
          f"total={stats['total_time']:.1f}s, "
          f"swaps={stats['total_swaps']}")

    if use_cache_limit:
        mx.metal.set_cache_limit(mx.metal.device_info()["memory_size"] // 4)

    # Generate
    t0 = time.perf_counter()
    out = mlx_lm.generate(model, tokenizer, prompt=PROMPT_CN, max_tokens=50, verbose=False)
    t_gen = time.perf_counter() - t0
    fb = get_fallback_stats(model)
    gen_speed = 50 / t_gen

    # Quality assessment
    out_preview = out[:200].replace('\n', ' ')
    print(f"Output: {out_preview}")
    print(f"Speed: {gen_speed:.1f} tok/s, Fallback: {fb['fallback_rate']:.1%}")

    mx.metal.clear_cache()

    return {
        "capacity": capacity,
        "cache_limit": use_cache_limit,
        "memory_gb": mem_after,
        "mem_after_delta": mem_delta,
        "discovery_time": stats["discovery_time"],
        "shard_load_time": stats["shard_load_time"],
        "scatter_time": stats["scatter_time"],
        "lookup_rebuild_time": stats["lookup_rebuild_time"],
        "total_time": stats["total_time"],
        "total_swaps": stats["total_swaps"],
        "total_missing": stats["total_missing"],
        "gen_speed": gen_speed,
        "fallback_rate": fb["fallback_rate"],
        "output_preview": out_preview,
    }


def main():
    results = []
    for capacity in CAPACITIES:
        # Default mode
        r = run_trial(capacity, use_cache_limit=False)
        results.append(r)
        # With cache_limit(0)
        r2 = run_trial(capacity, use_cache_limit=True)
        results.append(r2)

    # Print summary
    print(f"\n{'='*70}")
    print("PRESSURE CLIFF MAPPING — Default cache")
    print(f"{'='*70}")
    print(f"{'Cap':>4} | {'Memory':>7} | {'Discovery':>9} | {'Eval':>6} | {'Total':>7} | {'Swaps':>5} | {'tok/s':>5} | {'FB':>5}")
    print("-" * 65)
    for r in results:
        if not r["cache_limit"]:
            print(f"{r['capacity']:>4} | {r['memory_gb']:>6.2f}G | {r['discovery_time']:>8.1f}s | {r['lookup_rebuild_time']:>5.1f}s | {r['total_time']:>6.1f}s | {r['total_swaps']:>5} | {r['gen_speed']:>5.1f} | {r['fallback_rate']:>4.1%}")

    print(f"\n{'='*70}")
    print("PRESSURE CLIFF MAPPING — cache_limit(0)")
    print(f"{'='*70}")
    print(f"{'Cap':>4} | {'Memory':>7} | {'Discovery':>9} | {'Eval':>6} | {'Total':>7} | {'Swaps':>5} | {'tok/s':>5} | {'FB':>5}")
    print("-" * 65)
    for r in results:
        if r["cache_limit"]:
            print(f"{r['capacity']:>4} | {r['memory_gb']:>6.2f}G | {r['discovery_time']:>8.1f}s | {r['lookup_rebuild_time']:>5.1f}s | {r['total_time']:>6.1f}s | {r['total_swaps']:>5} | {r['gen_speed']:>5.1f} | {r['fallback_rate']:>4.1%}")

    # Append to final_validation.md
    with open("PATH_REMOVED", "a") as f:
        f.write("\n## 2. Metal Memory Pressure Cliff Mapping\n\n")
        f.write("Cross-domain delta warmup (English coding → Chinese poetry) across capacities.\n")
        f.write("Each capacity loaded fresh. Two variants: default MLX cache and cache_limit(0).\n\n")

        f.write("### Default Cache\n\n")
        f.write("| Capacity | Memory (GB) | Discovery (s) | Shard I/O (s) | Scatter (s) | Eval (s) | Delta Total (s) | Gen (tok/s) | Fallback | Quality |\n")
        f.write("|----------|-------------|--------------|--------------|------------|---------|----------------|------------|----------|----------|\n")
        for r in results:
            if not r["cache_limit"]:
                quality = "coherent" if r["gen_speed"] > 2 and r["fallback_rate"] < 0.1 else "degraded"
                f.write(f"| {r['capacity']} | {r['memory_gb']:.2f} | {r['discovery_time']:.1f} | {r['shard_load_time']:.1f} | {r['scatter_time']:.3f} | {r['lookup_rebuild_time']:.1f} | {r['total_time']:.1f} | {r['gen_speed']:.1f} | {r['fallback_rate']:.1%} | {quality} |\n")

        f.write("\n### With cache_limit(0)\n\n")
        f.write("| Capacity | Memory (GB) | Discovery (s) | Shard I/O (s) | Scatter (s) | Eval (s) | Delta Total (s) | Gen (tok/s) | Fallback | Quality |\n")
        f.write("|----------|-------------|--------------|--------------|------------|---------|----------------|------------|----------|----------|\n")
        for r in results:
            if r["cache_limit"]:
                quality = "coherent" if r["gen_speed"] > 2 and r["fallback_rate"] < 0.1 else "degraded"
                f.write(f"| {r['capacity']} | {r['memory_gb']:.2f} | {r['discovery_time']:.1f} | {r['shard_load_time']:.1f} | {r['scatter_time']:.3f} | {r['lookup_rebuild_time']:.1f} | {r['total_time']:.1f} | {r['gen_speed']:.1f} | {r['fallback_rate']:.1%} | {quality} |\n")

        # Analysis
        f.write("\n### Output Samples\n\n")
        for r in results:
            if not r["cache_limit"]:
                f.write(f"**Capacity {r['capacity']}** ({r['memory_gb']:.1f} GB): `{r['output_preview'][:100]}...`\n\n")

        # Find cliff
        defaults = [r for r in results if not r["cache_limit"]]
        f.write("### Analysis\n\n")

        # Find where discovery time jumps
        for i in range(1, len(defaults)):
            prev, curr = defaults[i-1], defaults[i]
            ratio = curr["discovery_time"] / prev["discovery_time"] if prev["discovery_time"] > 0 else 0
            if ratio > 1.5:
                f.write(f"**Pressure cliff detected:** Between capacity {prev['capacity']} ({prev['memory_gb']:.1f} GB) and {curr['capacity']} ({curr['memory_gb']:.1f} GB). ")
                f.write(f"Discovery time jumps {ratio:.1f}x ({prev['discovery_time']:.1f}s → {curr['discovery_time']:.1f}s).\n\n")

        f.write("**Recommendation:** The optimal capacity is the highest one that stays under the pressure cliff while producing coherent output.\n\n")

    print("\nResults appended to PATH_REMOVED")


if __name__ == "__main__":
    main()
