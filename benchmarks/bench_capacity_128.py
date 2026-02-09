import sys, time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, fast_delta_warmup,
    get_fallback_stats,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"


def run_benchmark(capacity):
    model_path = hf_repo_to_path(MODEL)

    print(f"\n{'='*60}")
    print(f"CAPACITY = {capacity}")
    print(f"{'='*60}")

    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())
    print(f"Base memory: {mx.get_active_memory() / 1e9:.2f} GB")

    # Phase 1: Warmup + upgrade on English prompt
    prompt_en = "Write a Python function for binary search"
    print(f"\nWarmup on: {prompt_en[:50]}...")
    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=prompt_en, max_tokens=10, verbose=False)
    t_warmup_gen = time.perf_counter() - t0

    t0 = time.perf_counter()
    upgrade_to_predictive(model, model_path, capacity)
    t_upgrade = time.perf_counter() - t0
    mem_after = mx.get_active_memory() / 1e9
    print(f"Warmup gen: {t_warmup_gen:.1f}s, Upgrade: {t_upgrade:.1f}s, Memory: {mem_after:.2f} GB")

    # Phase 2: Generate on same prompt
    print(f"\nGenerate 50 tokens (same prompt)...")
    t0 = time.perf_counter()
    out = mlx_lm.generate(model, tokenizer, prompt=prompt_en, max_tokens=50, verbose=False)
    t_gen = time.perf_counter() - t0
    fb = get_fallback_stats(model)
    print(f"Output: {out[:100]}...")
    print(f"Time: {t_gen:.1f}s, Speed: {50/t_gen:.1f} tok/s")
    print(f"Fallback: {fb['fallback_rate']:.1%} ({fb['total_fallbacks']}/{fb['total_requests']})")

    # Phase 3: Cross-domain delta warmup (English -> Chinese)
    prompt_cn = "用中文写一首关于春天的诗"
    print(f"\nCross-domain delta warmup: -> Chinese")
    stats = fast_delta_warmup(model, tokenizer, model_path, prompt_cn)
    print(f"  Discovery: {stats['discovery_time']:.2f}s")
    print(f"  Rebuild: {stats['rebuild_time']:.2f}s")
    print(f"  Shard I/O: {stats['shard_load_time']:.2f}s")
    print(f"  Scatter eval: {stats['lookup_rebuild_time']:.2f}s")
    print(f"  Total: {stats['total_time']:.2f}s")
    print(f"  Swaps: {stats['total_swaps']} ({stats['total_missing']} missing)")
    print(f"  Memory: {mx.get_active_memory() / 1e9:.2f} GB")

    # Phase 4: Generate after cross-domain delta
    print(f"\nGenerate 50 tokens after cross-domain delta...")
    t0 = time.perf_counter()
    out_cn = mlx_lm.generate(model, tokenizer, prompt=prompt_cn, max_tokens=50, verbose=False)
    t_gen_cn = time.perf_counter() - t0
    fb_cn = get_fallback_stats(model)
    print(f"Output: {out_cn[:100]}...")
    print(f"Time: {t_gen_cn:.1f}s, Speed: {50/t_gen_cn:.1f} tok/s")
    print(f"Fallback: {fb_cn['fallback_rate']:.1%}")

    # Phase 5: Same-domain delta warmup (Chinese -> English coding, different topic)
    prompt_en2 = "Write a Python function for merge sort"
    print(f"\nSame-domain delta warmup: -> merge sort")
    stats2 = fast_delta_warmup(model, tokenizer, model_path, prompt_en2)
    print(f"  Discovery: {stats2['discovery_time']:.2f}s")
    print(f"  Rebuild: {stats2['rebuild_time']:.2f}s")
    print(f"  Shard I/O: {stats2['shard_load_time']:.2f}s")
    print(f"  Scatter eval: {stats2['lookup_rebuild_time']:.2f}s")
    print(f"  Total: {stats2['total_time']:.2f}s")
    print(f"  Swaps: {stats2['total_swaps']} ({stats2['total_missing']} missing)")

    # Phase 6: Generate after same-domain delta
    print(f"\nGenerate 50 tokens after same-domain delta...")
    t0 = time.perf_counter()
    out_en2 = mlx_lm.generate(model, tokenizer, prompt=prompt_en2, max_tokens=50, verbose=False)
    t_gen_en2 = time.perf_counter() - t0
    fb_en2 = get_fallback_stats(model)
    print(f"Output: {out_en2[:100]}...")
    print(f"Time: {t_gen_en2:.1f}s, Speed: {50/t_gen_en2:.1f} tok/s")
    print(f"Fallback: {fb_en2['fallback_rate']:.1%}")

    # Per-layer miss distribution for cross-domain delta
    miss_dist = [s['missing'] for s in stats['per_layer']]
    print(f"\nCross-domain per-layer misses: min={min(miss_dist)} median={sorted(miss_dist)[len(miss_dist)//2]} max={max(miss_dist)}")

    miss_dist2 = [s['missing'] for s in stats2['per_layer']]
    print(f"Chinese->English per-layer misses: min={min(miss_dist2)} median={sorted(miss_dist2)[len(miss_dist2)//2]} max={max(miss_dist2)}")

    print(f"\n{'='*60}")
    print(f"SUMMARY (capacity={capacity})")
    print(f"{'='*60}")
    print(f"Memory after warmup: {mem_after:.2f} GB")
    print(f"Same-prompt gen: {50/t_gen:.1f} tok/s, {fb['fallback_rate']:.1%} fallback")
    print(f"Cross-domain delta: {stats['total_time']:.1f}s, {stats['total_swaps']} swaps")
    print(f"Post-cross gen: {50/t_gen_cn:.1f} tok/s, {fb_cn['fallback_rate']:.1%} fallback")
    print(f"Same-domain delta: {stats2['total_time']:.1f}s, {stats2['total_swaps']} swaps")
    print(f"Post-same gen: {50/t_gen_en2:.1f} tok/s, {fb_en2['fallback_rate']:.1%} fallback")

    return {
        "capacity": capacity,
        "memory_gb": mem_after,
        "warmup_time": t_warmup_gen + t_upgrade,
        "gen_speed": 50 / t_gen,
        "fallback_same": fb['fallback_rate'],
        "delta_cross_time": stats['total_time'],
        "delta_cross_swaps": stats['total_swaps'],
        "gen_speed_after_cross": 50 / t_gen_cn,
        "fallback_after_cross": fb_cn['fallback_rate'],
        "delta_same_time": stats2['total_time'],
        "delta_same_swaps": stats2['total_swaps'],
        "gen_speed_after_same": 50 / t_gen_en2,
        "fallback_after_same": fb_en2['fallback_rate'],
    }


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    run_benchmark(cap)
