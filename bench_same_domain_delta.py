import sys, time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, fast_delta_warmup,
    get_fallback_stats, reset_to_cached,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"

SAME_DOMAIN_PAIRS = [
    ("Write a Python function for binary search", "Write a Python function for merge sort"),
    ("Explain how TCP works", "Explain how HTTP works"),
    ("Debug this JavaScript: function foo() { return bar; }", "Write a React component for a todo list"),
]

CROSS_DOMAIN_PAIR = ("Write a Python function for binary search", "用中文写一首关于春天的诗")


def run_delta_test(model, tokenizer, model_path, prompt_a, prompt_b, label, capacity=256):
    """Warmup on prompt_a, delta to prompt_b, generate."""

    # Reset to fresh state
    reset_to_cached(model, model_path, capacity)

    # Warmup on prompt A
    mlx_lm.generate(model, tokenizer, prompt=prompt_a, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)

    # Delta warmup to prompt B
    stats = fast_delta_warmup(model, tokenizer, model_path, prompt_b)

    # Generate on prompt B
    t0 = time.perf_counter()
    out = mlx_lm.generate(model, tokenizer, prompt=prompt_b, max_tokens=50, verbose=False)
    t_gen = time.perf_counter() - t0
    fb = get_fallback_stats(model)

    miss_dist = [s['missing'] for s in stats['per_layer']]
    layers_zero = sum(1 for m in miss_dist if m == 0)
    layers_lt5 = sum(1 for m in miss_dist if 0 < m < 5)
    layers_gte5 = sum(1 for m in miss_dist if m >= 5)

    print(f"\n--- {label} ---")
    print(f"  A: {prompt_a[:50]}...")
    print(f"  B: {prompt_b[:50]}...")
    print(f"  Delta: {stats['total_time']:.1f}s (disc {stats['discovery_time']:.1f}s + rebuild {stats['rebuild_time']:.1f}s)")
    print(f"  Shard I/O: {stats['shard_load_time']:.1f}s, Scatter eval: {stats['lookup_rebuild_time']:.1f}s")
    print(f"  Swaps: {stats['total_swaps']} ({stats['total_missing']} missing)")
    print(f"  Layer miss distribution: {layers_zero} zero, {layers_lt5} (1-4), {layers_gte5} (5+)")
    print(f"  Miss range: {min(miss_dist)}-{max(miss_dist)}, median={sorted(miss_dist)[len(miss_dist)//2]}")
    print(f"  Gen speed: {50/t_gen:.1f} tok/s, Fallback: {fb['fallback_rate']:.1%}")
    print(f"  Output: {out[:80]}...")

    # Test with skip-layers thresholds
    for threshold in (3, 5):
        reset_to_cached(model, model_path, capacity)
        mlx_lm.generate(model, tokenizer, prompt=prompt_a, max_tokens=10, verbose=False)
        upgrade_to_predictive(model, model_path, capacity)
        stats_skip = fast_delta_warmup(model, tokenizer, model_path, prompt_b,
                                       min_swaps_threshold=threshold)
        t0 = time.perf_counter()
        out_skip = mlx_lm.generate(model, tokenizer, prompt=prompt_b, max_tokens=50, verbose=False)
        t_gen_skip = time.perf_counter() - t0
        fb_skip = get_fallback_stats(model)
        print(f"  skip_threshold={threshold}: {stats_skip['total_time']:.1f}s, "
              f"{stats_skip['layers_skipped']} layers skipped, "
              f"{50/t_gen_skip:.1f} tok/s, {fb_skip['fallback_rate']:.1%} fallback")


if __name__ == "__main__":
    capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    model_path = hf_repo_to_path(MODEL)
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())
    print(f"Base memory: {mx.get_active_memory() / 1e9:.2f} GB")

    # Cross-domain baseline
    run_delta_test(model, tokenizer, model_path, *CROSS_DOMAIN_PAIR, "CROSS-DOMAIN", capacity)

    # Same-domain pairs
    for i, (a, b) in enumerate(SAME_DOMAIN_PAIRS):
        run_delta_test(model, tokenizer, model_path, a, b, f"SAME-DOMAIN #{i+1}", capacity)
