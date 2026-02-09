"""Benchmark batched shard loading vs original per-layer loading.

Tests:
1. upgrade_to_predictive timing (batched)
2. delta_warmup timing (batched)
3. Correctness: generate text and verify coherent output

Usage:
    PATH_REMOVED bench_batched_loading.py
"""

import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_cache_stats,
    measure_fallback, delta_warmup, reset_to_cached,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 256
WARMUP_TOKENS = 10
GEN_TOKENS = 50


def load_model():
    model_path = hf_repo_to_path(MODEL)
    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=CAPACITY,
                                   predictive=True)
    print(f"Replaced {replaced} expert modules")
    mx.eval(model.parameters())
    print(f"Non-expert params: {mx.get_active_memory() / 1e9:.2f} GB")
    return model, tokenizer, model_path


def bench_upgrade(model, tokenizer, model_path, prompt):
    """Benchmark upgrade_to_predictive (batched shard loading)."""
    print(f"\n--- Warmup: {WARMUP_TOKENS} tokens ---")
    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=prompt,
                     max_tokens=WARMUP_TOKENS, verbose=False)
    t_warmup = time.perf_counter() - t0
    print(f"Warmup: {t_warmup:.2f}s")

    stats = get_cache_stats(model)
    print(f"Warmup hit rate: {stats['total_hit_rate']:.1%}")

    print(f"\n--- upgrade_to_predictive (batched) ---")
    t0 = time.perf_counter()
    upgraded = upgrade_to_predictive(model, model_path, CAPACITY)
    t_upgrade = time.perf_counter() - t0
    print(f"Upgraded {upgraded} modules in {t_upgrade:.2f}s")
    print(f"Memory after upgrade: {mx.get_active_memory() / 1e9:.2f} GB")
    return t_upgrade


def bench_generation(model, tokenizer, prompt):
    """Generate tokens and measure speed + correctness."""
    print(f"\n--- Generation: {GEN_TOKENS} tokens ---")
    t0 = time.perf_counter()
    output = mlx_lm.generate(model, tokenizer, prompt=prompt,
                              max_tokens=GEN_TOKENS, verbose=True)
    t_gen = time.perf_counter() - t0

    fb = measure_fallback(model)
    print(f"\nGeneration: {GEN_TOKENS} tokens in {t_gen:.2f}s "
          f"({GEN_TOKENS / t_gen:.1f} tok/s)")
    print(f"Fallback rate: {fb['fallback_rate']:.1%} "
          f"({fb['total_fallbacks']}/{fb['total_requests']})")
    print(f"Memory: {mx.get_active_memory() / 1e9:.2f} GB")
    return t_gen, fb


def bench_delta_warmup(model, tokenizer, model_path, new_prompt):
    """Benchmark delta_warmup (batched shard loading)."""
    print(f"\n--- delta_warmup (batched) ---")
    stats = delta_warmup(model, tokenizer, model_path, new_prompt)
    print(f"  Discovery: {stats['discovery_time']:.2f}s")
    print(f"  Rebuild: {stats['rebuild_time']:.2f}s")
    print(f"  Total: {stats['total_time']:.2f}s")
    print(f"  Swaps: {stats['total_swaps']} ({stats['total_missing']} missing)")
    return stats


def main():
    prompt1 = "Write a binary search implementation in Python with detailed comments"
    prompt2 = "Explain the theory of general relativity in simple terms"

    model, tokenizer, model_path = load_model()

    # Benchmark 1: upgrade_to_predictive
    t_upgrade = bench_upgrade(model, tokenizer, model_path, prompt1)

    # Benchmark 2: generation quality
    t_gen, fb = bench_generation(model, tokenizer, prompt1)

    # Benchmark 3: delta_warmup on different prompt
    delta_stats = bench_delta_warmup(model, tokenizer, model_path, prompt2)

    # Generate after delta warmup to verify correctness
    t_gen2, fb2 = bench_generation(model, tokenizer, prompt2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"upgrade_to_predictive: {t_upgrade:.2f}s")
    print(f"  (target: < 14s, previous: ~14s)")
    print(f"Generation 1: {GEN_TOKENS / t_gen:.1f} tok/s, fallback {fb['fallback_rate']:.1%}")
    print(f"delta_warmup: {delta_stats['total_time']:.2f}s "
          f"(discovery {delta_stats['discovery_time']:.2f}s + "
          f"rebuild {delta_stats['rebuild_time']:.2f}s)")
    print(f"  (target: < 48s rebuild, previous: ~57s rebuild)")
    print(f"Generation 2: {GEN_TOKENS / t_gen2:.1f} tok/s, fallback {fb2['fallback_rate']:.1%}")
    print(f"Peak memory: {mx.get_active_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
