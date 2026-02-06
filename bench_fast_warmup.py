"""Benchmark fast_delta_warmup() vs delta_warmup() for cross-prompt switching.

Tests:
1. Load model, warmup on "Write a Python binary search"
2. Upgrade to predictive cache
3. Run fast_delta_warmup() for cross-prompt switch to Chinese poetry prompt
4. Time: discovery, shard loading, scatter update, total
5. Generate 50 tokens to verify output quality
6. Compare against current delta_warmup() time
7. Report fallback rate

Usage:
    PATH_REMOVED bench_fast_warmup.py
"""

import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_cache_stats,
    measure_fallback, delta_warmup, fast_delta_warmup, reset_to_cached,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 256
WARMUP_TOKENS = 10
GEN_TOKENS = 50

PROMPT1 = "Write a Python function to implement binary search on a sorted list"
PROMPT2 = "用中文写一首关于春天的诗"


def setup_model():
    model_path = hf_repo_to_path(MODEL)
    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=CAPACITY,
                                   predictive=True)
    print(f"Replaced {replaced} expert modules")
    mx.eval(model.parameters())
    print(f"Base memory: {mx.get_active_memory() / 1e9:.2f} GB")
    return model, tokenizer, model_path


def initial_warmup(model, tokenizer, model_path, prompt):
    print(f"\nLCP warmup ({WARMUP_TOKENS} tokens)...")
    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=prompt,
                    max_tokens=WARMUP_TOKENS, verbose=False)
    t_warmup = time.perf_counter() - t0
    print(f"  Warmup: {t_warmup:.2f}s")

    stats = get_cache_stats(model)
    print(f"  Hit rate: {stats['total_hit_rate']:.1%}")

    print(f"\nUpgrading to predictive cache (capacity={CAPACITY})...")
    t0 = time.perf_counter()
    upgraded = upgrade_to_predictive(model, model_path, CAPACITY)
    t_upgrade = time.perf_counter() - t0
    print(f"  Upgraded {upgraded} modules in {t_upgrade:.2f}s")
    print(f"  Memory: {mx.get_active_memory() / 1e9:.2f} GB")


def generate_and_measure(model, tokenizer, prompt, label):
    print(f"\n--- {label}: {GEN_TOKENS} tokens ---")
    t0 = time.perf_counter()
    output = mlx_lm.generate(model, tokenizer, prompt=prompt,
                              max_tokens=GEN_TOKENS, verbose=True)
    t_gen = time.perf_counter() - t0

    fb = measure_fallback(model)
    tok_s = GEN_TOKENS / t_gen
    print(f"\n  {tok_s:.1f} tok/s, fallback {fb['fallback_rate']:.1%} "
          f"({fb['total_fallbacks']}/{fb['total_requests']})")
    print(f"  Memory: {mx.get_active_memory() / 1e9:.2f} GB")
    return tok_s, fb


def main():
    model, tokenizer, model_path = setup_model()

    # Initial warmup on prompt1
    initial_warmup(model, tokenizer, model_path, PROMPT1)

    # Verify generation works on prompt1
    tok_s1, fb1 = generate_and_measure(model, tokenizer, PROMPT1,
                                        "Generation (original prompt)")

    # Test 1: fast_delta_warmup with predictive discovery
    print("\n" + "=" * 60)
    print("TEST 1: fast_delta_warmup (predictive discovery)")
    print("=" * 60)

    stats_fast_pred = fast_delta_warmup(model, tokenizer, model_path, PROMPT2,
                                         discovery_tokens=WARMUP_TOKENS,
                                         discovery_method="predictive")
    print(f"  Discovery: {stats_fast_pred['discovery_time']:.2f}s")
    print(f"  Shard load: {stats_fast_pred['shard_load_time']:.2f}s")
    print(f"  Scatter: {stats_fast_pred['scatter_time']:.3f}s")
    print(f"  Lookup rebuild: {stats_fast_pred['lookup_rebuild_time']:.2f}s")
    print(f"  Rebuild total: {stats_fast_pred['rebuild_time']:.2f}s")
    print(f"  Total: {stats_fast_pred['total_time']:.2f}s")
    print(f"  Swaps: {stats_fast_pred['total_swaps']} "
          f"({stats_fast_pred['total_missing']} missing)")

    tok_s2, fb2 = generate_and_measure(model, tokenizer, PROMPT2,
                                        "Generation (after fast_delta predictive)")

    # Test 2: fast_delta_warmup with router-only discovery
    # First switch back to prompt1 cache state
    print("\n" + "=" * 60)
    print("TEST 2: fast_delta_warmup (router-only discovery)")
    print("=" * 60)

    # Re-warmup on prompt1 using current delta_warmup
    stats_revert = fast_delta_warmup(model, tokenizer, model_path, PROMPT1,
                                      discovery_tokens=WARMUP_TOKENS,
                                      discovery_method="predictive")
    print(f"  (Reverted to prompt1 in {stats_revert['total_time']:.2f}s)")

    stats_fast_router = fast_delta_warmup(model, tokenizer, model_path, PROMPT2,
                                           discovery_tokens=WARMUP_TOKENS,
                                           discovery_method="router-only")
    print(f"  Discovery: {stats_fast_router['discovery_time']:.2f}s")
    print(f"  Shard load: {stats_fast_router['shard_load_time']:.2f}s")
    print(f"  Scatter: {stats_fast_router['scatter_time']:.3f}s")
    print(f"  Lookup rebuild: {stats_fast_router['lookup_rebuild_time']:.2f}s")
    print(f"  Rebuild total: {stats_fast_router['rebuild_time']:.2f}s")
    print(f"  Total: {stats_fast_router['total_time']:.2f}s")
    print(f"  Swaps: {stats_fast_router['total_swaps']} "
          f"({stats_fast_router['total_missing']} missing)")

    tok_s3, fb3 = generate_and_measure(model, tokenizer, PROMPT2,
                                        "Generation (after fast_delta router-only)")

    # Test 3: original delta_warmup for comparison
    print("\n" + "=" * 60)
    print("TEST 3: original delta_warmup (baseline)")
    print("=" * 60)

    # Revert to prompt1 first
    stats_revert2 = fast_delta_warmup(model, tokenizer, model_path, PROMPT1,
                                       discovery_tokens=WARMUP_TOKENS,
                                       discovery_method="predictive")
    print(f"  (Reverted to prompt1 in {stats_revert2['total_time']:.2f}s)")

    stats_old = delta_warmup(model, tokenizer, model_path, PROMPT2,
                              discovery_tokens=WARMUP_TOKENS)
    print(f"  Discovery: {stats_old['discovery_time']:.2f}s")
    print(f"  Rebuild: {stats_old['rebuild_time']:.2f}s")
    print(f"  Total: {stats_old['total_time']:.2f}s")
    print(f"  Swaps: {stats_old['total_swaps']} ({stats_old['total_missing']} missing)")

    tok_s4, fb4 = generate_and_measure(model, tokenizer, PROMPT2,
                                        "Generation (after original delta_warmup)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'Total':>8} {'Disc':>8} {'Rebuild':>8} {'Swaps':>8}")
    print("-" * 70)
    print(f"{'fast_delta (predictive)':<35} "
          f"{stats_fast_pred['total_time']:>7.2f}s "
          f"{stats_fast_pred['discovery_time']:>7.2f}s "
          f"{stats_fast_pred['rebuild_time']:>7.2f}s "
          f"{stats_fast_pred['total_swaps']:>7d}")
    print(f"{'fast_delta (router-only)':<35} "
          f"{stats_fast_router['total_time']:>7.2f}s "
          f"{stats_fast_router['discovery_time']:>7.2f}s "
          f"{stats_fast_router['rebuild_time']:>7.2f}s "
          f"{stats_fast_router['total_swaps']:>7d}")
    print(f"{'delta_warmup (baseline)':<35} "
          f"{stats_old['total_time']:>7.2f}s "
          f"{stats_old['discovery_time']:>7.2f}s "
          f"{stats_old['rebuild_time']:>7.2f}s "
          f"{stats_old['total_swaps']:>7d}")
    print()
    print(f"{'Method':<35} {'tok/s':>8} {'Fallback':>10}")
    print("-" * 55)
    print(f"{'After fast_delta (predictive)':<35} {tok_s2:>7.1f} {fb2['fallback_rate']:>9.1%}")
    print(f"{'After fast_delta (router-only)':<35} {tok_s3:>7.1f} {fb3['fallback_rate']:>9.1%}")
    print(f"{'After delta_warmup (baseline)':<35} {tok_s4:>7.1f} {fb4['fallback_rate']:>9.1%}")
    print(f"\nPeak memory: {mx.get_active_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
