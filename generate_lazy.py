"""Lazy expert loading for Qwen3-Coder-Next-4bit on memory-constrained Macs.

Loads only router-selected experts on demand from memory-mapped safetensors,
keeping all non-expert weights (attention, embeddings, router, shared experts)
permanently in Metal memory. This reduces peak memory from ~40GB to ~5GB.

Usage (run from the mlx-lm venv):
    /path/to/mlx-lm/.venv/bin/python generate_lazy.py ["prompt"] [max_tokens] [capacity] [mode]

Modes:
    predictive      (default) — Warmup with LCP cache, then zero-eval forward pass.
    delta-warmup    — Warmup on default prompt, delta-update cache for actual prompt.
    sync-predictive — Same as predictive but WITH mx.eval per layer (benchmark control).
    cached          — Phase 2 eval-based LCP cache.
    lazy            — Phase 1 no-cache loading (capacity ignored).
"""

import sys
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_cache_stats,
    get_fallback_stats, delta_warmup,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
WARMUP_TOKENS = 10


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a hello world program in Python"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    cache_capacity = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    mode = sys.argv[4] if len(sys.argv) > 4 else "predictive"

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")

    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)

    if mode == "lazy":
        print("Expert loading mode: lazy (no cache)")
        replaced = enable_lazy_experts(model, model_path, cache_capacity_per_layer=0)
    elif mode == "cached":
        print(f"Expert loading mode: cached (capacity={cache_capacity})")
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity)
    elif mode == "delta-warmup":
        print(f"Expert loading mode: delta-warmup (capacity={cache_capacity})")
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity,
                                       predictive=True)
    elif mode in ("predictive", "sync-predictive"):
        print(f"Expert loading mode: {mode} (capacity={cache_capacity})")
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity,
                                       predictive=True)
    else:
        print(f"Unknown mode '{mode}', using predictive")
        mode = "predictive"
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity,
                                       predictive=True)
    print(f"Replaced {replaced} expert modules")

    print("Evaluating non-expert parameters into Metal memory...")
    mx.eval(model.parameters())
    print(f"Non-expert params loaded. Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    if mode == "delta-warmup":
        default_prompt = "Write a hello world program in Python"
        print(f"\nWarmup: generating {WARMUP_TOKENS} tokens on default prompt...")
        mlx_lm.generate(model, tokenizer, prompt=default_prompt,
                        max_tokens=WARMUP_TOKENS, verbose=False)
        print(f"\nUpgrading to predictive cache (capacity={cache_capacity})...")
        upgraded = upgrade_to_predictive(model, model_path, cache_capacity)
        print(f"Upgraded {upgraded} modules")
        print(f"Metal memory after upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

        print(f"\nDelta warmup for actual prompt...")
        stats = delta_warmup(model, tokenizer, model_path, prompt)
        print(f"  Discovery: {stats['discovery_time']:.2f}s")
        print(f"  Rebuild: {stats['rebuild_time']:.2f}s")
        print(f"  Swaps: {stats['total_swaps']} ({stats['total_missing']} missing)")
        print(f"  Total: {stats['total_time']:.2f}s")

    elif mode in ("predictive", "sync-predictive"):
        print(f"\nWarmup: generating {WARMUP_TOKENS} tokens to discover expert routing...")
        mlx_lm.generate(model, tokenizer, prompt=prompt,
                        max_tokens=WARMUP_TOKENS, verbose=False)
        warmup_stats = get_cache_stats(model)
        print(f"  Warmup hit rate: {warmup_stats['total_hit_rate']:.1%}")

        sync = mode == "sync-predictive"
        label = "sync-predictive (with mx.eval)" if sync else "zero-eval predictive"
        print(f"\nUpgrading to {label} cache (capacity={cache_capacity})...")
        upgraded = upgrade_to_predictive(model, model_path, cache_capacity, sync=sync)
        print(f"Upgraded {upgraded} modules")
        print(f"Metal memory after upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

    print(f"\nGenerating (max_tokens={max_tokens})...")
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=True,
    )

    print(f"\nFinal Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    if mode in ("predictive", "sync-predictive", "delta-warmup"):
        fb = get_fallback_stats(model)
        print(f"Fallback rate: {fb['fallback_rate']:.1%} "
              f"({fb['total_fallbacks']}/{fb['total_requests']})")

    if mode == "cached":
        stats = get_cache_stats(model)
        print(f"\nCache stats:")
        print(f"  Total hit rate: {stats['total_hit_rate']:.1%} "
              f"({stats['total_hits']} hits / {stats['total_hits'] + stats['total_misses']} lookups)")
        if stats['layers']:
            hit_rates = [l['hit_rate'] for l in stats['layers']]
            print(f"  Per-layer hit rate: min={min(hit_rates):.1%} "
                  f"median={sorted(hit_rates)[len(hit_rates)//2]:.1%} "
                  f"max={max(hit_rates):.1%}")
            cached = [l['cached_experts'] for l in stats['layers']]
            print(f"  Cached experts per layer: min={min(cached)} "
                  f"median={sorted(cached)[len(cached)//2]} "
                  f"max={max(cached)}")


if __name__ == "__main__":
    main()
