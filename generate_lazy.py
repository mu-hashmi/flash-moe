"""Lazy expert loading for Qwen3-Coder-Next-4bit on memory-constrained Macs.

Loads only router-selected experts on demand from memory-mapped safetensors,
keeping all non-expert weights (attention, embeddings, router, shared experts)
permanently in Metal memory. This reduces peak memory from ~40GB to ~5GB.

Usage (run from the mlx-lm venv):
    /path/to/mlx-lm/.venv/bin/python generate_lazy.py ["prompt"] [max_tokens] [capacity] [mode] [refresh_interval]

Modes:
    predictive      (default) — Warmup with LCP cache, then zero-eval forward pass.
    delta-warmup    — Warmup on default prompt, delta-update cache for actual prompt.
    async-delta     — Warmup on default prompt, then generate immediately while
                      incrementally swapping experts between tokens.
                      Append "-coherent" or "-hybrid" for hold-and-discard stream modes.
    sync-predictive — Same as predictive but WITH mx.eval per layer (benchmark control).
    cached          — Phase 2 eval-based LCP cache.
    lazy            — Phase 1 no-cache loading (capacity ignored).

refresh_interval (5th arg, default 0):
    When > 0 with predictive/sync-predictive mode, calls dynamic_cache_update()
    every N tokens during stream_generate. Fixes 300-token degradation.
"""

import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_cache_stats,
    get_fallback_stats, delta_warmup, incremental_delta_warmup,
    dynamic_cache_update,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
WARMUP_TOKENS = 10


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a hello world program in Python"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    cache_capacity = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    mode = sys.argv[4] if len(sys.argv) > 4 else "predictive"
    refresh_interval = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")

    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)

    # Parse stream_mode for async-delta variants
    stream_mode = "immediate"
    base_mode = mode
    if mode.startswith("async-delta"):
        base_mode = "async-delta"
        if mode == "async-delta-coherent":
            stream_mode = "coherent"
        elif mode == "async-delta-hybrid":
            stream_mode = "hybrid"

    if base_mode == "lazy":
        print("Expert loading mode: lazy (no cache)")
        replaced = enable_lazy_experts(model, model_path, cache_capacity_per_layer=0)
    elif base_mode == "cached":
        print(f"Expert loading mode: cached (capacity={cache_capacity})")
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity)
    elif base_mode in ("delta-warmup", "async-delta"):
        print(f"Expert loading mode: {mode} (capacity={cache_capacity})")
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity,
                                       predictive=True)
    elif base_mode in ("predictive", "sync-predictive"):
        print(f"Expert loading mode: {mode} (capacity={cache_capacity})")
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity,
                                       predictive=True)
    else:
        print(f"Unknown mode '{mode}', using predictive")
        base_mode = "predictive"
        replaced = enable_lazy_experts(model, model_path,
                                       cache_capacity_per_layer=cache_capacity,
                                       predictive=True)
    print(f"Replaced {replaced} expert modules")

    print("Evaluating non-expert parameters into Metal memory...")
    mx.eval(model.parameters())
    print(f"Non-expert params loaded. Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    if base_mode == "delta-warmup":
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

    elif base_mode == "async-delta":
        default_prompt = "Write a hello world program in Python"
        print(f"\nWarmup: generating {WARMUP_TOKENS} tokens on default prompt...")
        mlx_lm.generate(model, tokenizer, prompt=default_prompt,
                        max_tokens=WARMUP_TOKENS, verbose=False)
        print(f"\nUpgrading to predictive cache (capacity={cache_capacity})...")
        upgraded = upgrade_to_predictive(model, model_path, cache_capacity)
        print(f"Upgraded {upgraded} modules")
        print(f"Metal memory after upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

        print(f"\nAsync delta warmup: discovering experts for actual prompt...")
        warmup, disc_stats = incremental_delta_warmup(
            model, tokenizer, model_path, prompt,
            discovery_tokens=WARMUP_TOKENS,
        )
        print(f"  Discovery: {disc_stats['discovery_time']:.1f}s")
        print(f"  Layers to swap: {disc_stats['total_layers']}")
        print(f"  Total swaps: {disc_stats['total_swaps']}")

        print(f"\nGenerating with incremental swaps (max_tokens={max_tokens}, "
              f"stream={stream_mode})...")
        t_start = time.perf_counter()
        token_count = 0
        discarded_count = 0

        if stream_mode == "hybrid":
            print("[thinking...]", end='', flush=True)

        for response in mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                                max_tokens=max_tokens):
            token_count += 1

            if not warmup.is_complete:
                swapped = warmup.step(layers_per_step=2)

                if stream_mode == "immediate":
                    print(response.text, end='', flush=True)
                elif stream_mode == "coherent":
                    discarded_count += 1
                # hybrid: show nothing (placeholder already printed)

                if swapped and warmup.is_complete:
                    t_done = time.perf_counter() - t_start
                    if stream_mode == "hybrid":
                        # Clear placeholder, show transition
                        print(f"\r\033[K  [swap complete at token {token_count}, "
                              f"{t_done:.1f}s, {discarded_count} tokens discarded]",
                              flush=True)
                    elif stream_mode == "coherent":
                        print(f"  [swap complete at token {token_count}, "
                              f"{t_done:.1f}s, {discarded_count} tokens discarded]",
                              flush=True)
                    else:
                        print(f"\n  [swap complete at token {token_count}, "
                              f"{t_done:.1f}s elapsed]", flush=True)
                    discarded_count = token_count
            else:
                print(response.text, end='', flush=True)

        t_total = time.perf_counter() - t_start

        prog = warmup.progress
        print(f"\n\nAsync delta stats:")
        print(f"  Stream mode: {stream_mode}")
        print(f"  Tokens generated: {token_count}")
        if discarded_count > 0:
            print(f"  Tokens discarded (stale): {discarded_count}")
        print(f"  Total time: {t_total:.1f}s ({token_count / t_total:.1f} tok/s)")
        print(f"  Layers swapped: {prog['layers_done']}/{prog['layers_total']}")
        print(f"  Experts swapped: {prog['swaps_done']}/{prog['swaps_total']}")
        print(f"  Final memory: {mx.get_active_memory() / 1e9:.1f} GB")

        fb = get_fallback_stats(model)
        print(f"  Fallback rate: {fb['fallback_rate']:.1%} "
              f"({fb['total_fallbacks']}/{fb['total_requests']})")
        return

    elif base_mode in ("predictive", "sync-predictive"):
        print(f"\nWarmup: generating {WARMUP_TOKENS} tokens to discover expert routing...")
        mlx_lm.generate(model, tokenizer, prompt=prompt,
                        max_tokens=WARMUP_TOKENS, verbose=False)
        warmup_stats = get_cache_stats(model)
        print(f"  Warmup hit rate: {warmup_stats['total_hit_rate']:.1%}")

        sync = base_mode == "sync-predictive"
        label = "sync-predictive (with mx.eval)" if sync else "zero-eval predictive"
        print(f"\nUpgrading to {label} cache (capacity={cache_capacity})...")
        upgraded = upgrade_to_predictive(model, model_path, cache_capacity, sync=sync)
        print(f"Upgraded {upgraded} modules")
        print(f"Metal memory after upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

    if base_mode in ("predictive", "sync-predictive") and refresh_interval > 0:
        print(f"\nGenerating with dynamic refresh every {refresh_interval} tokens "
              f"(max_tokens={max_tokens})...")
        t_start = time.perf_counter()
        token_count = 0
        total_swaps = 0
        for response in mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                                max_tokens=max_tokens):
            print(response.text, end='', flush=True)
            token_count += 1
            if token_count % refresh_interval == 0:
                stats = dynamic_cache_update(model)
                swaps = sum(s["swaps"] for s in stats)
                total_swaps += swaps
                if swaps > 0:
                    print(f"\n  [refresh at token {token_count}: {swaps} swaps]", flush=True)
        t_total = time.perf_counter() - t_start

        print(f"\n\nDynamic refresh stats:")
        print(f"  Tokens generated: {token_count}")
        print(f"  Total time: {t_total:.1f}s ({token_count / t_total:.1f} tok/s)")
        print(f"  Total swaps: {total_swaps}")
        print(f"  Final memory: {mx.get_active_memory() / 1e9:.1f} GB")

        fb = get_fallback_stats(model)
        print(f"  Fallback rate: {fb['fallback_rate']:.1%} "
              f"({fb['total_fallbacks']}/{fb['total_requests']})")
        return

    print(f"\nGenerating (max_tokens={max_tokens})...")
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=True,
    )

    print(f"\nFinal Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    if base_mode in ("predictive", "sync-predictive", "delta-warmup"):
        fb = get_fallback_stats(model)
        print(f"Fallback rate: {fb['fallback_rate']:.1%} "
              f"({fb['total_fallbacks']}/{fb['total_requests']})")

    if base_mode == "cached":
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
