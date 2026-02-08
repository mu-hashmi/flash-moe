"""Cache-persistent wrapper for flash-moe generation.

Saves expert routing state after the first warmup. On subsequent runs with
the same cache file, skips the 75s warmup phase entirely — only the 12s
tensor loading (upgrade) is needed.

Usage:
    /path/to/mlx-lm/.venv/bin/python generate_persistent.py <cache_file> ["prompt"] [max_tokens] [capacity]

Examples:
    # First run: full warmup + save state (~89s)
    .../python generate_persistent.py cache.json "Explain quicksort" 200 208

    # Second run: load state + upgrade only (~12s)
    .../python generate_persistent.py cache.json "Explain quicksort" 200 208

    # Different prompt, same cache (uses delta warmup):
    .../python generate_persistent.py cache.json "Write a haiku" 100 208
"""

import os
import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, save_cache_state,
    load_cache_state, upgrade_from_saved_state, get_fallback_stats,
    fast_delta_warmup, get_cache_stats,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
WARMUP_TOKENS = 10


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cache_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a hello world program in Python"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    cache_capacity = int(sys.argv[4]) if len(sys.argv) > 4 else 208

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")

    t_total_start = time.perf_counter()

    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)

    print(f"Installing lazy expert modules (capacity={cache_capacity})...")
    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=cache_capacity,
                                   predictive=True)
    print(f"Replaced {replaced} expert modules")

    print("Evaluating non-expert parameters into Metal memory...")
    mx.eval(model.parameters())
    print(f"Non-expert params loaded. Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    if os.path.exists(cache_path):
        # Fast path: load saved state, skip warmup generation
        print(f"\nLoading saved cache state from {cache_path}...")
        t_load = time.perf_counter()
        cache_state = load_cache_state(cache_path)
        saved_capacity = cache_state.get("capacity")
        saved_meta = cache_state.get("metadata", {})
        print(f"  Saved capacity: {saved_capacity}, prompt: {saved_meta.get('prompt', '?')[:60]}")

        print(f"Building predictive cache from saved state (capacity={cache_capacity})...")
        upgraded = upgrade_from_saved_state(model, model_path, cache_state, cache_capacity)
        t_upgrade = time.perf_counter() - t_load
        print(f"Upgraded {upgraded} modules in {t_upgrade:.1f}s")
        print(f"Metal memory after upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

        # If the prompt differs from the saved one, do a delta warmup
        saved_prompt = saved_meta.get("prompt")
        if saved_prompt and saved_prompt != prompt:
            print(f"\nPrompt differs from saved state, running delta warmup...")
            stats = fast_delta_warmup(model, tokenizer, model_path, prompt,
                                      discovery_tokens=WARMUP_TOKENS)
            print(f"  Discovery: {stats['discovery_time']:.2f}s, "
                  f"Rebuild: {stats['rebuild_time']:.2f}s, "
                  f"Swaps: {stats['total_swaps']}")

    else:
        # Cold path: full warmup generation + upgrade
        print(f"\nNo saved state found. Running full warmup ({WARMUP_TOKENS} tokens)...")
        t_warmup = time.perf_counter()
        mlx_lm.generate(model, tokenizer, prompt=prompt,
                        max_tokens=WARMUP_TOKENS, verbose=False)
        t_warmup = time.perf_counter() - t_warmup
        print(f"Warmup complete in {t_warmup:.1f}s")

        warmup_stats = get_cache_stats(model)
        print(f"  Warmup hit rate: {warmup_stats['total_hit_rate']:.1%}")

        print(f"\nUpgrading to predictive cache (capacity={cache_capacity})...")
        t_upgrade = time.perf_counter()
        upgraded = upgrade_to_predictive(model, model_path, cache_capacity)
        t_upgrade = time.perf_counter() - t_upgrade
        print(f"Upgraded {upgraded} modules in {t_upgrade:.1f}s")
        print(f"Metal memory after upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

        # Save state for next run
        print(f"\nSaving cache state to {cache_path}...")
        save_cache_state(model, cache_path,
                         metadata={"prompt": prompt, "capacity": cache_capacity})
        print(f"  Saved ({os.path.getsize(cache_path) / 1024:.0f} KB)")

    t_ready = time.perf_counter() - t_total_start

    print(f"\nReady in {t_ready:.1f}s. Generating (max_tokens={max_tokens})...")
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, verbose=True,
    )

    print(f"\nFinal Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")
    fb = get_fallback_stats(model)
    print(f"Fallback rate: {fb['fallback_rate']:.1%} "
          f"({fb['total_fallbacks']}/{fb['total_requests']})")


if __name__ == "__main__":
    main()
