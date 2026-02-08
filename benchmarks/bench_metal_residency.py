#!/usr/bin/env python
"""Benchmark Metal residency and cache strategies for flash-moe.

Tests the effect of mx.set_wired_limit() and refined cache_limit
strategies on generation throughput and memory pressure cliff behavior.

Usage:
    /Users/muhash/mlx-lm/.venv/bin/python benchmarks/bench_metal_residency.py [profile_path] [capacity]

Each configuration runs in-process but with full gc/cache cleanup between runs.
"""

import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts,
    upgrade_from_profile,
    load_universal_profile,
    load_prepacked_weights,
    save_prepacked_weights,
    _with_cache_limit_zero,
    _find_switch_mlp,
    _detect_num_experts,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
PROMPT = "Write a Python function that implements binary search on a sorted array."
PREPACKED_PATH = "/tmp/flash_moe_bench_residency.safetensors"
GEN_TOKENS = 200


def cleanup_model(model):
    del model
    gc.collect()
    mx.clear_cache()
    time.sleep(1)


def prepare_model(model_path, capacity, profile_path, cache_bytes=0):
    """Load model and upgrade to predictive. Returns (model, tokenizer, startup_time)."""
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())

    with _with_cache_limit_zero(cache_bytes):
        if Path(PREPACKED_PATH).exists():
            load_prepacked_weights(model, PREPACKED_PATH, model_path=model_path)
        else:
            profile = load_universal_profile(profile_path)
            upgrade_from_profile(model, model_path, capacity, profile)
            save_prepacked_weights(model, PREPACKED_PATH)

    startup = time.perf_counter() - t0
    return model, tokenizer, startup


def measure_generation(model, tokenizer, max_tokens):
    """Generate tokens and return (text, tok_per_sec, elapsed)."""
    t0 = time.perf_counter()
    text = mlx_lm.generate(model, tokenizer, prompt=PROMPT,
                           max_tokens=max_tokens, verbose=False)
    elapsed = time.perf_counter() - t0
    tps = max_tokens / elapsed
    return text, tps, elapsed


def run_config(name, capacity, model_path, profile_path,
               wire=False, cache_bytes=0):
    """Run one configuration: prepare model, optionally wire, generate, measure."""
    gc.collect()
    mx.clear_cache()
    if hasattr(mx.metal, "set_wired_limit"):
        mx.set_wired_limit(0)
    time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  capacity={capacity}, wire={wire}, cache_bytes={cache_bytes}")
    print(f"{'=' * 60}")

    model, tokenizer, startup = prepare_model(
        model_path, capacity, profile_path, cache_bytes=cache_bytes,
    )

    active_gb = mx.get_active_memory() / 1e9
    peak_gb = mx.get_peak_memory() / 1e9
    print(f"  Startup: {startup:.1f}s | Active: {active_gb:.1f} GB | Peak: {peak_gb:.1f} GB")

    wired_gb = 0.0
    if wire and hasattr(mx.metal, "set_wired_limit"):
        active = mx.get_active_memory()
        limit = int(mx.device_info()["memory_size"] * 0.75)
        wired = min(active, limit)
        mx.set_wired_limit(wired)
        wired_gb = wired / 1e9
        print(f"  Wired: {wired_gb:.1f} GB")

    mx.reset_peak_memory()

    text, tps, elapsed = measure_generation(model, tokenizer, GEN_TOKENS)

    gen_peak_gb = mx.get_peak_memory() / 1e9
    gen_active_gb = mx.get_active_memory() / 1e9
    print(f"  Generation: {tps:.1f} tok/s ({elapsed:.1f}s for {GEN_TOKENS} tokens)")
    print(f"  Post-gen active: {gen_active_gb:.1f} GB | Gen peak: {gen_peak_gb:.1f} GB")
    print(f"  Output: {text[:100]}...")

    result = {
        "name": name,
        "capacity": capacity,
        "wire": wire,
        "wired_gb": round(wired_gb, 2),
        "cache_bytes": cache_bytes,
        "startup_s": round(startup, 2),
        "active_gb": round(active_gb, 2),
        "peak_gb": round(peak_gb, 2),
        "gen_tps": round(tps, 2),
        "gen_elapsed_s": round(elapsed, 2),
        "gen_peak_gb": round(gen_peak_gb, 2),
        "gen_active_gb": round(gen_active_gb, 2),
    }

    cleanup_model(model)
    return result


def main():
    profile_path = sys.argv[1] if len(sys.argv) > 1 else None
    base_capacity = int(sys.argv[2]) if len(sys.argv) > 2 else 208

    if not profile_path:
        print("ERROR: profile_path required (universal_experts.json)")
        print("Usage: python bench_metal_residency.py <profile.json> [capacity]")
        sys.exit(1)

    model_path = hf_repo_to_path(MODEL)
    device_gb = mx.device_info()["memory_size"] / 1e9

    print(f"Model: {MODEL}")
    print(f"Model path: {model_path}")
    print(f"Profile: {profile_path}")
    print(f"Device memory: {device_gb:.1f} GB")
    print(f"Base capacity: {base_capacity}")
    print(f"Generation tokens: {GEN_TOKENS}")
    print(f"set_wired_limit available: {hasattr(mx.metal, 'set_wired_limit')}")

    if Path(PREPACKED_PATH).exists():
        Path(PREPACKED_PATH).unlink()
    meta = Path(str(PREPACKED_PATH) + ".meta.json")
    if meta.exists():
        meta.unlink()

    results = []

    # --- Section 1: Wired limit effect at base capacity ---

    # 1a. Baseline: no wiring, cache_limit=0 during warmup
    results.append(run_config(
        "baseline (no wire)", base_capacity, model_path, profile_path,
        wire=False, cache_bytes=0,
    ))

    # 1b. With wiring
    results.append(run_config(
        "wired", base_capacity, model_path, profile_path,
        wire=True, cache_bytes=0,
    ))

    # --- Section 2: Push capacity past the pressure cliff ---

    for cap in [224, 240]:
        results.append(run_config(
            f"wired cap={cap}", cap, model_path, profile_path,
            wire=True, cache_bytes=0,
        ))

    # --- Section 3: Cache limit strategy during warmup ---

    # Ensure fresh prepacked for cache tests
    if Path(PREPACKED_PATH).exists():
        Path(PREPACKED_PATH).unlink()
    if meta.exists():
        meta.unlink()

    results.append(run_config(
        "cache=0 (current)", base_capacity, model_path, profile_path,
        wire=True, cache_bytes=0,
    ))

    # Fresh prepacked again
    if Path(PREPACKED_PATH).exists():
        Path(PREPACKED_PATH).unlink()
    if meta.exists():
        meta.unlink()

    results.append(run_config(
        "cache=256MB", base_capacity, model_path, profile_path,
        wire=True, cache_bytes=256 * 1024 * 1024,
    ))

    # --- Summary ---

    print(f"\n{'=' * 100}")
    print(f"=== Metal Residency Benchmark Results ===")
    print(f"Model: {MODEL} | Device: {device_gb:.0f} GB | Gen: {GEN_TOKENS} tokens")
    print(f"{'=' * 100}")

    header = (
        f"{'Config':<22} | {'Cap':>4} | {'Wire':>5} | {'Startup':>7} | "
        f"{'Active':>7} | {'tok/s':>6} | {'GenPeak':>7} | {'Cache':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        cache_label = "0" if r["cache_bytes"] == 0 else f"{r['cache_bytes'] // (1024*1024)}MB"
        print(
            f"{r['name']:<22} | {r['capacity']:>4} | "
            f"{'yes' if r['wire'] else 'no':>5} | {r['startup_s']:>6.1f}s | "
            f"{r['active_gb']:>5.1f} GB | {r['gen_tps']:>5.1f} | "
            f"{r['gen_peak_gb']:>5.1f} GB | {cache_label:>8}"
        )

    # Save JSON
    out_path = Path(__file__).parent / "metal_residency_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "device_gb": device_gb,
            "gen_tokens": GEN_TOKENS,
            "prompt": PROMPT,
            "configs": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    for p in [PREPACKED_PATH, meta]:
        if Path(p).exists():
            Path(p).unlink()


if __name__ == "__main__":
    main()
