#!/usr/bin/env python
"""Benchmark warmup optimizations for flash-moe.

Usage:
    PATH_REMOVED benchmarks/bench_warmup.py [profile_path] [capacity]

Runs each warmup configuration and reports timing breakdown.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts,
    upgrade_to_predictive,
    upgrade_to_predictive_with_pinning,
    router_only_discovery,
    save_prepacked_weights,
    load_prepacked_weights,
    upgrade_from_profile,
    load_universal_profile,
    save_cache_state,
    load_cache_state,
    upgrade_from_saved_state,
    get_fallback_stats,
    _with_cache_limit_zero,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
WARMUP_TOKENS = 10
PROMPT = "Write a Python function that implements binary search on a sorted array."
PREPACKED_PATH = "/tmp/flash_moe_bench_prepacked.safetensors"
CACHE_PATH = "/tmp/flash_moe_bench_cache.json"


def cleanup_temp_files(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def run_config(name, capacity, model_path, profile_path=None, opts=None):
    """Run one warmup configuration, return timing dict."""
    opts = opts or {}
    gc.collect()
    mx.clear_cache()

    timings = {}

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Phase 1: model load
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    timings["model_load"] = time.perf_counter() - t0
    print(f"  Model loaded: {timings['model_load']:.1f}s")

    # Phase 2: enable lazy experts + eval non-expert params
    t0 = time.perf_counter()
    with _with_cache_limit_zero():
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
        mx.eval(model.parameters())
    timings["enable_lazy"] = time.perf_counter() - t0
    mem_base = mx.get_active_memory() / 1e9
    print(f"  Lazy enabled + params eval'd: {timings['enable_lazy']:.1f}s ({mem_base:.1f} GB)")

    # Phase 3: discovery (config-dependent)
    t0 = time.perf_counter()
    if opts.get("use_full_warmup"):
        mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=WARMUP_TOKENS, verbose=False)
    elif opts.get("use_router_only"):
        router_only_discovery(model, tokenizer, PROMPT, max_tokens=WARMUP_TOKENS)
    # profile-based and prepacked skip discovery entirely
    timings["discovery"] = time.perf_counter() - t0
    print(f"  Discovery: {timings['discovery']:.1f}s")

    # Phase 4: upgrade/loading (config-dependent)
    t0 = time.perf_counter()
    with _with_cache_limit_zero():
        if opts.get("use_prepacked"):
            load_prepacked_weights(model, PREPACKED_PATH)
        elif opts.get("use_profile"):
            profile = load_universal_profile(profile_path)
            upgrade_from_profile(model, model_path, capacity, profile)
            if opts.get("save_prepacked"):
                save_prepacked_weights(model, PREPACKED_PATH)
        else:
            upgrade_to_predictive(model, model_path, capacity)
    timings["upgrade"] = time.perf_counter() - t0
    mem_after_upgrade = mx.get_active_memory() / 1e9
    print(f"  Upgrade: {timings['upgrade']:.1f}s ({mem_after_upgrade:.1f} GB)")

    # Phase 5: verify with short generation
    t0 = time.perf_counter()
    text = mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=20, verbose=False)
    timings["gen_20tok"] = time.perf_counter() - t0
    mem_final = mx.get_active_memory() / 1e9
    print(f"  20-token gen: {timings['gen_20tok']:.1f}s ({mem_final:.1f} GB)")
    print(f"  Output: {text[:80]}...")

    timings["mem_base"] = mem_base
    timings["mem_final"] = mem_final
    timings["total"] = sum(
        v for k, v in timings.items()
        if k in ("model_load", "enable_lazy", "discovery", "upgrade")
    )

    del model
    gc.collect()
    mx.clear_cache()

    return timings


def main():
    profile_path = sys.argv[1] if len(sys.argv) > 1 else None
    capacity = int(sys.argv[2]) if len(sys.argv) > 2 else 208

    model_path = hf_repo_to_path(MODEL)
    print(f"Model: {MODEL}")
    print(f"Model path: {model_path}")
    print(f"Capacity: {capacity}")
    if profile_path:
        print(f"Profile: {profile_path}")

    cleanup_temp_files(PREPACKED_PATH, CACHE_PATH)

    results = {}

    # 1. Baseline: full 10-token warmup + upgrade
    results["baseline"] = run_config(
        "baseline (full warmup)", capacity, model_path,
        opts={"use_full_warmup": True},
    )

    # 2. Router-only discovery + upgrade
    results["router-only"] = run_config(
        "router-only discovery", capacity, model_path,
        opts={"use_router_only": True},
    )

    # 3-5 require a profile
    if profile_path:
        # 3. Profile-based (no discovery, no prepacked)
        results["profile-based"] = run_config(
            "profile-based (skip discovery)", capacity, model_path,
            profile_path=profile_path,
            opts={"use_profile": True},
        )

        # 4. Combined cold: profile-based + save prepacked for next time
        cleanup_temp_files(PREPACKED_PATH)
        results["combined-cold"] = run_config(
            "combined cold (profile + save prepacked)", capacity, model_path,
            profile_path=profile_path,
            opts={"use_profile": True, "save_prepacked": True},
        )

        # 5. Prepacked warm: load prepacked weights directly
        if os.path.exists(PREPACKED_PATH):
            results["prepacked-warm"] = run_config(
                "prepacked warm (load saved weights)", capacity, model_path,
                opts={"use_prepacked": True},
            )

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"=== Warmup Optimization Benchmarks ===")
    print(f"Model: {MODEL} | Capacity: {capacity}")
    print(f"{'=' * 80}")

    header = f"{'Config':<20} | {'Load':>6} | {'Enable':>6} | {'Discover':>9} | {'Upgrade':>7} | {'Total':>7} | {'Memory':>7}"
    print(header)
    print("-" * len(header))

    for name, t in results.items():
        print(
            f"{name:<20} | {t['model_load']:>5.1f}s | {t['enable_lazy']:>5.1f}s | "
            f"{t['discovery']:>8.1f}s | {t['upgrade']:>6.1f}s | "
            f"{t['total']:>6.1f}s | {t['mem_final']:>5.1f} GB"
        )

    # Save JSON
    out_path = Path(__file__).parent / "warmup_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "capacity": capacity,
            "profile_path": profile_path,
            "configs": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    cleanup_temp_files(PREPACKED_PATH, CACHE_PATH)


if __name__ == "__main__":
    main()
