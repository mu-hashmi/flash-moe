"""Sweep capacity for Mixtral models and measure speed, memory, and fallback rate.

Usage:
    uv run python benchmarks/mixtral/bench_capacity.py [--model MODEL] [--caps 1,2,3,4]
    uv run python benchmarks/mixtral/bench_capacity.py --model 2bit --caps 2,3,4,5,6

Presets:
    4bit  → mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit
    2bit  → ~/.cache/flash-moe/Mixtral-8x22B-Instruct-v0.1-2bit (local)
    8x7b  → mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
"""
import argparse
import json
import sys
import time

import mlx.core as mx

MODEL_PRESETS = {
    "4bit": "mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit",
    "2bit": "/Users/muhash/.cache/flash-moe/Mixtral-8x22B-Instruct-v0.1-2bit",
    "sorcerer": "mlx-community/SorcererLM-8x22b-2bit",
    "8x7b": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
}

PROMPT = "Write a Python function that computes the nth Fibonacci number using memoization."
MAX_TOKENS = 50


def run_one(model_name, capacity, profile_path=None):
    import mlx_lm
    from flash_moe.lazy_experts.core import (
        enable_lazy_experts, upgrade_to_predictive, measure_fallback,
    )
    from flash_moe.lazy_experts.loading import _find_switch_mlp, _detect_num_experts
    from flash_moe.lazy_experts.persistence import load_universal_profile, upgrade_from_profile
    from mlx_lm.utils import hf_repo_to_path

    model, tokenizer = mlx_lm.load(model_name, lazy=True)
    model_path = hf_repo_to_path(model_name) if "/" in model_name else model_name

    num_experts = 0
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            num_experts = _detect_num_experts(switch)
            break

    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())
    base_gb = mx.get_active_memory() / 1e9

    t0 = time.perf_counter()
    if profile_path:
        profile = load_universal_profile(profile_path)
        upgrade_from_profile(model, model_path, capacity, profile)
    else:
        from flash_moe.lazy_experts.discovery import router_only_discovery
        router_only_discovery(model, tokenizer, PROMPT, max_tokens=5)
        upgrade_to_predictive(model, model_path, capacity)
    t_warmup = time.perf_counter() - t0

    loaded_gb = mx.get_active_memory() / 1e9

    if hasattr(mx, "set_wired_limit"):
        active = mx.get_active_memory()
        limit = int(mx.device_info()["memory_size"] * 0.75)
        mx.set_wired_limit(min(active, limit))

    t0 = time.perf_counter()
    result = mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    t_gen = time.perf_counter() - t0

    fb = measure_fallback(model)

    return {
        "capacity": capacity,
        "num_experts": num_experts,
        "base_gb": round(base_gb, 2),
        "loaded_gb": round(loaded_gb, 2),
        "warmup_s": round(t_warmup, 1),
        "gen_s": round(t_gen, 1),
        "tok_s": round(MAX_TOKENS / t_gen, 1),
        "fallback_rate": round(fb["fallback_rate"] * 100, 1),
        "output_preview": result[:100],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="4bit", help="Model preset or HF path")
    parser.add_argument("--caps", default="1,2", help="Comma-separated capacity values to sweep")
    parser.add_argument("--profile", default=None, help="Path to expert profile JSON")
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    args = parser.parse_args()

    model_name = MODEL_PRESETS.get(args.model, args.model)
    caps = [int(c) for c in args.caps.split(",")]

    print(f"Model: {model_name}")
    print(f"Capacities: {caps}")
    print(f"Prompt: {PROMPT[:60]}...")
    print(f"Max tokens: {MAX_TOKENS}")
    print()

    results = []
    for cap in caps:
        print(f"--- cap={cap} ---")
        try:
            r = run_one(model_name, cap, profile_path=args.profile)
            results.append(r)
            print(f"  Memory: {r['loaded_gb']} GB")
            print(f"  Speed: {r['tok_s']} tok/s ({r['gen_s']}s)")
            print(f"  Fallback: {r['fallback_rate']}%")
            print(f"  Output: {r['output_preview'][:80]}...")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"capacity": cap, "error": str(e)})
        print()

    print("=== Summary ===")
    print(f"{'cap':>4} {'mem_gb':>7} {'tok/s':>6} {'fallback':>9} {'warmup':>7}")
    for r in results:
        if "error" in r:
            print(f"{r['capacity']:>4} {'ERROR':>7}")
        else:
            print(f"{r['capacity']:>4} {r['loaded_gb']:>7.1f} {r['tok_s']:>6.1f} {r['fallback_rate']:>8.1f}% {r['warmup_s']:>6.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
