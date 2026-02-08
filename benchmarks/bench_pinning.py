"""Benchmark universal expert pinning vs baseline on long generation.

Tests the hypothesis that pinning universal experts fixes the 300-token
degradation by eliminating filler experts at startup.

Usage:
    /Users/muhash/mlx-lm/.venv/bin/python bench_pinning.py [profile_path] [capacity] [max_tokens]
"""

import argparse
import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_fallback_stats,
    load_universal_profile, upgrade_to_predictive_with_pinning,
    dynamic_cache_update, measure_fallback,
)

MODEL_PRESETS = {
    "qwen": ("mlx-community/Qwen3-Coder-Next-4bit", 208),
    "mixtral": ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 6),
    "glm": ("mlx-community/GLM-4.7-Flash-4bit", 48),
}
WARMUP_TOKENS = 10
PROMPT = "Write a comprehensive tutorial on building a web application with Python and Flask, including project setup, routing, templates, database integration, authentication, and deployment"


def run_generation(model, tokenizer, max_tokens, refresh_interval=0):
    """Generate tokens, track per-checkpoint quality and speed."""
    checkpoints = [100, 200, 300, 400, 500, 750, 1000]
    results = []
    t_start = time.perf_counter()
    token_count = 0
    total_swaps = 0
    text_buf = []

    for response in mlx_lm.stream_generate(model, tokenizer, prompt=PROMPT,
                                            max_tokens=max_tokens):
        text_buf.append(response.text)
        token_count += 1

        if refresh_interval > 0 and token_count % refresh_interval == 0:
            stats = dynamic_cache_update(model)
            total_swaps += sum(s["swaps"] for s in stats)

        if token_count in checkpoints:
            elapsed = time.perf_counter() - t_start
            tok_s = token_count / elapsed
            mem = mx.get_active_memory() / 1e9
            text_so_far = "".join(text_buf)
            # Check for repetition: last 200 chars
            tail = text_so_far[-200:] if len(text_so_far) >= 200 else text_so_far
            # Simple repetition detection: count most common 3-gram
            trigrams = [tail[i:i+3] for i in range(len(tail)-2)]
            if trigrams:
                from collections import Counter
                most_common_count = Counter(trigrams).most_common(1)[0][1]
                repetition_score = most_common_count / len(trigrams)
            else:
                repetition_score = 0.0

            results.append({
                "token": token_count,
                "elapsed": elapsed,
                "tok_s": tok_s,
                "mem_gb": mem,
                "repetition": repetition_score,
                "tail_sample": tail[-80:],
            })

    fb = measure_fallback(model)
    return {
        "total_tokens": token_count,
        "total_time": time.perf_counter() - t_start,
        "total_swaps": total_swaps,
        "fallback_rate": fb["fallback_rate"],
        "fallback_count": fb["total_fallbacks"],
        "checkpoints": results,
        "full_text": "".join(text_buf),
    }


def apply_chat_template(tokenizer, text):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
    return text


def load_and_setup(model_name, model_path, capacity, tokenizer_wrap=None):
    """Load model, enable lazy experts, warmup, return (model, tokenizer)."""
    model, tokenizer = mlx_lm.load(model_name, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())
    print(f"Params loaded: {mx.get_active_memory() / 1e9:.1f} GB")

    prompt = PROMPT
    if tokenizer_wrap:
        prompt = apply_chat_template(tokenizer, PROMPT)
    mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=WARMUP_TOKENS, verbose=False)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Pinning benchmark")
    parser.add_argument("--model", "-m", default="qwen",
                        help="Model preset (qwen/mixtral/glm) or HuggingFace name")
    parser.add_argument("--profile", default=None,
                        help="Profile JSON path (default: auto per model)")
    parser.add_argument("--capacity", "-c", type=int, default=None)
    parser.add_argument("--max-tokens", "-t", type=int, default=1000)
    args = parser.parse_args()

    if args.model in MODEL_PRESETS:
        model_name, default_capacity = MODEL_PRESETS[args.model]
    else:
        model_name = args.model
        default_capacity = 208

    capacity = args.capacity or default_capacity
    max_tokens = args.max_tokens
    use_chat = args.model in ("mixtral", "glm") or "instruct" in model_name.lower()
    profile_defaults = {
        "qwen": "profiles/qwen3-coder-next.json",
        "mixtral": "profiles/mixtral-8x7b.json",
        "glm": "profiles/glm-4.7-flash.json",
    }
    profile_path = args.profile or profile_defaults.get(args.model, f"{args.model}_experts.json")

    model_path = hf_repo_to_path(model_name)
    print(f"Model: {model_name}")
    print(f"Capacity: {capacity}, Max tokens: {max_tokens}")
    print(f"Profile: {profile_path}")

    # --- Config A: Baseline (no pinning, no refresh) ---
    print("=" * 60)
    print("CONFIG A: Baseline (no pinning, no refresh)")
    print("=" * 60)
    model, tokenizer = load_and_setup(model_name, model_path, capacity, use_chat)
    upgrade_to_predictive(model, model_path, capacity)
    print(f"Upgraded: {mx.get_active_memory() / 1e9:.1f} GB")

    result_a = run_generation(model, tokenizer, max_tokens)
    print(f"\nBaseline: {result_a['total_tokens']} tokens in {result_a['total_time']:.1f}s")
    for cp in result_a["checkpoints"]:
        print(f"  Token {cp['token']:4d}: {cp['tok_s']:.1f} tok/s, rep={cp['repetition']:.2f}, "
              f"{cp['tail_sample'][:60]}")
    del model
    mx.clear_cache()

    # --- Config B: Baseline + refresh every 50 tokens ---
    print("\n" + "=" * 60)
    print("CONFIG B: Baseline + refresh every 50 tokens")
    print("=" * 60)
    model, tokenizer = load_and_setup(model_name, model_path, capacity, use_chat)
    upgrade_to_predictive(model, model_path, capacity)
    print(f"Upgraded: {mx.get_active_memory() / 1e9:.1f} GB")

    result_b = run_generation(model, tokenizer, max_tokens, refresh_interval=50)
    print(f"\n+Refresh: {result_b['total_tokens']} tokens in {result_b['total_time']:.1f}s, "
          f"{result_b['total_swaps']} swaps")
    for cp in result_b["checkpoints"]:
        print(f"  Token {cp['token']:4d}: {cp['tok_s']:.1f} tok/s, rep={cp['repetition']:.2f}, "
              f"{cp['tail_sample'][:60]}")
    del model
    mx.clear_cache()

    # --- Config C: Pinning (no refresh) ---
    print("\n" + "=" * 60)
    print("CONFIG C: Pinning (no refresh)")
    print("=" * 60)
    profile = load_universal_profile(profile_path)
    n_universal = sum(len(v["universal"]) for v in profile["layers"].values())
    print(f"Loaded profile: {n_universal} total universal experts across {len(profile['layers'])} layers")

    model, tokenizer = load_and_setup(model_name, model_path, capacity, use_chat)
    upgrade_to_predictive_with_pinning(model, model_path, capacity, profile, pin_threshold=0.5)
    print(f"Upgraded with pinning: {mx.get_active_memory() / 1e9:.1f} GB")

    result_c = run_generation(model, tokenizer, max_tokens)
    print(f"\n+Pinning: {result_c['total_tokens']} tokens in {result_c['total_time']:.1f}s")
    for cp in result_c["checkpoints"]:
        print(f"  Token {cp['token']:4d}: {cp['tok_s']:.1f} tok/s, rep={cp['repetition']:.2f}, "
              f"{cp['tail_sample'][:60]}")
    del model
    mx.clear_cache()

    # --- Config D: Pinning + refresh every 50 tokens ---
    print("\n" + "=" * 60)
    print("CONFIG D: Pinning + refresh every 50 tokens")
    print("=" * 60)
    model, tokenizer = load_and_setup(model_name, model_path, capacity, use_chat)
    upgrade_to_predictive_with_pinning(model, model_path, capacity, profile, pin_threshold=0.5)
    print(f"Upgraded with pinning: {mx.get_active_memory() / 1e9:.1f} GB")

    result_d = run_generation(model, tokenizer, max_tokens, refresh_interval=50)
    print(f"\n+Pinning+Refresh: {result_d['total_tokens']} tokens in {result_d['total_time']:.1f}s, "
          f"{result_d['total_swaps']} swaps")
    for cp in result_d["checkpoints"]:
        print(f"  Token {cp['token']:4d}: {cp['tok_s']:.1f} tok/s, rep={cp['repetition']:.2f}, "
              f"{cp['tail_sample'][:60]}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}, Capacity: {capacity}")
    print(f"{'Config':<30} {'tok/s':>6} {'Swaps':>6} {'Rep@300':>8} {'Rep@500':>8} {'Rep@1000':>8}")
    for label, result in [("A: Baseline", result_a), ("B: +Refresh", result_b),
                          ("C: +Pinning", result_c), ("D: +Pinning+Refresh", result_d)]:
        tok_s = result["total_tokens"] / result["total_time"]
        reps = {cp["token"]: cp["repetition"] for cp in result["checkpoints"]}
        print(f"{label:<30} {tok_s:>6.1f} {result['total_swaps']:>6} "
              f"{reps.get(300, 0):>8.2f} {reps.get(500, 0):>8.2f} {reps.get(1000, 0):>8.2f}")


if __name__ == "__main__":
    main()
