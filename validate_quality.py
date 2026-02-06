"""Validation suite for Phase 2 predictive expert cache.

Tests quality across diverse prompts, warmup length sensitivity, memory growth,
and per-expert memory footprint. Supports dynamic cache updates between tokens.

Usage (run from the mlx-lm venv):
    /path/to/mlx-lm/.venv/bin/python validate_quality.py [experiment]

Experiments:
    quality     — Diverse prompt testing with dynamic cache updates (default)
    warmup      — Warmup length sensitivity (10/25/50/100 tokens)
    memory      — Memory growth over 500-token generation
    expert-size — Verify per-expert memory footprint
"""

import sys
import time
import json
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.generate import stream_generate
from mlx_lm.lazy_experts import (
    enable_lazy_experts,
    upgrade_to_predictive,
    get_cache_stats,
    dynamic_cache_update,
    get_fallback_stats,
    CachedQuantizedSwitchLinear,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 256

TEST_PROMPTS = {
    "python_code": "Write a Python function that implements binary search on a sorted list.",
    "javascript": "Write a JavaScript async function that fetches data from an API and handles errors.",
    "math": "Solve step by step: If f(x) = 3x^2 - 2x + 1, find f'(x) and evaluate f'(4).",
    "chinese": "用中文解释什么是递归算法，并给出一个简单的例子。",
    "spanish": "Explica en español cómo funciona el algoritmo de ordenamiento quicksort.",
    "reasoning": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain your reasoning carefully.",
    "creative": "Write a short poem about a robot learning to paint.",
    "technical": "Explain the difference between TCP and UDP protocols, including when to use each.",
    "debugging": "This Python code has a bug: `def fib(n): return fib(n-1) + fib(n-2)`. Find and fix it.",
    "long_context": (
        "Given the following data structure:\n"
        "users = [{'name': 'Alice', 'age': 30, 'scores': [85, 92, 78]},\n"
        "         {'name': 'Bob', 'age': 25, 'scores': [90, 88, 95]},\n"
        "         {'name': 'Charlie', 'age': 35, 'scores': [70, 75, 80]}]\n"
        "Write a Python function that returns the user with the highest average score, "
        "the overall average across all users, and a sorted list of users by age."
    ),
}


def load_model():
    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")
    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    return model, tokenizer, model_path


def setup_predictive(model, tokenizer, model_path, warmup_prompt, warmup_tokens, capacity):
    """Run LCP warmup and upgrade to predictive mode."""
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    print(f"Evaluating non-expert params...")
    mx.eval(model.parameters())
    mem = mx.get_active_memory() / 1e9
    print(f"Non-expert params: {mem:.1f} GB")

    print(f"Warmup: {warmup_tokens} tokens...")
    mlx_lm.generate(model, tokenizer, prompt=warmup_prompt, max_tokens=warmup_tokens, verbose=False)
    stats = get_cache_stats(model)
    print(f"Warmup hit rate: {stats['total_hit_rate']:.1%}")

    print(f"Upgrading to predictive (capacity={capacity})...")
    upgrade_to_predictive(model, model_path, capacity)
    mem = mx.get_active_memory() / 1e9
    print(f"Post-upgrade memory: {mem:.1f} GB\n")


def generate_with_updates(model, tokenizer, prompt, max_tokens, dynamic=True):
    """Generate tokens with optional dynamic cache updates between tokens.

    Returns (text, per_token_stats, total_time).
    """
    token_stats = []
    text = ""
    total_swaps = 0

    tic = time.perf_counter()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        text += response.text
        if dynamic:
            layer_stats = dynamic_cache_update(model)
            swaps = sum(s["swaps"] for s in layer_stats)
            fallbacks = sum(s["fallbacks"] for s in layer_stats)
            requests = sum(s["requests"] for s in layer_stats)
            total_swaps += swaps
            token_stats.append({
                "token_n": response.generation_tokens,
                "swaps": swaps,
                "fallbacks": fallbacks,
                "requests": requests,
                "fallback_rate": fallbacks / requests if requests > 0 else 0.0,
            })
    total_time = time.perf_counter() - tic

    return text, token_stats, total_time, total_swaps


def experiment_quality():
    """Test diverse prompts with dynamic cache updates."""
    model, tokenizer, model_path = load_model()
    warmup_prompt = "Write a hello world program in Python"
    setup_predictive(model, tokenizer, model_path, warmup_prompt, warmup_tokens=10, capacity=CAPACITY)

    max_tokens = 100
    results = []

    for name, prompt in TEST_PROMPTS.items():
        print(f"--- {name} ---")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        # Reset fallback counters
        for layer in model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
                if hasattr(proj, "_cache"):
                    proj._cache.total_requests = 0
                    proj._cache.total_fallbacks = 0

        text, token_stats, total_time, total_swaps = generate_with_updates(
            model, tokenizer, prompt, max_tokens, dynamic=True
        )
        gen_tokens = len(token_stats)
        tok_s = gen_tokens / total_time if total_time > 0 else 0

        fallback_stats = get_fallback_stats(model)
        early_fb = [s["fallback_rate"] for s in token_stats[:10] if s["requests"] > 0]
        late_fb = [s["fallback_rate"] for s in token_stats[-10:] if s["requests"] > 0]
        early_avg = sum(early_fb) / len(early_fb) if early_fb else 0
        late_avg = sum(late_fb) / len(late_fb) if late_fb else 0

        result = {
            "name": name,
            "tokens": gen_tokens,
            "tok_s": tok_s,
            "total_swaps": total_swaps,
            "overall_fallback_rate": fallback_stats["fallback_rate"],
            "early_fallback_rate": early_avg,
            "late_fallback_rate": late_avg,
            "output_preview": text[:200],
        }
        results.append(result)

        print(f"Output: {text[:150]}{'...' if len(text) > 150 else ''}")
        print(f"Speed: {tok_s:.1f} tok/s | Swaps: {total_swaps}")
        print(f"Fallback rate: overall={fallback_stats['fallback_rate']:.1%} "
              f"early={early_avg:.1%} late={late_avg:.1%}")
        print(f"Memory: {mx.get_active_memory() / 1e9:.1f} GB")
        print()

    # Summary table
    print("=" * 80)
    print(f"{'Prompt':<15} {'tok/s':>6} {'Swaps':>6} {'FB%':>6} {'Early%':>7} {'Late%':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} {r['tok_s']:>6.1f} {r['total_swaps']:>6} "
              f"{r['overall_fallback_rate']:>5.1%} {r['early_fallback_rate']:>6.1%} "
              f"{r['late_fallback_rate']:>6.1%}")
    print("=" * 80)

    avg_fb = sum(r["overall_fallback_rate"] for r in results) / len(results)
    avg_toks = sum(r["tok_s"] for r in results) / len(results)
    print(f"Average: {avg_toks:.1f} tok/s, {avg_fb:.1%} fallback rate")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.1f} GB")


def experiment_warmup():
    """Test warmup length sensitivity."""
    warmup_prompt = "Write a hello world program in Python"
    test_prompt = "Explain the concept of dynamic programming and give an example in Python."
    warmup_lengths = [10, 25, 50, 100]
    gen_tokens = 50

    results = []
    for wl in warmup_lengths:
        print(f"\n{'='*60}")
        print(f"Warmup length: {wl} tokens")
        print(f"{'='*60}")

        model, tokenizer, model_path = load_model()
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
        mx.eval(model.parameters())

        # Run warmup and measure LCP cache coverage
        mlx_lm.generate(model, tokenizer, prompt=warmup_prompt, max_tokens=wl, verbose=False)

        # Count unique experts discovered per layer
        per_layer_seen = []
        for layer in model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
                if isinstance(proj, CachedQuantizedSwitchLinear):
                    per_layer_seen.append(len(proj._cache.all_seen))

        avg_seen = sum(per_layer_seen) / len(per_layer_seen) if per_layer_seen else 0
        min_seen = min(per_layer_seen) if per_layer_seen else 0
        max_seen = max(per_layer_seen) if per_layer_seen else 0

        print(f"Unique experts per layer: avg={avg_seen:.0f} min={min_seen} max={max_seen}")

        # Upgrade and generate
        upgrade_to_predictive(model, model_path, CAPACITY)

        text, token_stats, total_time, total_swaps = generate_with_updates(
            model, tokenizer, test_prompt, gen_tokens, dynamic=True
        )
        fallback_stats = get_fallback_stats(model)

        result = {
            "warmup_tokens": wl,
            "avg_experts_seen": avg_seen,
            "min_experts_seen": min_seen,
            "max_experts_seen": max_seen,
            "gen_fallback_rate": fallback_stats["fallback_rate"],
            "total_swaps": total_swaps,
        }
        results.append(result)
        print(f"Generation fallback rate: {fallback_stats['fallback_rate']:.1%}")
        print(f"Dynamic swaps: {total_swaps}")

        del model
        mx.clear_cache()

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Warmup':>8} {'Avg Seen':>10} {'Min':>5} {'Max':>5} {'FB Rate':>8} {'Swaps':>6}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['warmup_tokens']:>8} {r['avg_experts_seen']:>10.0f} "
              f"{r['min_experts_seen']:>5} {r['max_experts_seen']:>5} "
              f"{r['gen_fallback_rate']:>7.1%} {r['total_swaps']:>6}")
    print(f"{'='*80}")


def experiment_memory():
    """Test memory growth over long generation."""
    model, tokenizer, model_path = load_model()
    setup_predictive(model, tokenizer, model_path,
                     "Write a hello world program in Python",
                     warmup_tokens=10, capacity=CAPACITY)

    prompt = "Write a detailed tutorial on building a web application with Python Flask, including routing, templates, database integration, and deployment."
    max_tokens = 500

    print(f"Generating {max_tokens} tokens with dynamic cache updates...")
    mem_samples = []
    text = ""
    tic = time.perf_counter()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        text += response.text
        dynamic_cache_update(model)
        n = response.generation_tokens
        if n % 50 == 0 or n <= 10:
            mem = mx.get_active_memory() / 1e9
            peak = mx.get_peak_memory() / 1e9
            tps = n / (time.perf_counter() - tic)
            mem_samples.append({"token": n, "active_gb": mem, "peak_gb": peak, "tok_s": tps})
            print(f"  Token {n:>4}: active={mem:.2f} GB  peak={peak:.2f} GB  {tps:.1f} tok/s")

    total_time = time.perf_counter() - tic
    final_tps = response.generation_tokens / total_time

    print(f"\nFinal: {response.generation_tokens} tokens in {total_time:.1f}s ({final_tps:.1f} tok/s)")
    print(f"Active memory: {mx.get_active_memory() / 1e9:.2f} GB")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")

    fallback_stats = get_fallback_stats(model)
    print(f"Fallback rate: {fallback_stats['fallback_rate']:.1%}")

    if mem_samples:
        growth = mem_samples[-1]["active_gb"] - mem_samples[0]["active_gb"]
        print(f"Memory growth (first → last sample): {growth:+.2f} GB")


def experiment_expert_size():
    """Verify per-expert memory footprint from tensor shapes."""
    model_path = hf_repo_to_path(MODEL)
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    shard_path = str(model_path / weight_map[key])

    print("Loading shard to inspect expert tensor shapes...")
    shard = mx.load(shard_path)

    # Compute per-expert size from shapes and dtypes
    dtype_bytes = {"uint32": 4, "bfloat16": 2, "float16": 2, "float32": 4}
    total_per_expert = 0

    for proj in ("gate_proj", "up_proj", "down_proj"):
        print(f"\n  {proj}:")
        for suffix in ("weight", "scales", "biases"):
            full_key = f"model.layers.0.mlp.switch_mlp.{proj}.{suffix}"
            if full_key not in shard:
                continue
            tensor = shard[full_key]
            n_experts = tensor.shape[0]
            per_expert_shape = tensor.shape[1:]
            n_elements = 1
            for d in per_expert_shape:
                n_elements *= d
            dt = str(tensor.dtype).split(".")[-1]
            elem_bytes = dtype_bytes.get(dt, 4)
            per_expert_bytes = n_elements * elem_bytes
            total_per_expert += per_expert_bytes
            print(f"    {suffix}: full={tensor.shape} dtype={tensor.dtype} "
                  f"→ per_expert={per_expert_shape} = {per_expert_bytes/1024:.1f} KB")

    expert_mb = total_per_expert / 1e6
    print(f"\nPer-expert triplet: {expert_mb:.3f} MB ({total_per_expert:,} bytes)")
    print(f"256 experts/layer: {expert_mb * 256 / 1e3:.2f} GB")
    print(f"48 layers × 256: {expert_mb * 256 * 48 / 1e3:.2f} GB")
    print(f"48 layers × 512 (full model experts): {expert_mb * 512 * 48 / 1e3:.2f} GB")


EXPERIMENTS = {
    "quality": experiment_quality,
    "warmup": experiment_warmup,
    "memory": experiment_memory,
    "expert-size": experiment_expert_size,
}

if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "quality"
    if exp not in EXPERIMENTS:
        print(f"Unknown experiment '{exp}'. Available: {', '.join(EXPERIMENTS)}")
        sys.exit(1)
    EXPERIMENTS[exp]()
