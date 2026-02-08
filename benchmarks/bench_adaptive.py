"""Benchmark per-layer adaptive expert budget vs uniform capacity.

Uses adaptive_capacity_upgrade() which allocates more slots to layers that
discover more experts during warmup, while keeping the total budget equal
to uniform capacity × num_layers.

Usage:
    /Users/muhash/mlx-lm/.venv/bin/python benchmarks/bench_adaptive.py [--model MODEL] [--capacity N] [--tokens N]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive,
    adaptive_capacity_upgrade, get_fallback_stats, measure_fallback,
    load_universal_profile, PredictiveCachedSwitchLinear,
    _find_switch_mlp, _detect_num_experts,
)

MODEL_PRESETS = {
    "qwen": ("mlx-community/Qwen3-Coder-Next-4bit", 208),
    "mixtral": ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 6),
    "glm": ("mlx-community/GLM-4.7-Flash-4bit", 48),
}

PROMPT = "Write a comprehensive tutorial on building a web application with Python and Flask, including project setup, routing, templates, database integration, authentication, and deployment"


def apply_chat_template(tokenizer, text):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
    return text


def measure_repetition(text, ngram_size=4):
    words = text.split()
    if len(words) < ngram_size:
        return 0.0
    ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def apply_pinning(model, profile_path, pin_threshold=0.5):
    """Set pinned_set on each layer's predictive cache from a profile."""
    profile = load_universal_profile(profile_path)
    num_prompts = profile["num_prompts"]
    min_count = int(pin_threshold * num_prompts)
    pinned_count = 0

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache
        layer_data = profile["layers"].get(str(i), {})
        counts = layer_data.get("activation_counts", {})
        universal = {int(eid) for eid, cnt in counts.items() if int(cnt) >= min_count}
        cache.pinned_set = universal & cache.cached_set
        pinned_count += len(cache.pinned_set)

    return pinned_count


def run_config(model_name, model_path, capacity_arg, max_tokens, use_chat,
               adaptive=False, profile_path=None):
    """Run one configuration: uniform or adaptive, optionally with pinning."""
    model, tokenizer = mlx_lm.load(model_name, lazy=True)

    moe_layers = 0
    num_experts = 0
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            moe_layers += 1
            num_experts = _detect_num_experts(switch)

    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity_arg,
                        predictive=True)
    mx.eval(model.parameters())

    prompt = apply_chat_template(tokenizer, PROMPT) if use_chat else PROMPT
    mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    mx.set_cache_limit(0)
    if adaptive:
        total_budget = capacity_arg * moe_layers
        info = adaptive_capacity_upgrade(model, model_path, total_budget,
                                         min_per_layer=32)
        capacities = info["capacities"]
        cap_min = min(capacities)
        cap_max = max(capacities)
        cap_mean = sum(capacities) / len(capacities)
        total_experts = info["total_experts"]
        print(f"  Adaptive: {total_experts} total experts, "
              f"per-layer: min={cap_min} mean={cap_mean:.0f} max={cap_max}")
    else:
        upgrade_to_predictive(model, model_path, capacity_arg)
        total_experts = capacity_arg * moe_layers
        capacities = [capacity_arg] * moe_layers
    mx.set_cache_limit(mx.device_info()["memory_size"] // 4)

    if profile_path:
        n_pinned = apply_pinning(model, profile_path)
        print(f"  Pinned {n_pinned} experts from {profile_path}")

    mem = mx.get_active_memory() / 1e9
    print(f"  Memory: {mem:.2f} GB")

    if hasattr(mx, "set_wired_limit"):
        wired = min(int(mx.get_active_memory()), int(mx.device_info()["memory_size"] * 0.75))
        mx.set_wired_limit(wired)

    # Generate and measure
    t_start = time.perf_counter()
    output = mlx_lm.generate(model, tokenizer, prompt=prompt,
                              max_tokens=max_tokens, verbose=False)
    gen_time = time.perf_counter() - t_start
    tok_per_s = max_tokens / gen_time

    repetition = measure_repetition(output)
    fb = measure_fallback(model)

    result = {
        "adaptive": adaptive,
        "total_experts": total_experts,
        "capacities": capacities,
        "mem_gb": round(mem, 3),
        "tok_per_s": round(tok_per_s, 1),
        "gen_time_s": round(gen_time, 2),
        "repetition": round(repetition, 3),
        "fallback_rate": round(fb["fallback_rate"], 4),
        "fallback_count": fb["total_fallbacks"],
        "output_preview": output[:200],
    }

    del model
    mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Adaptive per-layer budget benchmark")
    parser.add_argument("--model", "-m", default="qwen")
    parser.add_argument("--capacity", "-c", type=int, default=None,
                        help="Uniform capacity (adaptive will use same total budget)")
    parser.add_argument("--tokens", "-t", type=int, default=500)
    parser.add_argument("--profile", "-p", default=None,
                        help="Profile JSON for pinning (enables config C)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    if args.model in MODEL_PRESETS:
        model_name, default_capacity = MODEL_PRESETS[args.model]
        short_name = args.model
    else:
        model_name = args.model
        default_capacity = 208
        short_name = model_name.split("/")[-1].lower()

    capacity = args.capacity or default_capacity
    output_path = args.output or f"adaptive_{short_name}.json"

    use_chat = args.model in ("mixtral", "glm") or "instruct" in model_name.lower()
    model_path = hf_repo_to_path(model_name)

    print(f"Model: {model_name}")
    print(f"Uniform capacity: {capacity}, Tokens: {args.tokens}")

    # Config A: Uniform
    print(f"\n{'='*60}")
    print("CONFIG A: Uniform capacity")
    print(f"{'='*60}")
    result_uniform = run_config(model_name, model_path, capacity, args.tokens,
                                use_chat, adaptive=False)
    print(f"  {result_uniform['tok_per_s']} tok/s, rep={result_uniform['repetition']:.3f}, "
          f"fb={result_uniform['fallback_rate']:.2%}")

    # Config B: Adaptive
    print(f"\n{'='*60}")
    print("CONFIG B: Adaptive per-layer capacity")
    print(f"{'='*60}")
    result_adaptive = run_config(model_name, model_path, capacity, args.tokens,
                                 use_chat, adaptive=True)
    print(f"  {result_adaptive['tok_per_s']} tok/s, rep={result_adaptive['repetition']:.3f}, "
          f"fb={result_adaptive['fallback_rate']:.2%}")

    # Config C: Adaptive + Pinning (if profile provided)
    result_adaptive_pinned = None
    if args.profile:
        print(f"\n{'='*60}")
        print("CONFIG C: Adaptive + Pinning")
        print(f"{'='*60}")
        result_adaptive_pinned = run_config(model_name, model_path, capacity, args.tokens,
                                            use_chat, adaptive=True, profile_path=args.profile)
        print(f"  {result_adaptive_pinned['tok_per_s']} tok/s, rep={result_adaptive_pinned['repetition']:.3f}, "
              f"fb={result_adaptive_pinned['fallback_rate']:.2%}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Experts':>8} {'Memory':>8} {'tok/s':>7} {'Rep':>6} {'Fallback':>9}")
    configs = [("Uniform", result_uniform), ("Adaptive", result_adaptive)]
    if result_adaptive_pinned:
        configs.append(("Adaptive+Pinning", result_adaptive_pinned))
    for label, r in configs:
        print(f"{label:<25} {r['total_experts']:>8} {r['mem_gb']:>7.2f}G "
              f"{r['tok_per_s']:>7.1f} {r['repetition']:>6.3f} {r['fallback_rate']:>8.2%}")

    full_result = {
        "model": model_name,
        "uniform_capacity": capacity,
        "tokens": args.tokens,
        "uniform": result_uniform,
        "adaptive": result_adaptive,
        "adaptive_pinned": result_adaptive_pinned,
    }
    Path(output_path).write_text(json.dumps(full_result, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
