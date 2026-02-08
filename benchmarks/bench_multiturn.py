"""Benchmark multi-turn session behavior: memory growth, KV cache, and generation quality.

Simulates a coding-agent session by running multiple prompts sequentially on the
same model instance. Tracks active memory, generation speed, and output quality
across turns to identify unbounded growth or degradation.

Usage:
    PATH_REMOVED benchmarks/bench_multiturn.py [--model MODEL] [--turns N] [--tokens N]

Examples:
    .../python benchmarks/bench_multiturn.py --model qwen --turns 20
    .../python benchmarks/bench_multiturn.py --model mixtral --turns 10 --tokens 100
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive,
    upgrade_to_predictive_with_pinning, load_universal_profile,
    get_fallback_stats, fast_delta_warmup,
    _find_switch_mlp, _detect_num_experts,
)

MODEL_PRESETS = {
    "qwen": ("mlx-community/Qwen3-Coder-Next-4bit", 208),
    "mixtral": ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 6),
    "glm": ("mlx-community/GLM-4.7-Flash-4bit", 48),
}

# Prompts that simulate a coding-agent session: varied domains, lengths, languages
TURN_PROMPTS = [
    "Write a Python function to reverse a linked list",
    "Now add type hints and a docstring to that function",
    "Explain the time complexity of your implementation",
    "Write unit tests for the reverse function using pytest",
    "Implement a binary search in Rust",
    "用中文解释什么是快速排序",
    "Convert the Rust binary search to Python",
    "Write a Dockerfile for a FastAPI application",
    "Explain the difference between TCP and UDP",
    "Implement a thread-safe queue in Python",
    "Write a SQL query to find duplicate records",
    "Design a rate limiter using the token bucket algorithm",
    "Explain how garbage collection works in Python",
    "Write a regex to validate email addresses",
    "Implement merge sort with O(1) extra space",
    "Write a bash script to monitor disk usage",
    "Explain the CAP theorem with a real-world example",
    "Implement a simple LRU cache in Python",
    "Write a WebSocket server in Python",
    "Explain how TLS handshake works",
]


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
    """Measure n-gram repetition ratio as a quality proxy."""
    words = text.split()
    if len(words) < ngram_size:
        return 0.0
    ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn session memory benchmark")
    parser.add_argument("--model", "-m", default="qwen",
                        help="Model preset (qwen/mixtral/glm) or HuggingFace name")
    parser.add_argument("--capacity", "-c", type=int, default=None)
    parser.add_argument("--turns", "-n", type=int, default=20,
                        help="Number of conversation turns (default: 20)")
    parser.add_argument("--tokens", "-t", type=int, default=200,
                        help="Max tokens per turn (default: 200)")
    parser.add_argument("--profile", "-p", default=None,
                        help="Path to universal expert profile JSON for pinning")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: multiturn_<model>.json)")
    args = parser.parse_args()

    if args.model in MODEL_PRESETS:
        model_name, default_capacity = MODEL_PRESETS[args.model]
        short_name = args.model
    else:
        model_name = args.model
        default_capacity = 208
        short_name = model_name.split("/")[-1].lower()

    capacity = args.capacity if args.capacity is not None else default_capacity
    output_path = args.output or f"multiturn_{short_name}.json"
    use_chat_template = args.model in ("mixtral", "glm") or "instruct" in model_name.lower()
    num_turns = min(args.turns, len(TURN_PROMPTS))

    model_path = hf_repo_to_path(model_name)
    print(f"Model: {model_name}")
    print(f"Capacity: {capacity}, Turns: {num_turns}, Tokens/turn: {args.tokens}")
    print(f"Chat template: {use_chat_template}")

    # --- Startup ---
    print("\nLoading model...")
    model, tokenizer = mlx_lm.load(model_name, lazy=True)

    moe_layers = 0
    num_experts = 0
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            moe_layers += 1
            num_experts = _detect_num_experts(switch)

    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())
    mem_base = mx.get_active_memory() / 1e9
    print(f"Base memory (non-expert): {mem_base:.2f} GB")

    # Warmup + upgrade
    warmup_prompt = apply_chat_template(tokenizer, TURN_PROMPTS[0]) if use_chat_template else TURN_PROMPTS[0]
    mlx_lm.generate(model, tokenizer, prompt=warmup_prompt, max_tokens=10, verbose=False)
    mx.set_cache_limit(0)
    if args.profile:
        profile = load_universal_profile(args.profile)
        upgrade_to_predictive_with_pinning(model, model_path, capacity, profile, pin_threshold=0.5)
        print(f"Pinning: {args.profile}")
    else:
        upgrade_to_predictive(model, model_path, capacity)
    mx.set_cache_limit(mx.device_info()["memory_size"] // 4)

    mem_after_upgrade = mx.get_active_memory() / 1e9
    print(f"Memory after upgrade: {mem_after_upgrade:.2f} GB")

    if hasattr(mx, "set_wired_limit"):
        wired = min(int(mx.get_active_memory()), int(mx.device_info()["memory_size"] * 0.75))
        mx.set_wired_limit(wired)
        print(f"Wired: {wired / 1e9:.1f} GB")

    # --- Multi-turn loop ---
    print(f"\n{'='*70}")
    print(f"Starting {num_turns}-turn session")
    print(f"{'='*70}\n")

    results = []
    for turn_idx in range(num_turns):
        raw_prompt = TURN_PROMPTS[turn_idx % len(TURN_PROMPTS)]
        prompt = apply_chat_template(tokenizer, raw_prompt) if use_chat_template else raw_prompt

        mem_before = mx.get_active_memory() / 1e9

        # Delta warmup to simulate prompt switching
        if turn_idx > 0:
            t_delta = time.perf_counter()
            fast_delta_warmup(model, tokenizer, model_path, prompt, discovery_tokens=5)
            delta_time = time.perf_counter() - t_delta
        else:
            delta_time = 0.0

        t_gen = time.perf_counter()
        output = mlx_lm.generate(model, tokenizer, prompt=prompt,
                                  max_tokens=args.tokens, verbose=False)
        gen_time = time.perf_counter() - t_gen
        tok_per_s = args.tokens / gen_time if gen_time > 0 else 0

        mem_after = mx.get_active_memory() / 1e9
        repetition = measure_repetition(output)
        fb = get_fallback_stats(model)

        turn_result = {
            "turn": turn_idx,
            "prompt": raw_prompt[:60],
            "mem_before_gb": round(mem_before, 3),
            "mem_after_gb": round(mem_after, 3),
            "mem_delta_mb": round((mem_after - mem_before) * 1024, 1),
            "delta_warmup_s": round(delta_time, 2),
            "gen_time_s": round(gen_time, 2),
            "tok_per_s": round(tok_per_s, 1),
            "repetition": round(repetition, 3),
            "fallback_rate": round(fb["fallback_rate"], 4),
            "output_len": len(output),
        }
        results.append(turn_result)

        mem_trend = "+" if turn_result["mem_delta_mb"] > 10 else "=" if turn_result["mem_delta_mb"] > -10 else "-"
        print(f"Turn {turn_idx:2d} | {mem_after:.2f} GB ({mem_trend}{abs(turn_result['mem_delta_mb']):.0f} MB) | "
              f"{tok_per_s:.1f} tok/s | rep={repetition:.3f} | fb={fb['fallback_rate']:.2%} | "
              f"{raw_prompt[:40]}...")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SESSION SUMMARY")
    print(f"{'='*70}")

    mems = [r["mem_after_gb"] for r in results]
    speeds = [r["tok_per_s"] for r in results]
    reps = [r["repetition"] for r in results]

    mem_growth = mems[-1] - mems[0]
    speed_drift = speeds[-1] - speeds[0]

    print(f"Turns: {num_turns}")
    print(f"Memory: {mems[0]:.2f} → {mems[-1]:.2f} GB (growth: {mem_growth:+.2f} GB)")
    print(f"Speed:  {speeds[0]:.1f} → {speeds[-1]:.1f} tok/s (drift: {speed_drift:+.1f})")
    print(f"Repetition: min={min(reps):.3f} max={max(reps):.3f} mean={sum(reps)/len(reps):.3f}")
    print(f"Peak memory: {max(mems):.2f} GB")

    if mem_growth > 1.0:
        print(f"\n*** WARNING: Memory grew {mem_growth:.1f} GB over {num_turns} turns — possible leak ***")
    if speed_drift < -5.0:
        print(f"\n*** WARNING: Speed dropped {abs(speed_drift):.1f} tok/s — possible degradation ***")

    full_result = {
        "model": model_name,
        "capacity": capacity,
        "num_experts": num_experts,
        "moe_layers": moe_layers,
        "tokens_per_turn": args.tokens,
        "num_turns": num_turns,
        "profile": args.profile,
        "mem_base_gb": round(mem_base, 3),
        "mem_final_gb": round(mems[-1], 3),
        "mem_growth_gb": round(mem_growth, 3),
        "turns": results,
    }

    Path(output_path).write_text(json.dumps(full_result, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
