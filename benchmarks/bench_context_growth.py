"""Benchmark true multi-turn context: concatenating prior turns into the prompt.

Unlike bench_multiturn.py which resets context each turn, this simulates a real
coding agent by building up conversation history. Monitors prompt length growth,
KV cache memory impact, and when/if max_kv_size is needed.

Usage:
    /Users/muhash/mlx-lm/.venv/bin/python benchmarks/bench_context_growth.py [--model MODEL] [--turns N] [--tokens N]
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
    get_fallback_stats, _find_switch_mlp, _detect_num_experts,
)

MODEL_PRESETS = {
    "qwen": ("mlx-community/Qwen3-Coder-Next-4bit", 208),
    "mixtral": ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 6),
    "glm": ("mlx-community/GLM-4.7-Flash-4bit", 48),
}

TURN_PROMPTS = [
    "Write a Python function to reverse a linked list",
    "Now add type hints and a docstring to that function",
    "Explain the time complexity of your implementation",
    "Write unit tests for the reverse function using pytest",
    "Refactor the function to handle edge cases",
    "Now implement a doubly linked list version",
    "Add an iterator protocol to the linked list class",
    "Write a benchmark comparing your implementation to collections.deque",
    "Add thread safety with a lock",
    "Write a comprehensive docstring for the whole module",
]


def apply_chat_template(tokenizer, messages):
    """Format a list of (role, content) pairs using the tokenizer's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": r, "content": c} for r, c in messages],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
    return "\n\n".join(f"{r}: {c}" for r, c in messages)


def measure_repetition(text, ngram_size=4):
    words = text.split()
    if len(words) < ngram_size:
        return 0.0
    ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn context growth benchmark")
    parser.add_argument("--model", "-m", default="qwen")
    parser.add_argument("--capacity", "-c", type=int, default=None)
    parser.add_argument("--turns", "-n", type=int, default=10)
    parser.add_argument("--tokens", "-t", type=int, default=150,
                        help="Max tokens per turn (default: 150)")
    parser.add_argument("--profile", "-p", default=None,
                        help="Profile JSON for pinning")
    parser.add_argument("--max-kv-size", type=int, default=None,
                        help="If set, pass max_kv_size to generate (sliding window)")
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
    output_path = args.output or f"context_growth_{short_name}.json"
    use_chat = args.model in ("mixtral", "glm") or "instruct" in model_name.lower()
    num_turns = min(args.turns, len(TURN_PROMPTS))

    model_path = hf_repo_to_path(model_name)
    print(f"Model: {model_name}")
    print(f"Capacity: {capacity}, Turns: {num_turns}, Tokens/turn: {args.tokens}")
    if args.max_kv_size:
        print(f"max_kv_size: {args.max_kv_size}")

    print("\nLoading model...")
    model, tokenizer = mlx_lm.load(model_name, lazy=True)

    moe_layers = num_experts = 0
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            moe_layers += 1
            num_experts = _detect_num_experts(switch)

    enable_lazy_experts(model, model_path, cache_capacity_per_layer=capacity, predictive=True)
    mx.eval(model.parameters())

    warmup_prompt = "Hello"
    if use_chat:
        warmup_prompt = apply_chat_template(tokenizer, [("user", "Hello")])
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

    # Build conversation history turn by turn
    print(f"\n{'='*70}")
    print(f"Starting {num_turns}-turn session with growing context")
    print(f"{'='*70}\n")

    conversation = []
    results = []

    for turn_idx in range(num_turns):
        user_msg = TURN_PROMPTS[turn_idx]
        conversation.append(("user", user_msg))

        if use_chat:
            prompt = apply_chat_template(tokenizer, conversation)
        else:
            prompt = "\n\n".join(f"{r}: {c}" for r, c in conversation)
            prompt += "\nassistant: "

        prompt_tokens = len(tokenizer.encode(prompt))
        mem_before = mx.get_active_memory() / 1e9

        gen_kwargs = {"max_tokens": args.tokens, "verbose": False}
        if args.max_kv_size:
            gen_kwargs["max_kv_size"] = args.max_kv_size

        t_gen = time.perf_counter()
        output = mlx_lm.generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        gen_time = time.perf_counter() - t_gen

        # Truncate output for context to avoid blowup
        output_trimmed = output[:500] if len(output) > 500 else output
        conversation.append(("assistant", output_trimmed))

        mem_after = mx.get_active_memory() / 1e9
        tok_per_s = args.tokens / gen_time if gen_time > 0 else 0
        repetition = measure_repetition(output)
        fb = get_fallback_stats(model)

        turn_result = {
            "turn": turn_idx,
            "prompt_tokens": prompt_tokens,
            "mem_before_gb": round(mem_before, 3),
            "mem_after_gb": round(mem_after, 3),
            "mem_delta_mb": round((mem_after - mem_before) * 1024, 1),
            "gen_time_s": round(gen_time, 2),
            "tok_per_s": round(tok_per_s, 1),
            "repetition": round(repetition, 3),
            "fallback_rate": round(fb["fallback_rate"], 4),
        }
        results.append(turn_result)

        print(f"Turn {turn_idx:2d} | ctx={prompt_tokens:5d} tok | {mem_after:.2f} GB | "
              f"{tok_per_s:.1f} tok/s | rep={repetition:.3f} | {user_msg[:40]}...")

    # Summary
    print(f"\n{'='*70}")
    print("SESSION SUMMARY")
    print(f"{'='*70}")

    mems = [r["mem_after_gb"] for r in results]
    ctxs = [r["prompt_tokens"] for r in results]
    speeds = [r["tok_per_s"] for r in results]

    print(f"Turns: {num_turns}")
    print(f"Context: {ctxs[0]} → {ctxs[-1]} tokens ({ctxs[-1]/ctxs[0]:.1f}x growth)")
    print(f"Memory: {mems[0]:.2f} → {mems[-1]:.2f} GB (growth: {mems[-1]-mems[0]:+.2f} GB)")
    print(f"Speed:  {speeds[0]:.1f} → {speeds[-1]:.1f} tok/s")
    print(f"Peak memory: {max(mems):.2f} GB")

    if max(mems) > mx.device_info()["memory_size"] / 1e9 * 0.85:
        print(f"\n*** WARNING: Peak memory near device limit ***")

    full_result = {
        "model": model_name,
        "capacity": capacity,
        "tokens_per_turn": args.tokens,
        "max_kv_size": args.max_kv_size,
        "profile": args.profile,
        "num_turns": num_turns,
        "context_growth": {"start": ctxs[0], "end": ctxs[-1]},
        "mem_growth_gb": round(mems[-1] - mems[0], 3),
        "turns": results,
    }

    Path(output_path).write_text(json.dumps(full_result, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
