"""Task 3: Broad prompt validation across 12 diverse prompts.

Tests both predictive (fresh warmup) and async-delta (switching from English coding) modes
at capacity 192.
"""

import time
import json
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, fast_delta_warmup,
    get_fallback_stats, incremental_delta_warmup,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 192
MAX_TOKENS = 50

PROMPTS = [
    ("English: relativity", "Explain the theory of general relativity in simple terms"),
    ("English: WWI", "Write a detailed analysis of the causes of World War I"),
    ("Code: A* Python", "Write a Python implementation of A* pathfinding with visualization"),
    ("Code: React chat", "Create a React component for a real-time chat interface with WebSocket"),
    ("Code: Rust hashmap", "Implement a concurrent hash map in Rust with fine-grained locking"),
    ("Math: primes", "Prove that there are infinitely many prime numbers"),
    ("Chinese: quantum", "用中文详细解释量子计算的基本原理"),
    ("Chinese: poem", "用中文写一首关于秋天的诗"),
    ("Japanese: AI", "日本語で、人工知能の未来について500字のエッセイを書いてください"),
    ("Logic: river", "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item at a time. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How does the farmer get everything across?"),
    ("Multi-turn", "MULTI_TURN"),  # handled specially
    ("Code+analysis: sorting", "Compare merge sort and quicksort, then implement both in Python with benchmarks"),
]

DELTA_SOURCE_PROMPT = "Write a Python implementation of A* pathfinding with visualization"


def run_predictive(prompt_label, prompt, model_path):
    """Fresh model load, warmup on this prompt, generate."""
    print(f"\n  [Predictive] {prompt_label}")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
    mx.eval(model.parameters())

    # Warmup on this prompt
    mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, CAPACITY)
    mem = mx.get_active_memory() / 1e9

    # Generate
    t0 = time.perf_counter()
    out = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
    t_gen = time.perf_counter() - t0
    fb = get_fallback_stats(model)

    speed = MAX_TOKENS / t_gen
    preview = out[:200].replace('\n', ' ')
    print(f"    {speed:.1f} tok/s, {fb['fallback_rate']:.1%} fallback, {mem:.1f} GB")
    print(f"    Output: {preview[:100]}...")

    del model
    mx.metal.clear_cache()

    return {
        "mode": "predictive",
        "label": prompt_label,
        "speed": speed,
        "fallback": fb["fallback_rate"],
        "memory": mem,
        "output": preview,
    }


def run_async_delta(prompt_label, prompt, model, tokenizer, model_path):
    """Switch from cached state via async delta warmup, stream generate."""
    print(f"\n  [Async-delta] {prompt_label}")

    warmup, disc_stats = incremental_delta_warmup(
        model, tokenizer, model_path, prompt, discovery_tokens=10,
    )

    t_start = time.perf_counter()
    token_count = 0
    text_parts = []
    for response in mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                             max_tokens=MAX_TOKENS):
        text_parts.append(response.text)
        token_count += 1
        if not warmup.is_complete:
            warmup.step(layers_per_step=2)
    t_total = time.perf_counter() - t_start

    fb = get_fallback_stats(model)
    mem = mx.get_active_memory() / 1e9
    speed = token_count / t_total if t_total > 0 else 0
    full_text = "".join(text_parts)
    preview = full_text[:200].replace('\n', ' ')

    prog = warmup.progress
    print(f"    {speed:.1f} tok/s, {fb['fallback_rate']:.1%} fallback, {mem:.1f} GB")
    print(f"    Swaps: {prog['swaps_done']}/{prog['swaps_total']}, discovery: {disc_stats['discovery_time']:.1f}s")
    print(f"    Output: {preview[:100]}...")

    return {
        "mode": "async-delta",
        "label": prompt_label,
        "speed": speed,
        "fallback": fb["fallback_rate"],
        "memory": mem,
        "discovery_time": disc_stats["discovery_time"],
        "total_swaps": disc_stats["total_swaps"],
        "output": preview,
    }


def setup_delta_base(model_path):
    """Load model, warmup on English coding prompt, return model ready for delta switches."""
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
    mx.eval(model.parameters())
    mlx_lm.generate(model, tokenizer, prompt=DELTA_SOURCE_PROMPT, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, CAPACITY)
    print(f"  Base model ready: {mx.get_active_memory() / 1e9:.1f} GB")
    return model, tokenizer


def main():
    model_path = hf_repo_to_path(MODEL)
    results = []

    # Part A: Predictive mode (fresh load per prompt)
    print("=" * 60)
    print("PART A: PREDICTIVE MODE (fresh model per prompt)")
    print("=" * 60)

    for label, prompt in PROMPTS:
        if prompt == "MULTI_TURN":
            # Multi-turn: Python → Chinese → Rust
            print(f"\n  [Predictive] {label}: Python → Chinese → Rust")
            model, tokenizer = mlx_lm.load(MODEL, lazy=True)
            enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
            mx.eval(model.parameters())

            # Stage 1: Python
            p1 = "Write a Python function for binary search"
            mlx_lm.generate(model, tokenizer, prompt=p1, max_tokens=10, verbose=False)
            upgrade_to_predictive(model, model_path, CAPACITY)

            # Stage 2: Delta to Chinese
            p2 = "用中文写一首关于秋天的诗"
            fast_delta_warmup(model, tokenizer, model_path, p2)
            out2 = mlx_lm.generate(model, tokenizer, prompt=p2, max_tokens=20, verbose=False)

            # Stage 3: Delta to Rust
            p3 = "Implement a stack data structure in Rust"
            t0 = time.perf_counter()
            fast_delta_warmup(model, tokenizer, model_path, p3)
            t_delta = time.perf_counter() - t0
            t0 = time.perf_counter()
            out3 = mlx_lm.generate(model, tokenizer, prompt=p3, max_tokens=MAX_TOKENS, verbose=False)
            t_gen = time.perf_counter() - t0
            fb = get_fallback_stats(model)
            mem = mx.get_active_memory() / 1e9

            preview = out3[:200].replace('\n', ' ')
            speed = MAX_TOKENS / t_gen
            print(f"    Final stage (Rust): {speed:.1f} tok/s, {fb['fallback_rate']:.1%} fallback")
            print(f"    Chinese sample: {out2[:80]}...")
            print(f"    Rust output: {preview[:100]}...")

            results.append({
                "mode": "predictive",
                "label": label,
                "speed": speed,
                "fallback": fb["fallback_rate"],
                "memory": mem,
                "output": f"[3-stage] Chinese: {out2[:60]}... | Rust: {preview[:60]}...",
            })
            del model
            mx.metal.clear_cache()
        else:
            r = run_predictive(label, prompt, model_path)
            results.append(r)

    # Part B: Async-delta mode (switch from English coding)
    print("\n" + "=" * 60)
    print("PART B: ASYNC-DELTA MODE (switching from English coding)")
    print("=" * 60)

    model, tokenizer = setup_delta_base(model_path)

    for label, prompt in PROMPTS:
        if prompt == "MULTI_TURN":
            # Multi-turn via delta: already warmed on English coding
            # Switch to Chinese, then Rust
            print(f"\n  [Async-delta] {label}: [base:Python] → Chinese → Rust")

            p2 = "用中文写一首关于秋天的诗"
            warmup2, _ = incremental_delta_warmup(model, tokenizer, model_path, p2, discovery_tokens=10)
            text2 = []
            for resp in mlx_lm.stream_generate(model, tokenizer, prompt=p2, max_tokens=20):
                text2.append(resp.text)
                if not warmup2.is_complete:
                    warmup2.step(layers_per_step=2)

            p3 = "Implement a stack data structure in Rust"
            warmup3, disc3 = incremental_delta_warmup(model, tokenizer, model_path, p3, discovery_tokens=10)
            t0 = time.perf_counter()
            token_count = 0
            text3 = []
            for resp in mlx_lm.stream_generate(model, tokenizer, prompt=p3, max_tokens=MAX_TOKENS):
                text3.append(resp.text)
                token_count += 1
                if not warmup3.is_complete:
                    warmup3.step(layers_per_step=2)
            t_total = time.perf_counter() - t0
            fb = get_fallback_stats(model)
            mem = mx.get_active_memory() / 1e9
            speed = token_count / t_total if t_total > 0 else 0

            ch_out = "".join(text2)[:80]
            rust_out = "".join(text3)[:100]
            print(f"    Rust stage: {speed:.1f} tok/s, {fb['fallback_rate']:.1%} fallback")
            print(f"    Chinese: {ch_out}...")
            print(f"    Rust: {rust_out}...")

            results.append({
                "mode": "async-delta",
                "label": label,
                "speed": speed,
                "fallback": fb["fallback_rate"],
                "memory": mem,
                "discovery_time": disc3["discovery_time"],
                "total_swaps": disc3["total_swaps"],
                "output": f"[3-stage] Chinese: {ch_out[:50]}... | Rust: {rust_out[:50]}...",
            })
        else:
            # Need to re-establish base state for each test
            # Reload base to avoid cumulative drift
            del model
            mx.metal.clear_cache()
            model, tokenizer = setup_delta_base(model_path)
            r = run_async_delta(label, prompt, model, tokenizer, model_path)
            results.append(r)

    # Write results
    with open("PATH_REMOVED", "a") as f:
        f.write("\n## 3. Broad Prompt Validation (12 Prompts, Capacity 192)\n\n")

        f.write("### Predictive Mode (fresh warmup per prompt)\n\n")
        f.write("| # | Prompt | tok/s | Fallback | Memory (GB) | Quality |\n")
        f.write("|---|--------|-------|----------|-------------|----------|\n")
        pred = [r for r in results if r["mode"] == "predictive"]
        for i, r in enumerate(pred, 1):
            q = _assess_quality(r)
            f.write(f"| {i} | {r['label']} | {r['speed']:.1f} | {r['fallback']:.1%} | {r['memory']:.1f} | {q} |\n")

        f.write("\n### Async-Delta Mode (switching from English coding)\n\n")
        f.write("| # | Prompt | tok/s | Fallback | Memory (GB) | Discovery (s) | Swaps | Quality |\n")
        f.write("|---|--------|-------|----------|-------------|--------------|-------|----------|\n")
        delta = [r for r in results if r["mode"] == "async-delta"]
        for i, r in enumerate(delta, 1):
            q = _assess_quality(r)
            disc = r.get("discovery_time", "-")
            swaps = r.get("total_swaps", "-")
            disc_str = f"{disc:.1f}" if isinstance(disc, float) else disc
            f.write(f"| {i} | {r['label']} | {r['speed']:.1f} | {r['fallback']:.1%} | {r['memory']:.1f} | {disc_str} | {swaps} | {q} |\n")

        f.write("\n### Output Samples\n\n")
        for r in results:
            f.write(f"**{r['mode']} — {r['label']}:** `{r['output'][:120]}...`\n\n")

        # Summary stats
        pred_speeds = [r["speed"] for r in pred]
        delta_speeds = [r["speed"] for r in delta]
        f.write("### Summary\n\n")
        f.write(f"- **Predictive mode:** {min(pred_speeds):.1f}–{max(pred_speeds):.1f} tok/s (mean {sum(pred_speeds)/len(pred_speeds):.1f})\n")
        f.write(f"- **Async-delta mode:** {min(delta_speeds):.1f}–{max(delta_speeds):.1f} tok/s (mean {sum(delta_speeds)/len(delta_speeds):.1f})\n")
        coherent_pred = sum(1 for r in pred if "coherent" in _assess_quality(r).lower() or "good" in _assess_quality(r).lower())
        coherent_delta = sum(1 for r in delta if "coherent" in _assess_quality(r).lower() or "good" in _assess_quality(r).lower())
        f.write(f"- **Quality:** {coherent_pred}/{len(pred)} coherent (predictive), {coherent_delta}/{len(delta)} coherent (async-delta)\n\n")

    print("\nResults appended to PATH_REMOVED")

    # Also dump raw JSON for later analysis
    with open("PATH_REMOVED", "w") as f:
        json.dump(results, f, indent=2, default=str)


def _assess_quality(r):
    out = r.get("output", "")
    # Check for repetition patterns
    if any(phrase * 3 in out for phrase in ["输入", "共四句", "不计其名"]):
        return "GARBLED (repetitive)"
    if r.get("fallback", 0) > 0.2:
        return "Degraded (high fallback)"
    return "Coherent"


if __name__ == "__main__":
    main()
