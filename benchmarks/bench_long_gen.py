"""Task 12: Long generation test (500 and 1000 tokens).

Verify no memory growth, stable speed, coherent output throughout.
"""

import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_fallback_stats,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 192
PROMPT = "Write a comprehensive tutorial on building a web application with Python and Flask, including database integration, authentication, and deployment"


def run_long_gen(max_tokens):
    print(f"\n{'='*50}")
    print(f"LONG GENERATION: {max_tokens} tokens")
    print(f"{'='*50}")

    model_path = hf_repo_to_path(MODEL)
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
    mx.eval(model.parameters())

    mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, CAPACITY)
    mem_start = mx.get_active_memory() / 1e9
    print(f"Memory at start: {mem_start:.2f} GB")

    # Stream generate with checkpoints
    t_start = time.perf_counter()
    token_count = 0
    text_parts = []
    checkpoints = {}  # token_count -> (time, memory)

    checkpoint_at = set()
    quarter = max_tokens // 4
    checkpoint_at.add(quarter)
    checkpoint_at.add(quarter * 2)
    checkpoint_at.add(quarter * 3)
    checkpoint_at.add(max_tokens)

    for response in mlx_lm.stream_generate(model, tokenizer, prompt=PROMPT,
                                             max_tokens=max_tokens):
        text_parts.append(response.text)
        token_count += 1
        if token_count in checkpoint_at:
            t_elapsed = time.perf_counter() - t_start
            mem = mx.get_active_memory() / 1e9
            checkpoints[token_count] = (t_elapsed, mem)
            speed = token_count / t_elapsed
            print(f"  Token {token_count}: {t_elapsed:.1f}s elapsed, {speed:.1f} tok/s, {mem:.2f} GB")

    t_total = time.perf_counter() - t_start
    mem_end = mx.get_active_memory() / 1e9
    fb = get_fallback_stats(model)
    full_text = "".join(text_parts)

    overall_speed = token_count / t_total
    print(f"\nFinal: {token_count} tokens in {t_total:.1f}s ({overall_speed:.1f} tok/s)")
    print(f"Memory: start={mem_start:.2f} GB, end={mem_end:.2f} GB, growth={mem_end-mem_start:+.2f} GB")
    print(f"Fallback: {fb['fallback_rate']:.1%}")

    # Quality check: sample beginning, middle, end
    print(f"\nFirst 200 chars: {full_text[:200]}")
    mid = len(full_text) // 2
    print(f"\nMiddle 200 chars: {full_text[mid:mid+200]}")
    print(f"\nLast 200 chars: {full_text[-200:]}")

    del model
    mx.metal.clear_cache()

    return {
        "max_tokens": max_tokens,
        "actual_tokens": token_count,
        "total_time": t_total,
        "speed": overall_speed,
        "mem_start": mem_start,
        "mem_end": mem_end,
        "mem_growth": mem_end - mem_start,
        "fallback": fb["fallback_rate"],
        "checkpoints": checkpoints,
        "text_start": full_text[:200],
        "text_mid": full_text[mid:mid+200],
        "text_end": full_text[-200:],
    }


def main():
    results = []
    for max_tok in [500, 1000]:
        r = run_long_gen(max_tok)
        results.append(r)

    with open("PATH_REMOVED", "a") as f:
        f.write("\n## 12. Long Generation Test\n\n")
        f.write(f"Capacity {CAPACITY}, predictive mode on English tutorial prompt.\n\n")

        f.write("| Metric | 500 tokens | 1000 tokens |\n")
        f.write("|--------|-----------|------------|\n")
        r500, r1000 = results[0], results[1]
        f.write(f"| Total time | {r500['total_time']:.1f}s | {r1000['total_time']:.1f}s |\n")
        f.write(f"| Speed | {r500['speed']:.1f} tok/s | {r1000['speed']:.1f} tok/s |\n")
        f.write(f"| Memory start | {r500['mem_start']:.2f} GB | {r1000['mem_start']:.2f} GB |\n")
        f.write(f"| Memory end | {r500['mem_end']:.2f} GB | {r1000['mem_end']:.2f} GB |\n")
        f.write(f"| Memory growth | {r500['mem_growth']:+.2f} GB | {r1000['mem_growth']:+.2f} GB |\n")
        f.write(f"| Fallback | {r500['fallback']:.1%} | {r1000['fallback']:.1%} |\n\n")

        f.write("### Speed Over Time\n\n")
        for r in results:
            f.write(f"**{r['max_tokens']} tokens:**\n")
            for tok, (t, mem) in sorted(r["checkpoints"].items()):
                f.write(f"- Token {tok}: {t:.1f}s, {tok/t:.1f} tok/s, {mem:.2f} GB\n")
            f.write("\n")

        f.write("### Output Coherence\n\n")
        for r in results:
            f.write(f"**{r['max_tokens']} tokens — Start:** `{r['text_start'][:100]}...`\n\n")
            f.write(f"**{r['max_tokens']} tokens — Middle:** `{r['text_mid'][:100]}...`\n\n")
            f.write(f"**{r['max_tokens']} tokens — End:** `{r['text_end'][-100:]}...`\n\n")

        f.write("### Analysis\n\n")
        f.write(f"Memory growth: {r500['mem_growth']:+.2f} GB (500 tok), {r1000['mem_growth']:+.2f} GB (1000 tok). ")
        if abs(r1000['mem_growth']) < 0.5:
            f.write("Stable — no memory leak detected.\n\n")
        else:
            f.write(f"Growth detected — investigate KV cache accumulation.\n\n")

    print("\nResults appended to PATH_REMOVED")


if __name__ == "__main__":
    main()
