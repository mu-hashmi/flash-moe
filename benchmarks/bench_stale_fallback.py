"""Task 7: Stale cache fallback quality analysis.

Warmup on English coding, upgrade to predictive, then generate cross-domain
WITHOUT delta warmup to measure how stale-cache fallback affects output quality.
"""

import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_fallback_stats,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 192
BASE_PROMPT = "Write a Python implementation of A* pathfinding with visualization"

TEST_PROMPTS = [
    ("Chinese: quantum", "用中文详细解释量子计算的基本原理"),
    ("Japanese: AI essay", "日本語で、人工知能の未来について500字のエッセイを書いてください"),
    ("English: history", "Write a detailed analysis of the causes of World War I"),
    ("Math: primes", "Prove that there are infinitely many prime numbers"),
]


def main():
    model_path = hf_repo_to_path(MODEL)
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
    mx.eval(model.parameters())
    print(f"Base memory: {mx.get_active_memory() / 1e9:.1f} GB")

    # Warmup on English coding
    mlx_lm.generate(model, tokenizer, prompt=BASE_PROMPT, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, CAPACITY)
    mem = mx.get_active_memory() / 1e9
    print(f"After upgrade on English coding: {mem:.1f} GB")

    results = []

    for label, prompt in TEST_PROMPTS:
        print(f"\n{'='*50}")
        print(f"STALE CACHE TEST: {label}")
        print(f"{'='*50}")

        # Generate WITHOUT delta warmup
        t0 = time.perf_counter()
        out = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        t_gen = time.perf_counter() - t0
        fb = get_fallback_stats(model)
        speed = 50 / t_gen
        preview = out[:200].replace('\n', ' ')
        print(f"Speed: {speed:.1f} tok/s")
        print(f"Fallback: {fb['fallback_rate']:.1%} ({fb['total_fallbacks']}/{fb['total_requests']})")
        print(f"Output: {preview}")

        results.append({
            "label": label,
            "speed": speed,
            "fallback_rate": fb["fallback_rate"],
            "total_fallbacks": fb["total_fallbacks"],
            "total_requests": fb["total_requests"],
            "output": preview,
        })

    # Write to final_validation.md
    with open("/Users/muhash/flash-moe/final_validation.md", "a") as f:
        f.write("\n## 7. Stale Cache Fallback Quality\n\n")
        f.write(f"Warmed up on English coding (A* pathfinding), upgraded to predictive at capacity {CAPACITY}.\n")
        f.write("Then generated 50 tokens on cross-domain prompts WITHOUT delta warmup.\n\n")
        f.write("| Prompt | tok/s | Fallback Rate | Fallbacks/Requests | Quality |\n")
        f.write("|--------|-------|--------------|-------------------|----------|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['speed']:.1f} | {r['fallback_rate']:.1%} | {r['total_fallbacks']}/{r['total_requests']} | see below |\n")

        f.write("\n### Output Samples\n\n")
        for r in results:
            f.write(f"**{r['label']}:** `{r['output'][:150]}...`\n\n")

        f.write("### Analysis\n\n")
        avg_fb = sum(r["fallback_rate"] for r in results) / len(results)
        f.write(f"**Average fallback rate:** {avg_fb:.1%}\n\n")
        f.write("Without delta warmup, the model uses experts cached for English coding to generate cross-domain output. ")
        f.write("The fallback mechanism substitutes the nearest cached expert when the router-requested expert isn't in cache. ")
        f.write("Quality depends on how much expert overlap exists between domains.\n\n")

    print("\nResults appended to /Users/muhash/flash-moe/final_validation.md")


if __name__ == "__main__":
    main()
