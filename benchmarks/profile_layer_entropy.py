"""Profile per-layer routing entropy and working set sizes for adaptive allocation.

Runs warmup + upgrade for diverse prompts, collects per-layer expert activation
frequencies, computes Shannon entropy and working set profiles, then uses the
MoEpic greedy algorithm to compute optimal per-layer cache allocations.

Usage:
    PATH_REMOVED profile_layer_entropy.py [capacity] [total_budget] [output_path]
"""

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, reset_to_cached,
    CachedQuantizedSwitchLinear, compute_adaptive_allocations,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
WARMUP_TOKENS = 10

PROMPTS = [
    "Explain general relativity in simple terms",
    "What caused World War I?",
    "Write an A* pathfinding implementation in Python",
    "Build a real-time chat application using React",
    "Implement a concurrent hashmap in Rust",
    "Prove there are infinitely many prime numbers",
    "用中文解释量子计算的基本原理",
    "写一首关于春天的中文诗",
    "人工知能の未来について日本語でエッセイを書いてください",
    "A farmer needs to cross a river with a wolf, goat, and cabbage",
    "Write a sorting algorithm comparison with Big-O analysis",
    "Explain the difference between TCP and UDP",
    "Write a Python decorator for memoization",
    "Implement binary search tree in JavaScript",
    "Explain machine learning to a 10 year old",
    "Write a REST API with FastAPI and SQLAlchemy",
    "什么是区块链技术？用简单的语言解释",
    "Write a recursive descent parser in C",
    "Explain the CAP theorem with examples",
    "Design a URL shortener system",
    "Write a matrix multiplication in NumPy",
    "Explain how transformers work in NLP",
]


def compute_entropy(counts):
    """Shannon entropy of a frequency distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def main():
    capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 208
    total_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 208 * 48
    output_path = sys.argv[3] if len(sys.argv) > 3 else "layer_entropy.json"

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")
    print(f"Capacity: {capacity}, Total budget: {total_budget}")

    print("Loading model...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=capacity,
                                   predictive=True)
    mx.eval(model.parameters())
    print(f"Replaced {replaced} modules, {mx.get_active_memory() / 1e9:.1f} GB")

    # Initial warmup + upgrade
    mlx_lm.generate(model, tokenizer, prompt=PROMPTS[0],
                    max_tokens=WARMUP_TOKENS, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)

    # Accumulate per-layer activation counts across all prompts
    layer_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for p_idx, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        reset_to_cached(model, model_path, capacity)
        mlx_lm.generate(model, tokenizer, prompt=prompt,
                        max_tokens=WARMUP_TOKENS, verbose=False)

        for i, layer in enumerate(model.layers):
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
                continue
            proj = getattr(layer.mlp.switch_mlp, "gate_proj")
            if not isinstance(proj, CachedQuantizedSwitchLinear):
                continue
            for eid, freq in proj._cache.frequency.items():
                layer_counts[i][eid] += freq

        upgrade_to_predictive(model, model_path, capacity)
        elapsed = time.perf_counter() - t0
        print(f"  [{p_idx + 1}/{len(PROMPTS)}] {elapsed:.1f}s: {prompt[:50]}...")

    # Compute per-layer profiles
    layer_profiles = {}
    for layer_idx in sorted(layer_counts):
        counts = layer_counts[layer_idx]
        entropy = compute_entropy(counts)
        # Working set: sorted by frequency descending
        working_set = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        unique_count = len(counts)

        layer_profiles[layer_idx] = {
            "working_set": working_set,
            "entropy": entropy,
            "unique_count": unique_count,
        }

    # Run MoEpic greedy allocation
    print(f"\nRunning MoEpic greedy allocation (budget={total_budget})...")
    result = compute_adaptive_allocations(layer_profiles, total_budget)
    allocations = result["allocations"]
    miss_rates = result["miss_rates"]

    # Build output
    output = {
        "num_prompts": len(PROMPTS),
        "capacity_per_layer": capacity,
        "total_budget": total_budget,
        "greedy_iterations": result["iterations"],
        "layers": {},
    }

    for layer_idx in sorted(layer_profiles):
        profile = layer_profiles[layer_idx]
        output["layers"][str(layer_idx)] = {
            "entropy": profile["entropy"],
            "unique_count": profile["unique_count"],
            "allocation": allocations[layer_idx],
            "estimated_miss_rate": miss_rates[layer_idx],
            "top_10_experts": [eid for eid, _ in profile["working_set"][:10]],
            "activation_counts": {str(eid): cnt for eid, cnt in profile["working_set"]},
        }

    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"Results saved to {output_path}")

    # Summary
    alloc_list = [allocations[li] for li in sorted(allocations)]
    entropy_list = [layer_profiles[li]["entropy"] for li in sorted(layer_profiles)]
    miss_list = [miss_rates[li] for li in sorted(miss_rates)]
    unique_list = [layer_profiles[li]["unique_count"] for li in sorted(layer_profiles)]

    print(f"\n--- Summary ---")
    print(f"MoE layers: {len(layer_profiles)}")
    print(f"Total budget: {total_budget} ({total_budget // len(layer_profiles)} uniform)")
    print(f"Greedy iterations: {result['iterations']}")
    print(f"Entropy: min={min(entropy_list):.2f} median={sorted(entropy_list)[len(entropy_list)//2]:.2f} "
          f"max={max(entropy_list):.2f}")
    print(f"Unique experts: min={min(unique_list)} median={sorted(unique_list)[len(unique_list)//2]} "
          f"max={max(unique_list)}")
    print(f"Allocations: min={min(alloc_list)} median={sorted(alloc_list)[len(alloc_list)//2]} "
          f"max={max(alloc_list)}")
    print(f"Est miss rates: min={min(miss_list):.4f} median={sorted(miss_list)[len(miss_list)//2]:.4f} "
          f"max={max(miss_list):.4f}")

    # Compare with uniform
    uniform_cap = total_budget // len(layer_profiles)
    uniform_miss = sum(
        1.0 - sum(cnt for _, cnt in layer_profiles[li]["working_set"][:uniform_cap])
        / max(sum(cnt for _, cnt in layer_profiles[li]["working_set"]), 1)
        for li in sorted(layer_profiles)
    ) / len(layer_profiles)
    adaptive_miss = sum(miss_list) / len(miss_list)
    print(f"\nUniform avg miss rate: {uniform_miss:.4f}")
    print(f"Adaptive avg miss rate: {adaptive_miss:.4f}")
    if uniform_miss > 0:
        print(f"Improvement: {(uniform_miss - adaptive_miss) / uniform_miss:.1%}")


if __name__ == "__main__":
    main()
