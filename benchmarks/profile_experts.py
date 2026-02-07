"""Profile expert routing across diverse prompts to identify universal experts.

Runs warmup + upgrade cycles for each prompt, records per-layer expert activations,
and identifies "universal" experts activated in >threshold fraction of prompts.

Usage:
    /Users/muhash/mlx-lm/.venv/bin/python profile_experts.py [capacity] [threshold] [output_path]
"""

import json
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
    _find_switch_mlp, CachedQuantizedSwitchLinear, PredictiveCachedSwitchLinear,
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


def collect_expert_activations(model, tokenizer, model_path, prompt, capacity):
    """Run warmup + upgrade for one prompt, return per-layer expert sets."""
    reset_to_cached(model, model_path, capacity)

    mlx_lm.generate(model, tokenizer, prompt=prompt,
                    max_tokens=WARMUP_TOKENS, verbose=False)

    # Harvest discovered experts from LCP caches before upgrade
    layer_experts = {}
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj")
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        layer_experts[i] = set(int(eid) for eid in proj._cache.all_seen)

    upgrade_to_predictive(model, model_path, capacity)

    # Also drain _indices_buffer from the predictive cache for extra coverage
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache
        for indices in cache._indices_buffer:
            flat = np.asarray(indices.reshape(-1))
            if i in layer_experts:
                layer_experts[i].update(int(x) for x in np.unique(flat))
        cache._indices_buffer.clear()

    mx.clear_cache()
    return layer_experts


def main():
    capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 208
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    output_path = sys.argv[3] if len(sys.argv) > 3 else "universal_experts.json"

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")
    print(f"Capacity: {capacity}, Threshold: {threshold}")
    print(f"Prompts: {len(PROMPTS)}")

    print("Loading model with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)

    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=capacity,
                                   predictive=True)
    print(f"Replaced {replaced} modules")

    mx.eval(model.parameters())
    print(f"Non-expert params loaded: {mx.get_active_memory() / 1e9:.1f} GB")

    # Bootstrap: first warmup + upgrade so reset_to_cached works in the loop
    print(f"\nBootstrap warmup...")
    mlx_lm.generate(model, tokenizer, prompt=PROMPTS[0],
                    max_tokens=WARMUP_TOKENS, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)
    print(f"Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    activation_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    num_prompts = len(PROMPTS)
    t_total = time.perf_counter()

    for p_idx, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        layer_experts = collect_expert_activations(
            model, tokenizer, model_path, prompt, capacity)
        elapsed = time.perf_counter() - t0

        for layer_idx, experts in layer_experts.items():
            for eid in experts:
                activation_counts[layer_idx][eid] += 1

        n_experts = sum(len(e) for e in layer_experts.values())
        print(f"  [{p_idx + 1}/{num_prompts}] {elapsed:.1f}s, "
              f"{len(layer_experts)} layers, {n_experts} total activations: "
              f"{prompt[:50]}...")

    print(f"\nTotal profiling time: {time.perf_counter() - t_total:.0f}s")

    # Build results
    result = {
        "num_prompts": num_prompts,
        "threshold": threshold,
        "capacity": capacity,
        "layers": {},
    }

    total_universal = 0
    for layer_idx in sorted(activation_counts):
        counts = activation_counts[layer_idx]
        min_count = int(threshold * num_prompts)
        universal = sorted(eid for eid, cnt in counts.items() if cnt >= min_count)
        total_universal += len(universal)

        result["layers"][str(layer_idx)] = {
            "universal": universal,
            "activation_counts": {str(eid): cnt for eid, cnt in sorted(counts.items())},
            "total_unique": len(counts),
        }

    output = Path(output_path)
    output.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {output}")

    # Summary
    layers = result["layers"]
    universal_per_layer = [len(v["universal"]) for v in layers.values()]
    unique_per_layer = [v["total_unique"] for v in layers.values()]
    print(f"\n--- Summary ---")
    print(f"Prompts profiled: {num_prompts}")
    print(f"Threshold: {threshold} ({int(threshold * num_prompts)}/{num_prompts} prompts)")
    print(f"MoE layers: {len(layers)}")
    print(f"Universal experts per layer: "
          f"min={min(universal_per_layer)} median={sorted(universal_per_layer)[len(universal_per_layer)//2]} "
          f"max={max(universal_per_layer)} total={total_universal}")
    print(f"Unique experts per layer: "
          f"min={min(unique_per_layer)} median={sorted(unique_per_layer)[len(unique_per_layer)//2]} "
          f"max={max(unique_per_layer)}")


if __name__ == "__main__":
    main()
