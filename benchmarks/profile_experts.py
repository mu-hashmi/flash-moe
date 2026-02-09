"""Profile expert routing across diverse prompts to identify universal experts.

Runs warmup + upgrade cycles for each prompt, records per-layer expert activations,
and identifies "universal" experts activated in >threshold fraction of prompts.

Usage:
    PATH_REMOVED profile_experts.py [--model MODEL] [--capacity N] [--threshold F] [--output PATH]

    MODEL shortcuts: qwen, mixtral, glm (or any HuggingFace model name)

Examples:
    .../python profile_experts.py --model mixtral --output mixtral_experts.json
    .../python profile_experts.py --model glm --capacity 48 --output glm_experts.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, reset_to_cached,
    _find_switch_mlp, _detect_num_experts,
    CachedQuantizedSwitchLinear, PredictiveCachedSwitchLinear,
)

MODEL_PRESETS = {
    "qwen": ("mlx-community/Qwen3-Coder-Next-4bit", 208),
    "mixtral": ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 6),
    "mixtral-8x22b": ("mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit", 2),
    "glm": ("mlx-community/GLM-4.7-Flash-4bit", 48),
}

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


def apply_chat_template(tokenizer, text):
    """Wrap text in chat template if the tokenizer supports it."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
    return text


def collect_expert_activations(model, tokenizer, model_path, prompt, capacity,
                               use_chat_template=False):
    """Run warmup + upgrade for one prompt, return per-layer expert sets."""
    reset_to_cached(model, model_path, capacity)

    formatted = apply_chat_template(tokenizer, prompt) if use_chat_template else prompt
    mlx_lm.generate(model, tokenizer, prompt=formatted,
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
    parser = argparse.ArgumentParser(description="Profile expert routing for universal expert identification")
    parser.add_argument("--model", "-m", default="qwen",
                        help="Model preset (qwen/mixtral/glm) or HuggingFace name")
    parser.add_argument("--capacity", "-c", type=int, default=None,
                        help="Expert capacity per layer (default: auto per model)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Fraction of prompts for 'universal' classification (default: 0.5)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <model>_experts.json)")
    args = parser.parse_args()

    if args.model in MODEL_PRESETS:
        model_name, default_capacity = MODEL_PRESETS[args.model]
        short_name = args.model
    else:
        model_name = args.model
        default_capacity = 208
        short_name = model_name.split("/")[-1].lower()

    capacity = args.capacity if args.capacity is not None else default_capacity
    output_path = args.output or f"{short_name}_experts.json"
    use_chat_template = args.model in ("mixtral", "glm") or "instruct" in model_name.lower()

    model_path = hf_repo_to_path(model_name)
    print(f"Model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Capacity: {capacity}, Threshold: {args.threshold}")
    print(f"Chat template: {use_chat_template}")
    print(f"Prompts: {len(PROMPTS)}")

    print("Loading model with lazy=True...")
    model, tokenizer = mlx_lm.load(model_name, lazy=True)

    # Report model structure
    moe_layers = 0
    num_experts = 0
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            moe_layers += 1
            num_experts = _detect_num_experts(switch)
    print(f"MoE layers: {moe_layers}, Experts per layer: {num_experts}")

    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=capacity,
                                   predictive=True)
    print(f"Replaced {replaced} modules")

    mx.eval(model.parameters())
    print(f"Non-expert params loaded: {mx.get_active_memory() / 1e9:.1f} GB")

    # Bootstrap: first warmup + upgrade so reset_to_cached works in the loop
    print(f"\nBootstrap warmup...")
    bootstrap_prompt = apply_chat_template(tokenizer, PROMPTS[0]) if use_chat_template else PROMPTS[0]
    mlx_lm.generate(model, tokenizer, prompt=bootstrap_prompt,
                    max_tokens=WARMUP_TOKENS, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)
    print(f"Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    activation_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    num_prompts = len(PROMPTS)
    t_total = time.perf_counter()

    for p_idx, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        layer_experts = collect_expert_activations(
            model, tokenizer, model_path, prompt, capacity,
            use_chat_template=use_chat_template)
        elapsed = time.perf_counter() - t0

        for layer_idx, experts in layer_experts.items():
            for eid in experts:
                activation_counts[layer_idx][eid] += 1

        n_experts = sum(len(e) for e in layer_experts.values())
        print(f"  [{p_idx + 1}/{num_prompts}] {elapsed:.1f}s, "
              f"{len(layer_experts)} layers, {n_experts} total activations: "
              f"{prompt[:50]}...")

    print(f"\nTotal profiling time: {time.perf_counter() - t_total:.0f}s")

    result = {
        "model": model_name,
        "num_prompts": num_prompts,
        "threshold": args.threshold,
        "capacity": capacity,
        "num_experts": num_experts,
        "moe_layers": moe_layers,
        "layers": {},
    }

    total_universal = 0
    for layer_idx in sorted(activation_counts):
        counts = activation_counts[layer_idx]
        min_count = int(args.threshold * num_prompts)
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

    layers = result["layers"]
    universal_per_layer = [len(v["universal"]) for v in layers.values()]
    unique_per_layer = [v["total_unique"] for v in layers.values()]
    print(f"\n--- Summary ---")
    print(f"Model: {model_name}")
    print(f"Prompts profiled: {num_prompts}")
    print(f"Threshold: {args.threshold} ({int(args.threshold * num_prompts)}/{num_prompts} prompts)")
    print(f"MoE layers: {len(layers)}, Experts per layer: {num_experts}")
    print(f"Capacity: {capacity}/{num_experts} ({capacity/num_experts*100:.0f}%)")
    print(f"Universal experts per layer: "
          f"min={min(universal_per_layer)} median={sorted(universal_per_layer)[len(universal_per_layer)//2]} "
          f"max={max(universal_per_layer)} total={total_universal}")
    print(f"Unique experts per layer: "
          f"min={min(unique_per_layer)} median={sorted(unique_per_layer)[len(unique_per_layer)//2]} "
          f"max={max(unique_per_layer)}")


if __name__ == "__main__":
    main()
