"""Profile expert routing across diverse prompts to identify universal experts.

Runs warmup + upgrade cycles for each prompt, records per-layer expert activations,
and identifies "universal" experts activated in >threshold fraction of prompts.

Usage:
    uv run python benchmarks/profile_experts.py [--model MODEL] [--capacity N] [--threshold F] [--output PATH]

    MODEL shortcuts: qwen, mixtral, glm (or any HuggingFace model name)

Examples:
    .../python profile_experts.py --model mixtral --output mixtral_experts.json
    .../python profile_experts.py --model glm --capacity 48 --output glm_experts.json
"""

import argparse
import json
import random
import time
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.utils import hf_repo_to_path
from tool_chat_scenarios import get_profile_scenarios, render_scenario_prompt
from mlx_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, reset_to_cached,
    router_only_discovery,
    _find_switch_mlp, _detect_num_experts,
    CachedQuantizedSwitchLinear, PredictiveCachedSwitchLinear,
)

MODEL_PRESETS = {
    "qwen": ("mlx-community/Qwen3-Coder-Next-4bit", 208),
}

WARMUP_TOKENS = 10

PROMPTS_DIVERSE = [
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

PROMPTS_CODING = [
    # Python
    "Write a Python function that implements binary search on a sorted array",
    "Implement a Python class for a thread-safe LRU cache with TTL expiration",
    "Write a Python async HTTP client with retry logic and exponential backoff",
    "Debug this Python code: def fib(n): return fib(n-1) + fib(n-2)",
    "Refactor this function to use dataclasses instead of dicts for configuration",
    # JavaScript/TypeScript
    "Implement a React hook for debounced search with AbortController",
    "Write a TypeScript generic type that extracts all nested keys from an object",
    "Build a simple Express.js middleware for rate limiting with sliding window",
    # Rust/Go/C
    "Implement a concurrent hashmap in Rust using RwLock and sharding",
    "Write a Go HTTP server with graceful shutdown and context cancellation",
    "Write a recursive descent parser for arithmetic expressions in C",
    # Systems / Architecture
    "Design a URL shortener system with Redis caching and PostgreSQL storage",
    "Write a Dockerfile for a Python FastAPI app with multi-stage build",
    "Implement a simple key-value store with write-ahead logging",
    # Tool use / Agentic patterns (Claude Code scenario)
    'You have access to tools: [{"name": "read_file", "parameters": {"path": "string"}}]. Read the file at src/main.py and explain what it does.',
    'Based on the error "TypeError: Cannot read property map of undefined", fix the React component that fetches user data.',
    'Generate a JSON schema for a REST API endpoint that creates a new user with name, email, and role fields.',
    "Write a bash script that finds all Python files with syntax errors in a directory",
    # Code review / explanation
    "Explain what this code does: async def stream(): async for chunk in response.aiter_bytes(): yield chunk",
    "What are the performance implications of using SELECT * vs named columns in PostgreSQL?",
    "Compare the tradeoffs between WebSockets and Server-Sent Events for real-time updates",
    "Write unit tests for a function that parses ISO 8601 date strings with timezone handling",
]

PROMPT_PRESETS = {
    "diverse": PROMPTS_DIVERSE,
    "coding": PROMPTS_CODING,
    "tool-chat": None,
    "mixed": None,
}


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


def collect_expert_activations(
    model,
    tokenizer,
    model_path,
    prompt,
    capacity,
    discovery_mode: str,
):
    """Run warmup + upgrade for one prompt, return per-layer expert sets."""
    reset_to_cached(model, model_path, capacity)

    if discovery_mode == "router-only":
        router_only_discovery(model, tokenizer, prompt, max_tokens=WARMUP_TOKENS)
    else:
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


def _sample_items(pool: list, n: int, rng: random.Random) -> list:
    if n <= 0:
        return []
    if not pool:
        return []
    sampled = []
    while len(sampled) < n:
        batch = list(pool)
        rng.shuffle(batch)
        take = min(n - len(sampled), len(batch))
        sampled.extend(batch[:take])
    return [deepcopy(item) for item in sampled]


def build_mixed_prompt_items(
    coding_weight: int,
    toolchat_weight: int,
    num_prompts: int,
    seed: int,
) -> list:
    if coding_weight < 0 or toolchat_weight < 0:
        raise ValueError("coding_weight and toolchat_weight must be >= 0")
    if num_prompts <= 0:
        raise ValueError("num_prompts must be > 0")
    total_weight = coding_weight + toolchat_weight
    if total_weight <= 0:
        raise ValueError("coding_weight + toolchat_weight must be > 0")

    coding_n = int(round(num_prompts * coding_weight / total_weight))
    toolchat_n = num_prompts - coding_n
    rng = random.Random(seed)

    coding_items = _sample_items(PROMPTS_CODING, coding_n, rng)
    toolchat_items = _sample_items(get_profile_scenarios(), toolchat_n, rng)
    mixed = coding_items + toolchat_items
    rng.shuffle(mixed)
    return mixed


def resolve_prompt_items(
    preset: str,
    coding_weight: int = 70,
    toolchat_weight: int = 30,
    num_prompts: int = 24,
    seed: int = 0,
) -> list:
    if preset == "tool-chat":
        return get_profile_scenarios()
    if preset == "mixed":
        return build_mixed_prompt_items(
            coding_weight=coding_weight,
            toolchat_weight=toolchat_weight,
            num_prompts=num_prompts,
            seed=seed,
        )
    return PROMPT_PRESETS[preset]


def render_prompt_item(tokenizer, item, use_chat_template: bool) -> tuple[str, str]:
    if isinstance(item, dict):
        return item["name"], render_scenario_prompt(tokenizer, item)
    prompt = apply_chat_template(tokenizer, item) if use_chat_template else item
    return item, prompt


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
    parser.add_argument("--prompts", "-p", choices=list(PROMPT_PRESETS.keys()),
                        default="diverse", help="Prompt preset (default: diverse)")
    parser.add_argument("--coding-weight", type=int, default=70,
                        help="Coding prompt weight for --prompts mixed (default: 70)")
    parser.add_argument("--toolchat-weight", type=int, default=30,
                        help="Tool-chat prompt weight for --prompts mixed (default: 30)")
    parser.add_argument("--num-prompts", type=int, default=24,
                        help="Total sampled prompts for --prompts mixed (default: 24)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed for --prompts mixed (default: 0)")
    parser.add_argument("--discovery", choices=["warmup", "router-only"], default=None,
                        help="Discovery method (default: warmup, auto router-only for tool-chat/mixed)")
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
    use_chat_template = args.model in ("mixtral", "glm", "qwen2-moe", "qwen3-30b") or "instruct" in model_name.lower()
    prompt_items = resolve_prompt_items(
        args.prompts,
        coding_weight=args.coding_weight,
        toolchat_weight=args.toolchat_weight,
        num_prompts=args.num_prompts,
        seed=args.seed,
    )
    discovery_mode = args.discovery or (
        "router-only" if args.prompts in ("tool-chat", "mixed") else "warmup"
    )

    model_path = hf_repo_to_path(model_name)
    print(f"Model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Capacity: {capacity}, Threshold: {args.threshold}")
    print(f"Prompt preset: {args.prompts}")
    if args.prompts == "mixed":
        print(
            f"Mix: coding={args.coding_weight} tool-chat={args.toolchat_weight} "
            f"num_prompts={args.num_prompts} seed={args.seed}"
        )
    print(f"Discovery mode: {discovery_mode}")
    print(f"Chat template (text presets): {use_chat_template}")
    print(f"Prompts: {len(prompt_items)}")

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
    _, bootstrap_prompt = render_prompt_item(tokenizer, prompt_items[0], use_chat_template)
    if discovery_mode == "router-only":
        router_only_discovery(model, tokenizer, bootstrap_prompt, max_tokens=WARMUP_TOKENS)
    else:
        mlx_lm.generate(model, tokenizer, prompt=bootstrap_prompt,
                        max_tokens=WARMUP_TOKENS, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)
    print(f"Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    activation_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    num_prompts = len(prompt_items)
    t_total = time.perf_counter()

    for p_idx, item in enumerate(prompt_items):
        prompt_name, prompt = render_prompt_item(tokenizer, item, use_chat_template)
        t0 = time.perf_counter()
        layer_experts = collect_expert_activations(
            model, tokenizer, model_path, prompt, capacity, discovery_mode=discovery_mode)
        elapsed = time.perf_counter() - t0

        for layer_idx, experts in layer_experts.items():
            for eid in experts:
                activation_counts[layer_idx][eid] += 1

        n_experts = sum(len(e) for e in layer_experts.values())
        print(f"  [{p_idx + 1}/{num_prompts}] {elapsed:.1f}s, "
              f"{len(layer_experts)} layers, {n_experts} total activations: "
              f"{prompt_name[:60]}...")

    print(f"\nTotal profiling time: {time.perf_counter() - t_total:.0f}s")

    result = {
        "model": model_name,
        "num_prompts": num_prompts,
        "threshold": args.threshold,
        "capacity": capacity,
        "num_experts": num_experts,
        "moe_layers": moe_layers,
        "prompt_preset": args.prompts,
        "mix": {
            "coding_weight": args.coding_weight,
            "toolchat_weight": args.toolchat_weight,
            "num_prompts": args.num_prompts,
            "seed": args.seed,
        } if args.prompts == "mixed" else None,
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
