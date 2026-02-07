"""Smoke-test flash-moe generalization on non-Qwen MoE models."""
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.utils import hf_repo_to_path

from mlx_lm.lazy_experts import (
    enable_lazy_experts,
    upgrade_to_predictive,
    get_fallback_stats,
    _find_switch_mlp,
    _find_moe_block,
    _detect_num_experts,
)


def test_model(model_name: str, capacity: int, max_tokens: int = 50):
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Capacity: {capacity}, Max tokens: {max_tokens}")
    print(f"{'='*70}\n")

    # Step 1: Load model with lazy=True
    print("[1/6] Loading model (lazy=True)...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_name, lazy=True)
    model_path = hf_repo_to_path(model_name)
    print(f"  Loaded in {time.time()-t0:.1f}s, path: {model_path}")

    # Inspect model structure
    print("\n[INFO] Model structure inspection:")
    moe_count = 0
    dense_count = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is not None:
            moe_count += 1
            if i < 3 or i == len(model.layers) - 1:
                ne = _detect_num_experts(switch)
                print(f"  Layer {i}: MoE, {ne} experts, key_base={key_base}")
        else:
            dense_count += 1
            if i < 3:
                print(f"  Layer {i}: Dense")

    print(f"  Total: {moe_count} MoE layers, {dense_count} dense layers")

    # Step 2: Enable lazy experts
    print(f"\n[2/6] enable_lazy_experts(capacity={capacity}, predictive=True)...")
    t0 = time.time()
    replaced = enable_lazy_experts(model, model_path,
                                    cache_capacity_per_layer=capacity,
                                    predictive=True)
    print(f"  Replaced {replaced} modules in {time.time()-t0:.1f}s")

    # Step 3: Eval non-expert parameters
    print("\n[3/6] mx.eval(model.parameters()) — loading non-expert weights...")
    t0 = time.time()
    mx.eval(model.parameters())
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    # Step 4: Warmup (use chat template if available)
    warmup_prompt = "Hello"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            warmup_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
    print(f"\n[4/6] Warmup: generating 10 tokens...")
    t0 = time.time()
    warmup_out = mlx_lm.generate(model, tokenizer, prompt=warmup_prompt,
                                  max_tokens=10, verbose=False)
    print(f"  Warmup output: {warmup_out!r}")
    print(f"  Warmup took {time.time()-t0:.1f}s")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    # Step 5: Upgrade to predictive
    print(f"\n[5/6] upgrade_to_predictive(capacity={capacity})...")
    t0 = time.time()
    mx.metal.set_cache_limit(0)
    upgraded = upgrade_to_predictive(model, model_path, capacity)
    mx.metal.set_cache_limit(mx.metal.device_info()["memory_size"] // 4)
    print(f"  Upgraded {upgraded} modules in {time.time()-t0:.1f}s")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    # Step 6: Generate (use chat template if available)
    raw_prompt = "Write a short poem about the ocean."
    prompt = raw_prompt
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
    print(f"\n[6/6] Generating {max_tokens} tokens...")
    print(f"  Prompt: {raw_prompt!r}")
    t0 = time.time()
    output = mlx_lm.generate(model, tokenizer, prompt=prompt,
                              max_tokens=max_tokens, verbose=False)
    elapsed = time.time() - t0
    print(f"  Output ({len(output)} chars):")
    print(f"  {output}")
    print(f"  Time: {elapsed:.1f}s ({max_tokens/elapsed:.1f} tok/s)")

    # Stats
    stats = get_fallback_stats(model)
    print(f"\n[STATS]")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total fallbacks: {stats['total_fallbacks']}")
    if stats['total_requests'] > 0:
        pct = stats['total_fallbacks'] / stats['total_requests'] * 100
        print(f"  Fallback rate: {pct:.1f}%")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    print(f"\n{'='*70}")
    print(f"PASS: {model_name}")
    print(f"{'='*70}\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: test_generalization.py <mixtral|glm> [capacity] [max_tokens]")
        sys.exit(1)

    target = sys.argv[1].lower()
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    if target == "mixtral":
        capacity = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        test_model("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
                   capacity, max_tokens)
    elif target == "glm":
        capacity = int(sys.argv[2]) if len(sys.argv) > 2 else 48
        test_model("mlx-community/GLM-4.7-Flash-4bit",
                   capacity, max_tokens)
    else:
        print(f"Unknown target: {target}. Use 'mixtral' or 'glm'.")
        sys.exit(1)
