"""Test a single MoE model with mlx-moe.

Usage:
    uv run python benchmarks/test_model.py MODEL_NAME [--capacity N] [--baseline] [--tokens N]

Examples:
    uv run python benchmarks/test_model.py mlx-community/Qwen3-30B-A3B-4bit --baseline
    uv run python benchmarks/test_model.py mlx-community/Qwen3-30B-A3B-4bit --capacity 37
    uv run python benchmarks/test_model.py mlx-community/Qwen2-57B-A14B-Instruct-4bit
"""

import argparse
import gc
import json
import time

import mlx.core as mx

TEST_PROMPTS = [
    "Write a Python function that implements binary search on a sorted array.",
    "Explain the difference between TCP and UDP in simple terms.",
    "What causes the seasons on Earth?",
]


def run_baseline(model_name, max_tokens=200):
    """Run stock mlx-lm generation (no mlx-moe) as baseline."""
    import mlx_lm

    print(f"\n{'='*60}")
    print(f"BASELINE: {model_name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_name)
    mx.eval(model.parameters())
    load_time = time.perf_counter() - t0
    mem_gb = mx.get_active_memory() / 1e9
    print(f"Load: {load_time:.1f}s, Memory: {mem_gb:.1f} GB")

    prompt = TEST_PROMPTS[0]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass

    print(f"\nGenerating {max_tokens} tokens...")
    t0 = time.perf_counter()
    text = mlx_lm.generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=True)
    gen_time = time.perf_counter() - t0
    mem_after = mx.get_active_memory() / 1e9

    print(f"\nBaseline results:")
    print(f"  Memory: {mem_gb:.1f} -> {mem_after:.1f} GB")
    print(f"  Time: {gen_time:.1f}s")
    print(f"  Output preview: {text[:200]}...")

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return {"memory_gb": mem_gb, "memory_after_gb": mem_after,
            "load_time": load_time, "gen_time": gen_time}


def run_mlx_moe(model_name, capacity=None, max_tokens=200):
    """Run mlx-moe generation with diagnostics."""
    from mlx_moe.lazy_experts.generate import _startup
    from mlx_moe.lazy_experts.core import get_fallback_stats, measure_fallback
    from mlx_moe.lazy_experts.loading import _find_switch_mlp, _detect_num_experts
    import mlx_lm

    print(f"\n{'='*60}")
    print(f"MLX-MOE: {model_name} (capacity={capacity or 'auto'})")
    print(f"{'='*60}")

    prompt_text = TEST_PROMPTS[0]

    t0 = time.perf_counter()
    model, tokenizer, model_path = _startup(
        model_name, prompt_text, capacity=capacity)
    startup_time = time.perf_counter() - t0
    mem_gb = mx.get_active_memory() / 1e9

    # Report architecture
    moe_layers = 0
    num_experts = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is not None:
            moe_layers += 1
            if num_experts == 0:
                num_experts = _detect_num_experts(switch)
    print(f"\nArchitecture: {moe_layers} MoE layers, {num_experts} experts/layer")
    print(f"Startup: {startup_time:.1f}s, Memory: {mem_gb:.1f} GB")

    # Generate on each test prompt
    results = []
    for i, prompt_text in enumerate(TEST_PROMPTS):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True, tokenize=False)
            except Exception:
                prompt = prompt_text
        else:
            prompt = prompt_text

        print(f"\n--- Prompt {i+1}: {prompt_text[:50]}... ---")

        t0 = time.perf_counter()
        tokens = 0
        tps = 0.0
        text = ""
        for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                             max_tokens=max_tokens):
            text += resp.text
            tokens = resp.generation_tokens
            tps = resp.generation_tps

        gen_time = time.perf_counter() - t0
        mem_after = mx.get_active_memory() / 1e9

        fb = measure_fallback(model)

        print(f"  {tokens} tokens, {tps:.1f} tok/s, {gen_time:.1f}s")
        print(f"  Memory: {mem_after:.1f} GB")
        print(f"  Fallback: {fb['fallback_rate']*100:.1f}% ({fb['total_fallbacks']}/{fb['total_requests']})")
        print(f"  Output: {text[:150]}...")

        results.append({
            "prompt": prompt_text[:50],
            "tokens": tokens,
            "tps": tps,
            "gen_time": gen_time,
            "memory_gb": mem_after,
            "fallback_rate": fb["fallback_rate"],
        })

    # Cleanup
    st_map = getattr(model, "_st_map", None)
    if st_map is not None:
        st_map.close()
    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return {
        "startup_time": startup_time,
        "memory_gb": mem_gb,
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "prompts": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test a MoE model with mlx-moe")
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument("--capacity", "-c", type=int, default=None,
                        help="Expert capacity per layer (default: auto)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run stock mlx-lm baseline")
    parser.add_argument("--tokens", "-t", type=int, default=200,
                        help="Max tokens to generate (default: 200)")
    args = parser.parse_args()

    all_results = {"model": args.model, "capacity": args.capacity}

    if args.baseline:
        baseline = run_baseline(args.model, max_tokens=args.tokens)
        all_results["baseline"] = baseline
        # Need a fresh process for mlx-moe after baseline since Metal buffers leak
        print("\n[NOTE: For accurate mlx-moe results after baseline, run in separate process]")

    flash = run_mlx_moe(args.model, capacity=args.capacity, max_tokens=args.tokens)
    all_results["mlx_moe"] = flash

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
