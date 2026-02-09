"""Benchmark expert splitting for Mixtral models.

Sweeps split_t values and measures speed, memory, and output quality
for top-only inference via SplitSwitchGLU.

Usage:
    uv run python benchmarks/mixtral/bench_split.py [--model MODEL] [--splits 1024,2048,3072]
    uv run python benchmarks/mixtral/bench_split.py --model 4bit --splits 2048,3072 --max-tokens 100

Presets:
    4bit  → mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit
    2bit  → ~/.cache/flash-moe/Mixtral-8x22B-Instruct-v0.1-2bit (local)
    8x7b  → mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
"""
import argparse
import json
import time

import mlx.core as mx

MODEL_PRESETS = {
    "4bit": "mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit",
    "2bit": "/Users/muhash/.cache/flash-moe/Mixtral-8x22B-Instruct-v0.1-2bit",
    "8x7b": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
}

PROMPT = "Write a Python function that computes the nth Fibonacci number using memoization."


def run_one(model_name, split_t, max_tokens):
    import mlx_lm
    from flash_moe.lazy_experts.core import enable_lazy_experts, enable_split_experts
    from flash_moe.lazy_experts.loading import (
        _find_switch_mlp, _detect_num_experts, SafetensorsMap,
    )
    from mlx_lm.utils import hf_repo_to_path
    from pathlib import Path

    model, tokenizer = mlx_lm.load(model_name, lazy=True)
    model_path = hf_repo_to_path(model_name) if "/" in model_name else model_name

    num_experts = 0
    intermediate_size = 0
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            num_experts = _detect_num_experts(switch)
            gate = getattr(switch, "gate_proj")
            if hasattr(gate, "weight"):
                intermediate_size = gate.weight.shape[1]
            break

    # Replace expert modules with lazy stubs (no array params) so mx.eval
    # only materializes the non-expert base weights (~3 GB).
    enable_lazy_experts(model, model_path)
    mx.eval(model.parameters())
    base_gb = mx.get_active_memory() / 1e9
    print(f"  Base loaded: {base_gb:.1f} GB", flush=True)

    all_shards = sorted(str(p) for p in Path(model_path).glob("*.safetensors"))
    st_map = SafetensorsMap(all_shards)

    print(f"  Calling enable_split_experts(split_t={split_t})...", flush=True)
    t0 = time.perf_counter()
    n_replaced = enable_split_experts(model, model_path, split_t, st_map=st_map)
    t_split = time.perf_counter() - t0

    st_map.close()
    loaded_gb = mx.get_active_memory() / 1e9

    if hasattr(mx, "set_wired_limit"):
        active = mx.get_active_memory()
        limit = int(mx.device_info()["memory_size"] * 0.75)
        mx.set_wired_limit(min(active, limit))

    t0 = time.perf_counter()
    result = mlx_lm.generate(
        model, tokenizer, prompt=PROMPT, max_tokens=max_tokens, verbose=False)
    t_gen = time.perf_counter() - t0

    split_pct = split_t / intermediate_size * 100 if intermediate_size > 0 else 0

    return {
        "split_t": split_t,
        "split_pct": round(split_pct, 1),
        "num_experts": num_experts,
        "intermediate_size": intermediate_size,
        "base_gb": round(base_gb, 2),
        "loaded_gb": round(loaded_gb, 2),
        "split_time_s": round(t_split, 1),
        "gen_s": round(t_gen, 1),
        "tok_s": round(max_tokens / t_gen, 1),
        "modules_replaced": n_replaced,
        "output": result,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="4bit", help="Model preset or HF path")
    parser.add_argument("--splits", default="1024,2048,3072",
                        help="Comma-separated split_t values to sweep")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    args = parser.parse_args()

    model_name = MODEL_PRESETS.get(args.model, args.model)
    splits = [int(s) for s in args.splits.split(",")]

    print(f"Model: {model_name}")
    print(f"Split values: {splits}")
    print(f"Prompt: {PROMPT[:60]}...")
    print(f"Max tokens: {args.max_tokens}")
    print()

    results = []
    for split_t in splits:
        print(f"--- split_t={split_t} ---")
        try:
            r = run_one(model_name, split_t, args.max_tokens)
            results.append(r)
            print(f"  Memory: {r['loaded_gb']} GB (base: {r['base_gb']} GB)")
            print(f"  Split: {r['split_pct']}% of intermediate ({r['split_t']}/{r['intermediate_size']})")
            print(f"  Speed: {r['tok_s']} tok/s ({r['gen_s']}s)")
            print(f"  Modules replaced: {r['modules_replaced']}")
            print(f"  Output: {r['output'][:120]}...")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
            results.append({"split_t": split_t, "error": str(e)})
        print()

    print("=== Summary ===")
    print(f"{'split_t':>8} {'%':>5} {'mem_gb':>7} {'tok/s':>6} {'split_s':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['split_t']:>8} {'ERROR':>5}")
        else:
            print(f"{r['split_t']:>8} {r['split_pct']:>5.1f} {r['loaded_gb']:>7.1f} "
                  f"{r['tok_s']:>6.1f} {r['split_time_s']:>7.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
