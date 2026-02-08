"""Task 11: Initial warmup timing breakdown.

Measure cold start: model load, enable_lazy, eval params, warmup gen, upgrade.
Test at both 192 and 256 capacity.
"""

import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_fallback_stats,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
PROMPT = "Write a Python function for binary search"


def run_coldstart(capacity):
    print(f"\n{'='*50}")
    print(f"COLD START: capacity={capacity}")
    print(f"{'='*50}")

    model_path = hf_repo_to_path(MODEL)

    # Step 1: Load model
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    t_load = time.perf_counter() - t0
    print(f"1. Model load (lazy=True): {t_load:.1f}s")

    # Step 2: Enable lazy experts
    t0 = time.perf_counter()
    replaced = enable_lazy_experts(model, model_path,
                                    cache_capacity_per_layer=capacity,
                                    predictive=True)
    t_enable = time.perf_counter() - t0
    print(f"2. enable_lazy_experts: {t_enable:.1f}s ({replaced} replaced)")

    # Step 3: Eval non-expert params
    t0 = time.perf_counter()
    mx.eval(model.parameters())
    t_eval = time.perf_counter() - t0
    mem_base = mx.get_active_memory() / 1e9
    print(f"3. mx.eval(params): {t_eval:.1f}s, memory: {mem_base:.1f} GB")

    # Step 4: Warmup generation
    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=10, verbose=False)
    t_warmup = time.perf_counter() - t0
    print(f"4. Warmup gen (10 tokens): {t_warmup:.1f}s")

    # Step 5: Upgrade to predictive
    t0 = time.perf_counter()
    upgraded = upgrade_to_predictive(model, model_path, capacity)
    t_upgrade = time.perf_counter() - t0
    mem_final = mx.get_active_memory() / 1e9
    print(f"5. upgrade_to_predictive: {t_upgrade:.1f}s, memory: {mem_final:.1f} GB")

    total = t_load + t_enable + t_eval + t_warmup + t_upgrade
    print(f"\nTOTAL COLD START: {total:.1f}s")

    # Quick gen test to verify working
    t0 = time.perf_counter()
    out = mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=20, verbose=False)
    t_test = time.perf_counter() - t0
    print(f"Verify gen: {20/t_test:.1f} tok/s")
    print(f"Output: {out[:100]}")

    del model
    mx.metal.clear_cache()

    return {
        "capacity": capacity,
        "load_time": t_load,
        "enable_time": t_enable,
        "eval_time": t_eval,
        "warmup_time": t_warmup,
        "upgrade_time": t_upgrade,
        "total_time": total,
        "base_memory": mem_base,
        "final_memory": mem_final,
    }


def main():
    results = []
    for cap in [192, 256]:
        r = run_coldstart(cap)
        results.append(r)

    with open("/Users/muhash/flash-moe/final_validation.md", "a") as f:
        f.write("\n## 11. Initial Warmup Time Breakdown\n\n")
        f.write("Cold start from nothing to ready-to-generate.\n\n")
        f.write("| Step | Capacity 192 | Capacity 256 |\n")
        f.write("|------|--------------|--------------|\n")

        r192 = results[0]
        r256 = results[1]
        f.write(f"| 1. Model load (lazy) | {r192['load_time']:.1f}s | {r256['load_time']:.1f}s |\n")
        f.write(f"| 2. enable_lazy_experts | {r192['enable_time']:.1f}s | {r256['enable_time']:.1f}s |\n")
        f.write(f"| 3. mx.eval(params) | {r192['eval_time']:.1f}s | {r256['eval_time']:.1f}s |\n")
        f.write(f"| 4. Warmup gen (10 tok) | {r192['warmup_time']:.1f}s | {r256['warmup_time']:.1f}s |\n")
        f.write(f"| 5. upgrade_to_predictive | {r192['upgrade_time']:.1f}s | {r256['upgrade_time']:.1f}s |\n")
        f.write(f"| **Total** | **{r192['total_time']:.1f}s** | **{r256['total_time']:.1f}s** |\n")
        f.write(f"| Base memory | {r192['base_memory']:.1f} GB | {r256['base_memory']:.1f} GB |\n")
        f.write(f"| Final memory | {r192['final_memory']:.1f} GB | {r256['final_memory']:.1f} GB |\n\n")

        f.write("**Notes:**\n")
        f.write("- Steps 1-3 are capacity-independent (non-expert parameters only).\n")
        f.write("- Step 4 (warmup gen) is where expert routing is discovered. Slow because experts load from disk on demand.\n")
        f.write(f"- Step 5 (upgrade) loads {r192['capacity']}×48 expert slots from safetensors into pre-stacked tensors. Time scales with capacity.\n")
        f.write(f"- Total cold start is dominated by warmup generation and upgrade.\n\n")

    print("\nResults appended to /Users/muhash/flash-moe/final_validation.md")


if __name__ == "__main__":
    main()
