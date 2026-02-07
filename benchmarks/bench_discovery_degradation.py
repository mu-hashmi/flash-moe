"""Task 5: Discovery degradation across reset cycles.

Run 5 cycles of: warmup -> upgrade -> delta_warmup -> reset_to_cached -> repeat.
Check if discovery time or quality degrades over cycles.
"""

import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, fast_delta_warmup,
    get_fallback_stats, reset_to_cached,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 192
PROMPT_EN = "Write a Python function for binary search"
PROMPT_CN = "用中文写一首关于春天的诗"
CYCLES = 5


def main():
    model_path = hf_repo_to_path(MODEL)
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
    mx.eval(model.parameters())
    print(f"Base memory: {mx.get_active_memory() / 1e9:.1f} GB")

    results = []

    for cycle in range(1, CYCLES + 1):
        print(f"\n{'='*50}")
        print(f"CYCLE {cycle}")
        print(f"{'='*50}")

        if cycle > 1:
            # Reset to cached state
            print("Resetting to cached state...")
            t0 = time.perf_counter()
            reset_to_cached(model, model_path, CAPACITY)
            t_reset = time.perf_counter() - t0
            mem_reset = mx.get_active_memory() / 1e9
            print(f"Reset: {t_reset:.1f}s, memory: {mem_reset:.1f} GB")
        else:
            t_reset = 0
            mem_reset = mx.get_active_memory() / 1e9

        # Warmup
        t0 = time.perf_counter()
        mlx_lm.generate(model, tokenizer, prompt=PROMPT_EN, max_tokens=10, verbose=False)
        t_warmup = time.perf_counter() - t0

        # Upgrade
        t0 = time.perf_counter()
        upgrade_to_predictive(model, model_path, CAPACITY)
        t_upgrade = time.perf_counter() - t0
        mem_upgrade = mx.get_active_memory() / 1e9
        print(f"Warmup: {t_warmup:.1f}s, Upgrade: {t_upgrade:.1f}s, Mem: {mem_upgrade:.1f} GB")

        # Delta warmup
        stats = fast_delta_warmup(model, tokenizer, model_path, PROMPT_CN)
        mem_delta = mx.get_active_memory() / 1e9
        print(f"Delta: discovery={stats['discovery_time']:.1f}s, "
              f"eval={stats['lookup_rebuild_time']:.1f}s, "
              f"total={stats['total_time']:.1f}s, "
              f"swaps={stats['total_swaps']}")

        # Generate
        t0 = time.perf_counter()
        out = mlx_lm.generate(model, tokenizer, prompt=PROMPT_CN, max_tokens=50, verbose=False)
        t_gen = time.perf_counter() - t0
        fb = get_fallback_stats(model)
        speed = 50 / t_gen
        preview = out[:100].replace('\n', ' ')
        print(f"Gen: {speed:.1f} tok/s, Fallback: {fb['fallback_rate']:.1%}")
        print(f"Output: {preview}")

        results.append({
            "cycle": cycle,
            "reset_time": t_reset,
            "warmup_time": t_warmup,
            "upgrade_time": t_upgrade,
            "mem_upgrade": mem_upgrade,
            "discovery_time": stats["discovery_time"],
            "eval_time": stats["lookup_rebuild_time"],
            "delta_total": stats["total_time"],
            "swaps": stats["total_swaps"],
            "gen_speed": speed,
            "fallback": fb["fallback_rate"],
            "mem_delta": mem_delta,
            "output": preview,
        })

    # Summary
    print(f"\n{'='*60}")
    print("DEGRADATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Cycle':>5} | {'Reset':>5} | {'Warmup':>6} | {'Upgrade':>7} | {'Discovery':>9} | {'Eval':>5} | {'Total':>6} | {'Swaps':>5} | {'tok/s':>5} | {'Mem':>5}")
    print("-" * 80)
    for r in results:
        print(f"{r['cycle']:>5} | {r['reset_time']:>4.1f}s | {r['warmup_time']:>5.1f}s | {r['upgrade_time']:>6.1f}s | {r['discovery_time']:>8.1f}s | {r['eval_time']:>4.1f}s | {r['delta_total']:>5.1f}s | {r['swaps']:>5} | {r['gen_speed']:>5.1f} | {r['mem_delta']:>4.1f}G")

    # Write to final_validation.md
    with open("/Users/muhash/flash-moe/final_validation.md", "a") as f:
        f.write("\n## 5. Discovery Degradation Across Reset Cycles\n\n")
        f.write(f"5 cycles of: reset_to_cached → warmup on English → upgrade_to_predictive → fast_delta_warmup (English→Chinese) → generate 50 tokens. Capacity {CAPACITY}.\n\n")
        f.write("| Cycle | Reset (s) | Warmup (s) | Upgrade (s) | Discovery (s) | Eval (s) | Delta Total (s) | Swaps | tok/s | Memory (GB) |\n")
        f.write("|-------|-----------|-----------|------------|--------------|---------|----------------|-------|-------|-------------|\n")
        for r in results:
            f.write(f"| {r['cycle']} | {r['reset_time']:.1f} | {r['warmup_time']:.1f} | {r['upgrade_time']:.1f} | {r['discovery_time']:.1f} | {r['eval_time']:.1f} | {r['delta_total']:.1f} | {r['swaps']} | {r['gen_speed']:.1f} | {r['mem_delta']:.1f} |\n")

        # Analysis
        disc_times = [r["discovery_time"] for r in results]
        total_times = [r["delta_total"] for r in results]
        gen_speeds = [r["gen_speed"] for r in results]
        degradation = max(disc_times) / min(disc_times) if min(disc_times) > 0 else 0
        f.write(f"\n**Discovery time range:** {min(disc_times):.1f}s – {max(disc_times):.1f}s ({degradation:.2f}x spread)\n")
        f.write(f"**Gen speed range:** {min(gen_speeds):.1f} – {max(gen_speeds):.1f} tok/s\n")
        if degradation > 1.3:
            f.write(f"**Degradation detected:** Discovery slows by {degradation:.1f}x over {CYCLES} cycles.\n\n")
        else:
            f.write(f"**No significant degradation:** Discovery time stable across cycles.\n\n")

    print("\nResults appended to /Users/muhash/flash-moe/final_validation.md")


if __name__ == "__main__":
    main()
