"""Task 4: Per-token speed profile during async-delta warmup.

Switch from A* Python to Chinese poetry, recording time per token.
Shows when swaps complete and how they affect generation speed.
"""

import time
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_fallback_stats,
    incremental_delta_warmup,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 192
SOURCE_PROMPT = "Write a Python implementation of A* pathfinding with visualization"
TARGET_PROMPT = "用中文写一首关于秋天的诗"


def main():
    model_path = hf_repo_to_path(MODEL)
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=CAPACITY, predictive=True)
    mx.eval(model.parameters())
    print(f"Base memory: {mx.get_active_memory() / 1e9:.1f} GB")

    # Warmup on source prompt
    mlx_lm.generate(model, tokenizer, prompt=SOURCE_PROMPT, max_tokens=10, verbose=False)
    upgrade_to_predictive(model, model_path, CAPACITY)
    print(f"After upgrade: {mx.get_active_memory() / 1e9:.1f} GB")

    # Async-delta warmup
    warmup, disc_stats = incremental_delta_warmup(
        model, tokenizer, model_path, TARGET_PROMPT, discovery_tokens=10,
    )
    print(f"Discovery: {disc_stats['discovery_time']:.1f}s, "
          f"Total swaps: {disc_stats['total_swaps']}, "
          f"Layers: {disc_stats['total_layers']}")

    # Stream generate with per-token timing
    token_times = []
    t_start = time.perf_counter()
    t_prev = t_start
    token_count = 0
    max_tokens = 80  # enough to see past swap completion

    for response in mlx_lm.stream_generate(model, tokenizer, prompt=TARGET_PROMPT,
                                             max_tokens=max_tokens):
        t_now = time.perf_counter()
        token_count += 1
        dt = (t_now - t_prev) * 1000  # ms
        cumulative = t_now - t_start

        swaps_done = warmup.is_complete
        swapped_count = 0
        if not warmup.is_complete:
            swapped_count = warmup.step(layers_per_step=2)
            if warmup.is_complete:
                swaps_done = True

        prog = warmup.progress
        token_times.append({
            "token": token_count,
            "dt_ms": dt,
            "cumulative_s": cumulative,
            "swaps_complete": swaps_done,
            "layers_done": prog["layers_done"],
            "layers_total": prog["layers_total"],
            "text": response.text,
        })
        t_prev = t_now

    t_total = time.perf_counter() - t_start

    # Print table
    print(f"\n{'Token':>5} | {'dt (ms)':>8} | {'Cumul (s)':>9} | {'Swaps':>10} | {'Text':>20}")
    print("-" * 65)

    swap_complete_token = None
    for entry in token_times:
        status = "DONE" if entry["swaps_complete"] else f"{entry['layers_done']}/{entry['layers_total']}"
        text_preview = repr(entry["text"])[:20]
        print(f"{entry['token']:>5} | {entry['dt_ms']:>8.1f} | {entry['cumulative_s']:>9.2f} | {status:>10} | {text_preview}")
        if entry["swaps_complete"] and swap_complete_token is None:
            swap_complete_token = entry["token"]

    # Summary
    fb = get_fallback_stats(model)
    print(f"\nTotal: {token_count} tokens in {t_total:.1f}s ({token_count/t_total:.1f} tok/s)")
    print(f"Swaps complete at token: {swap_complete_token}")
    print(f"Fallback: {fb['fallback_rate']:.1%}")
    print(f"Memory: {mx.get_active_memory() / 1e9:.1f} GB")

    # Write to final_validation.md
    with open("PATH_REMOVED", "a") as f:
        f.write("\n## 4. Per-Token Speed Profile (Async-Delta)\n\n")
        f.write(f"Switch: A* Python → Chinese poetry at capacity {CAPACITY}.\n")
        f.write(f"Discovery: {disc_stats['discovery_time']:.1f}s, {disc_stats['total_swaps']} swaps across {disc_stats['total_layers']} layers.\n\n")

        f.write("| Token | dt (ms) | Cumul (s) | Swaps Status | Text |\n")
        f.write("|-------|---------|-----------|-------------|------|\n")

        # Show first 40 tokens or until 10 after swaps complete
        cutoff = min(len(token_times), 60)
        if swap_complete_token:
            cutoff = min(cutoff, swap_complete_token + 10)
        for entry in token_times[:cutoff]:
            status = "DONE" if entry["swaps_complete"] else f"{entry['layers_done']}/{entry['layers_total']}"
            text_esc = entry["text"].replace("|", "\\|").replace("\n", " ")[:15]
            f.write(f"| {entry['token']} | {entry['dt_ms']:.0f} | {entry['cumulative_s']:.2f} | {status} | {text_esc} |\n")

        f.write(f"\n**Swaps complete at token {swap_complete_token}** ({token_times[swap_complete_token-1]['cumulative_s']:.1f}s elapsed).\n")
        f.write(f"Overall: {token_count} tokens in {t_total:.1f}s ({token_count/t_total:.1f} tok/s). Fallback: {fb['fallback_rate']:.1%}.\n\n")

        # Phase analysis
        if swap_complete_token and swap_complete_token < len(token_times):
            during_swap = token_times[:swap_complete_token]
            after_swap = token_times[swap_complete_token:]
            avg_during = sum(e["dt_ms"] for e in during_swap) / len(during_swap)
            avg_after = sum(e["dt_ms"] for e in after_swap) / len(after_swap) if after_swap else 0
            f.write(f"**During swaps (tokens 1-{swap_complete_token}):** {avg_during:.0f}ms/token avg ({1000/avg_during:.1f} tok/s)\n")
            if after_swap:
                f.write(f"**After swaps (tokens {swap_complete_token+1}-{token_count}):** {avg_after:.0f}ms/token avg ({1000/avg_after:.1f} tok/s)\n\n")

    print("\nResults appended to PATH_REMOVED")


if __name__ == "__main__":
    main()
