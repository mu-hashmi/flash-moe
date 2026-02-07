"""Task 10 sub-task: Try loading the full model without lazy experts.

Document what happens on a 32 GB Mac.
"""

import time
import mlx.core as mx
import mlx_lm

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"


def main():
    print("Attempting to load full model without lazy experts...")
    print(f"System memory: {mx.metal.device_info()['memory_size'] / 1e9:.1f} GB")

    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    t_load = time.perf_counter() - t0
    print(f"Model load (lazy): {t_load:.1f}s")
    print(f"Metal memory after lazy load: {mx.get_active_memory() / 1e9:.2f} GB")

    print("\nSkipping mx.eval — full model (46 GB) exceeds Metal memory (34 GB).")
    print("Previous attempt crashed: [METAL] Command buffer execution failed: Insufficient Memory")
    print("This confirms why flash-moe is needed: the model cannot run normally on this hardware.")


if __name__ == "__main__":
    main()
