"""Lazy expert loading for Qwen3-Coder-Next-4bit on memory-constrained Macs.

Loads only router-selected experts on demand from memory-mapped safetensors,
keeping all non-expert weights (attention, embeddings, router, shared experts)
permanently in Metal memory. This reduces peak memory from ~40GB to ~5GB.

Usage (run from the mlx-lm venv):
    /path/to/mlx-lm/.venv/bin/python generate_lazy.py ["prompt"] [max_tokens]
"""

import sys
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import enable_lazy_experts

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a hello world program in Python"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")

    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)

    replaced = enable_lazy_experts(model, model_path)
    print(f"Replaced {replaced} expert modules with lazy loaders")

    print("Evaluating non-expert parameters into Metal memory...")
    mx.eval(model.parameters())
    print(f"Non-expert params loaded. Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")

    print(f"\nGenerating (max_tokens={max_tokens})...")
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=True,
    )

    print(f"\nFinal Metal memory: {mx.get_active_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
