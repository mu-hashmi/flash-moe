"""Streaming generation wrapper for flash-moe.

Startup is blocking; tokens stream after model is ready.

Usage:
    PATH_REMOVED generate_streaming.py [--model MODEL] [--prompt TEXT] [--tokens N]
"""

import argparse
import time

import mlx.core as mx
from mlx_lm.lazy_experts import flash_stream_generate

MODEL_PRESETS = {
    "qwen": "mlx-community/Qwen3-Coder-Next-4bit",
    "mixtral": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
    "glm": "mlx-community/GLM-4.7-Flash-4bit",
}


def main():
    parser = argparse.ArgumentParser(description="Streaming flash-moe generation")
    parser.add_argument("--model", "-m", default="qwen",
                        help="Model preset (qwen/mixtral/glm) or HuggingFace name")
    parser.add_argument("--prompt", "-p", default="Write a hello world program in Python")
    parser.add_argument("--tokens", "-t", type=int, default=200)
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory for persistent state")
    parser.add_argument("--profile", default=None,
                        help="Path to universal expert profile JSON")
    args = parser.parse_args()

    model_name = MODEL_PRESETS.get(args.model, args.model)

    print(f"Model: {model_name}")
    print(f"Prompt: {args.prompt[:60]}...")
    print(f"Max tokens: {args.tokens}")
    print(f"Startup...", flush=True)

    t_start = time.perf_counter()
    token_count = 0
    first_token_time = None

    for response in flash_stream_generate(
        model_name, args.prompt, max_tokens=args.tokens,
        cache_dir=args.cache_dir, profile_path=args.profile,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter() - t_start
            print(f"\nFirst token: {first_token_time:.1f}s\n")

        print(response.text, end="", flush=True)
        token_count += 1

    t_total = time.perf_counter() - t_start

    print(f"\n\n--- Stats ---")
    print(f"Time to first token: {first_token_time:.1f}s")
    print(f"Tokens: {token_count}")
    print(f"Total time: {t_total:.1f}s")
    if token_count > 0:
        gen_time = t_total - first_token_time
        print(f"Generation: {gen_time:.1f}s ({token_count / gen_time:.1f} tok/s)")
    print(f"Memory: {mx.get_active_memory() / 1e9:.1f} GB")
    print(f"Finish reason: {response.finish_reason}")


if __name__ == "__main__":
    main()
