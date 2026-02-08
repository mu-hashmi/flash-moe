"""Benchmark delta_warmup with scatter-based rebuild (now the default).

Loads model, warms on prompt A, upgrades to predictive, then delta-warms
on prompt B. Reports timing breakdown: discovery vs rebuild.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, delta_warmup, measure_fallback,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 256
PROMPT_A = "Hello, world!"
PROMPT_B = "Write a Python function that implements binary search on a sorted list."
WARMUP_TOKENS = 10
GEN_TOKENS = 50

model_path = hf_repo_to_path(MODEL)
print(f"Model path: {model_path}")

# Load model
print("Loading model...")
t0 = time.perf_counter()
model, tokenizer = mlx_lm.load(str(model_path), lazy=True)
replaced = enable_lazy_experts(model, model_path, CAPACITY, predictive=False)
print(f"  Replaced {replaced} modules")
mx.eval(model.parameters())
print(f"  Base model loaded: {mx.get_active_memory() / 1e9:.2f} GB ({time.perf_counter() - t0:.1f}s)")

# LCP warmup on prompt A
print(f"\nWarming up on prompt A ({WARMUP_TOKENS} tokens)...")
t0 = time.perf_counter()
mlx_lm.generate(model, tokenizer, prompt=PROMPT_A, max_tokens=WARMUP_TOKENS, verbose=False)
print(f"  Warmup done: {time.perf_counter() - t0:.1f}s")

# Upgrade to predictive
print("Upgrading to predictive...")
t0 = time.perf_counter()
upgraded = upgrade_to_predictive(model, model_path, CAPACITY)
print(f"  Upgraded {upgraded} modules: {mx.get_active_memory() / 1e9:.2f} GB ({time.perf_counter() - t0:.1f}s)")

# Delta warmup on prompt B
print(f"\nDelta warmup on prompt B...")
stats = delta_warmup(model, tokenizer, model_path, PROMPT_B, discovery_tokens=WARMUP_TOKENS)
print(f"  Discovery: {stats['discovery_time']:.2f}s")
print(f"  Rebuild:   {stats['rebuild_time']:.2f}s")
print(f"  Total:     {stats['total_time']:.2f}s")
print(f"  Swaps:     {stats['total_swaps']} across {len([s for s in stats['per_layer'] if s['swapped'] > 0])} layers")
print(f"  Missing:   {stats['total_missing']}")
print(f"  Memory:    {mx.get_active_memory() / 1e9:.2f} GB")

# Generate to verify correctness
print(f"\nGenerating {GEN_TOKENS} tokens...")
t0 = time.perf_counter()
result = mlx_lm.generate(model, tokenizer, prompt=PROMPT_B, max_tokens=GEN_TOKENS, verbose=False)
t_gen = time.perf_counter() - t0

fb = measure_fallback(model)
print(f"  Time: {t_gen:.1f}s ({GEN_TOKENS / t_gen:.1f} tok/s)")
print(f"  Fallback: {fb['fallback_rate']:.1%}")
print(f"  Output: {result[:200]}...")
