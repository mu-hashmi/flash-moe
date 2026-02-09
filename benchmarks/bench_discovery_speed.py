"""Diagnose delta_warmup discovery speed: prefill vs generation breakdown.

The key question: when delta_warmup() reports 54s "discovery", how much
is prompt prefill vs the 10 tokens of generation?

Approach: patch mlx_lm.generate's stream to timestamp prefill completion
vs generation start/end. This gives us exact numbers without reimplementing
the generation loop.
"""

import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, delta_warmup,
    PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
CAPACITY = 256
WARMUP_TOKENS = 10

model_path = hf_repo_to_path(MODEL)
print(f"Model path: {model_path}")

# Load and setup
print("Loading model...")
t0 = time.perf_counter()
model, tokenizer = mlx_lm.load(str(model_path), lazy=True)
replaced = enable_lazy_experts(model, model_path, CAPACITY, predictive=False)
mx.eval(model.parameters())
print(f"  Base: {mx.get_active_memory() / 1e9:.2f} GB ({time.perf_counter() - t0:.1f}s)")

# LCP warmup on a short prompt to minimize this phase
print(f"\nLCP warmup ({WARMUP_TOKENS} tokens on 'hi')...")
t0 = time.perf_counter()
mlx_lm.generate(model, tokenizer, prompt="hi", max_tokens=WARMUP_TOKENS, verbose=False)
print(f"  Done: {time.perf_counter() - t0:.1f}s")

# Upgrade
print("Upgrading to predictive...")
t0 = time.perf_counter()
upgraded = upgrade_to_predictive(model, model_path, CAPACITY)
print(f"  {upgraded} modules: {mx.get_active_memory() / 1e9:.2f} GB ({time.perf_counter() - t0:.1f}s)")

# Now we're in predictive mode. Test discovery speed with different prompts.
PROMPTS = {
    "tiny (1 tok)": "hi",
    "short (13 tok)": "Write a Python function that implements binary search on a sorted list.",
    "medium (45 tok)": (
        "Write a detailed Python implementation of a red-black tree data structure "
        "with insert, delete, and search operations. Include proper rotation functions "
        "and color-fixing after insertions and deletions. Add comprehensive docstrings "
        "and type hints throughout."
    ),
    "chinese": "用中文详细解释什么是动态规划算法，并用Python实现一个背包问题的解法。",
}

# Measure prompt token counts
print("\n=== Prompt token counts ===")
for label, prompt in PROMPTS.items():
    tokens = tokenizer.encode(prompt)
    print(f"  {label}: {len(tokens)} tokens")


def clear_caches():
    for layer in model.layers:
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache._indices_buffer.clear()
            proj._cache.total_requests = 0
            proj._cache.total_fallbacks = 0


def count_discovered():
    total_req, total_miss = 0, 0
    for layer in model.layers:
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue
        cache_obj = proj._cache
        requested = set()
        for indices in cache_obj._indices_buffer:
            flat = np.asarray(indices.reshape(-1))
            requested |= set(int(x) for x in np.unique(flat))
        missing = requested - cache_obj.cached_set
        total_req += len(requested)
        total_miss += len(missing)
        cache_obj._indices_buffer.clear()
    return total_req, total_miss


# Test 1: measure generate() speed with predictive cache (the discovery path)
print("\n=== generate() speed with predictive cache ===")
print("(This is exactly what delta_warmup's discovery step does)\n")

for label, prompt in PROMPTS.items():
    clear_caches()

    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=prompt,
                     max_tokens=WARMUP_TOKENS, verbose=False)
    t_total = time.perf_counter() - t0

    total_req, total_miss = count_discovered()
    n_prompt = len(tokenizer.encode(prompt))

    print(f"  {label} ({n_prompt} prompt tok): {t_total:.2f}s total, "
          f"{total_req} experts seen, {total_miss} missing")

# Test 2: measure just prefill (model forward, no generation)
# We need to call through the actual generate pipeline to get proper KV cache,
# so instead we'll use max_tokens=0 or max_tokens=1
print("\n=== Prefill-only (max_tokens=1) ===\n")

for label, prompt in PROMPTS.items():
    clear_caches()

    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=prompt,
                     max_tokens=1, verbose=False)
    t_total = time.perf_counter() - t0

    n_prompt = len(tokenizer.encode(prompt))
    print(f"  {label} ({n_prompt} prompt tok): {t_total:.3f}s")

# Test 3: actual delta_warmup timing
print("\n=== delta_warmup() full breakdown ===\n")

for label, prompt in PROMPTS.items():
    stats = delta_warmup(model, tokenizer, model_path, prompt,
                          discovery_tokens=WARMUP_TOKENS)

    print(f"  {label}:")
    print(f"    Discovery: {stats['discovery_time']:.2f}s")
    print(f"    Rebuild:   {stats['rebuild_time']:.2f}s")
    print(f"    Total:     {stats['total_time']:.2f}s")
    print(f"    Swaps: {stats['total_swaps']}, Missing: {stats['total_missing']}")
    print(f"    Memory: {mx.get_active_memory() / 1e9:.2f} GB")
    print()
