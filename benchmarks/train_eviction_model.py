"""Train tiny FFN eviction models using Belady oracle from routing traces.

Collects routing traces from 20+ prompts x 512 tokens, computes Belady optimal
distances, and trains a 3-layer FFN per MoE layer to predict eviction scores.

Usage:
    PATH_REMOVED train_eviction_model.py [capacity] [max_tokens] [output_dir]
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx_lm.utils import hf_repo_to_path
from mlx_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, reset_to_cached,
    PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear,
)

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
WARMUP_TOKENS = 10

PROMPTS = [
    "Explain general relativity in simple terms",
    "What caused World War I?",
    "Write an A* pathfinding implementation in Python",
    "Build a real-time chat application using React",
    "Implement a concurrent hashmap in Rust",
    "Prove there are infinitely many prime numbers",
    "用中文解释量子计算的基本原理",
    "写一首关于春天的中文诗",
    "人工知能の未来について日本語でエッセイを書いてください",
    "A farmer needs to cross a river with a wolf, goat, and cabbage",
    "Write a sorting algorithm comparison with Big-O analysis",
    "Explain the difference between TCP and UDP",
    "Write a Python decorator for memoization",
    "Implement binary search tree in JavaScript",
    "Explain machine learning to a 10 year old",
    "Write a REST API with FastAPI and SQLAlchemy",
    "什么是区块链技术？用简单的语言解释",
    "Write a recursive descent parser in C",
    "Explain the CAP theorem with examples",
    "Design a URL shortener system",
    "Write a matrix multiplication in NumPy",
    "Explain how transformers work in NLP",
]


class EvictionFFN(nn.Module):
    """Tiny 3-layer FFN for eviction scoring."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def __call__(self, x):
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        return self.l3(x)


def collect_routing_trace(model, tokenizer, prompt, max_tokens, capacity):
    """Generate tokens and record per-layer expert selections at each token.

    Returns dict[layer_idx] -> list of sets, one per token.
    """
    # Reset to cached for fresh warmup
    model_path = hf_repo_to_path(MODEL)
    reset_to_cached(model, model_path, capacity)
    mlx_lm.generate(model, tokenizer, prompt=prompt,
                    max_tokens=WARMUP_TOKENS, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)

    # Clear buffers
    for layer in model.layers:
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache._indices_buffer.clear()

    # Generate and collect routing decisions per token
    layer_traces: dict[int, list[set[int]]] = defaultdict(list)
    token_count = 0

    for response in mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                            max_tokens=max_tokens):
        token_count += 1

        for i, layer in enumerate(model.layers):
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
                continue
            proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
            if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                     SyncPredictiveCachedSwitchLinear)):
                continue

            cache = proj._cache
            token_experts = set()
            for indices in cache._indices_buffer:
                flat = np.asarray(indices.reshape(-1))
                token_experts.update(int(x) for x in np.unique(flat))
            cache._indices_buffer.clear()
            layer_traces[i].append(token_experts)

    return dict(layer_traces)


def compute_belady_oracle(trace, cached_ids):
    """Compute Belady optimal distances for each cached expert at each timestep.

    For each timestep t and each cached expert e, find the next time e is
    requested after t. If never requested again, distance = len(trace).

    Args:
        trace: list of sets (expert IDs requested at each token)
        cached_ids: set of expert IDs in the cache

    Returns list of dicts: [{expert_id: distance_to_next_use}, ...] per timestep.
    """
    T = len(trace)
    # Precompute: for each expert, sorted list of timesteps where it appears
    appearances: dict[int, list[int]] = defaultdict(list)
    for t, experts in enumerate(trace):
        for eid in experts:
            if eid in cached_ids:
                appearances[eid].append(t)

    oracle = []
    for t in range(T):
        distances = {}
        for eid in cached_ids:
            apps = appearances[eid]
            # Binary search for first appearance after t
            lo, hi = 0, len(apps)
            while lo < hi:
                mid = (lo + hi) // 2
                if apps[mid] <= t:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(apps):
                distances[eid] = apps[lo] - t
            else:
                distances[eid] = T - t  # never used again
        oracle.append(distances)

    return oracle


def build_training_data(trace, oracle, cached_ids, frequency, last_active, step_offset):
    """Build (features, targets) from oracle for one layer.

    Features: [1/recency, freq/max_freq]
    Targets: Belady distance (normalized to 0-1 range)
    """
    features = []
    targets = []
    T = len(trace)
    freq = dict(frequency)
    last_act = dict(last_active)
    step = step_offset

    for t in range(T):
        step += 1
        # Update frequency/recency for requested experts
        for eid in trace[t]:
            freq[eid] = freq.get(eid, 0) + 1
            last_act[eid] = step

        max_freq = max(freq.values()) if freq else 1
        distances = oracle[t]

        for eid in cached_ids:
            if eid not in distances:
                continue
            recency = step - last_act.get(eid, 0)
            f = freq.get(eid, 0)
            features.append([1.0 / max(recency, 1), f / max(max_freq, 1)])
            targets.append(distances[eid] / max(T, 1))

    return features, targets


def train_layer_model(features, targets, epochs=50, lr=1e-3, batch_size=256):
    """Train a tiny FFN on (features, targets) pairs."""
    model = EvictionFFN()
    optimizer = optim.Adam(learning_rate=lr)

    X = mx.array(features)
    Y = mx.array(targets).reshape(-1, 1)
    n = X.shape[0]

    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        # Shuffle
        perm = mx.array(np.random.permutation(n))
        X_shuf = X[perm]
        Y_shuf = Y[perm]

        total_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_batch = X_shuf[start:end]
            y_batch = Y_shuf[start:end]

            loss, grads = loss_and_grad(model, x_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += float(loss.item())
            batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(batches, 1)
            print(f"    Epoch {epoch + 1}/{epochs}: MSE = {avg_loss:.6f}")

    return model


def main():
    capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 208
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "eviction_models"

    model_path = hf_repo_to_path(MODEL)
    print(f"Model path: {model_path}")
    print(f"Capacity: {capacity}, Max tokens: {max_tokens}")

    print("Loading model...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)
    replaced = enable_lazy_experts(model, model_path,
                                   cache_capacity_per_layer=capacity,
                                   predictive=True)
    mx.eval(model.parameters())
    print(f"Replaced {replaced} modules, {mx.get_active_memory() / 1e9:.1f} GB")

    # Initial warmup + upgrade
    mlx_lm.generate(model, tokenizer, prompt=PROMPTS[0],
                    max_tokens=WARMUP_TOKENS, verbose=False)
    upgrade_to_predictive(model, model_path, capacity)

    # Collect routing traces from all prompts
    all_traces: dict[int, list[list[set[int]]]] = defaultdict(list)

    for p_idx, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        trace = collect_routing_trace(model, tokenizer, prompt, max_tokens, capacity)
        elapsed = time.perf_counter() - t0

        for layer_idx, token_sets in trace.items():
            all_traces[layer_idx].append(token_sets)

        total_tokens = sum(len(ts) for ts in trace.values())
        print(f"  [{p_idx + 1}/{len(PROMPTS)}] {elapsed:.1f}s, "
              f"{len(trace)} layers, ~{total_tokens // len(trace)} tokens/layer: "
              f"{prompt[:50]}...")

    # Get cached_ids from current predictive caches
    layer_cached_ids: dict[int, set[int]] = {}
    layer_freq: dict[int, dict] = {}
    layer_last_active: dict[int, dict] = {}
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                 SyncPredictiveCachedSwitchLinear)):
            continue
        cache = proj._cache
        layer_cached_ids[i] = set(cache.cached_ids)
        layer_freq[i] = dict(cache.frequency)
        layer_last_active[i] = dict(cache.last_active)

    # Train per-layer models
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trained_layers = []
    for layer_idx in sorted(all_traces.keys()):
        traces = all_traces[layer_idx]
        cached_ids = layer_cached_ids.get(layer_idx, set())
        if not cached_ids:
            continue

        print(f"\nLayer {layer_idx}: {len(traces)} traces, {len(cached_ids)} cached experts")

        # Build training data from all traces
        all_features = []
        all_targets = []
        for trace in traces:
            oracle = compute_belady_oracle(trace, cached_ids)
            feats, tgts = build_training_data(
                trace, oracle, cached_ids,
                layer_freq.get(layer_idx, {}),
                layer_last_active.get(layer_idx, {}),
                step_offset=0,
            )
            all_features.extend(feats)
            all_targets.extend(tgts)

        if len(all_features) < 100:
            print(f"  Skipping: only {len(all_features)} samples")
            continue

        print(f"  Training on {len(all_features)} samples...")
        ffn = train_layer_model(all_features, all_targets)

        # Save model weights
        layer_path = output_path / f"layer_{layer_idx}.safetensors"
        mx.save(str(layer_path), dict(ffn.parameters()))
        trained_layers.append(layer_idx)
        print(f"  Saved to {layer_path}")

    # Save metadata
    meta = {
        "capacity": capacity,
        "max_tokens": max_tokens,
        "num_prompts": len(PROMPTS),
        "trained_layers": trained_layers,
        "model_arch": "EvictionFFN(2->128->128->1)",
    }
    meta_path = output_path / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nTrained {len(trained_layers)} layer models, saved to {output_path}/")


if __name__ == "__main__":
    main()
