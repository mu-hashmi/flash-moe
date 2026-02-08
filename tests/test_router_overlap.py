"""Test router-only discovery methods against ground-truth expert routing.

Compares three router-only discovery approaches:
  1. router_only_forward — skip MoE, run only shared expert + router
  2. speculative_router_probe — capture all layers' hidden states, probe union
  3. speculative_router_cross_layer — probe all routers with first layer's states

Ground truth: actual expert IDs selected during Phase 2 cached generation.
Metric: overlap % = |intersection(predicted, ground_truth)| / |ground_truth|

Usage:
    /Users/muhash/mlx-lm/.venv/bin/python test_router_overlap.py
"""

import sys
import time
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts,
    router_only_forward,
    speculative_router_probe,
    speculative_router_cross_layer,
    ExpertCache,
    CachedQuantizedSwitchLinear,
    _build_shard_map,
)
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
WARMUP_TOKENS = 10

PROMPTS = [
    "Write a Python function to implement binary search",
    "Explain quantum entanglement in simple terms",
    "用中文写一首关于春天的诗",
    "Debug this JavaScript: function add(a,b) { return a - b; }",
    "Solve the differential equation dy/dx = 2xy",
    "Write a Rust implementation of a linked list",
]


def install_fresh_cached(model, shard_map: dict, capacity: int) -> int:
    """Install fresh CachedQuantizedSwitchLinear modules regardless of current type."""
    replaced = 0
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
        layer_cache = ExpertCache(capacity)
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = CachedQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
                proj_name=name,
                cache=layer_cache,
            )
            setattr(switch, name, replacement)
            replaced += 1
    return replaced


def collect_ground_truth(model, tokenizer, prompt: str,
                         shard_map: dict, capacity: int) -> dict[int, set[int]]:
    """Run Phase 2 cached generation and harvest all_seen expert IDs per layer."""
    install_fresh_cached(model, shard_map, capacity)
    mlx_lm.generate(model, tokenizer, prompt=prompt,
                    max_tokens=WARMUP_TOKENS, verbose=False)

    ground_truth: dict[int, set[int]] = {}
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
        up = getattr(switch, "up_proj")
        if isinstance(up, CachedQuantizedSwitchLinear):
            ground_truth[i] = set(up._cache.all_seen)

    mx.clear_cache()
    return ground_truth


def compute_overlap(predicted: dict[int, set[int]],
                    ground_truth: dict[int, set[int]]) -> dict:
    """Compute per-layer and aggregate overlap statistics."""
    per_layer = []
    total_intersect = 0
    total_gt = 0
    total_pred = 0

    for layer_idx in sorted(ground_truth.keys()):
        gt = ground_truth[layer_idx]
        pred = predicted.get(layer_idx, set())
        intersect = gt & pred
        overlap_pct = len(intersect) / len(gt) * 100 if gt else 0.0
        per_layer.append({
            "layer": layer_idx,
            "gt_count": len(gt),
            "pred_count": len(pred),
            "intersect": len(intersect),
            "overlap_pct": overlap_pct,
        })
        total_intersect += len(intersect)
        total_gt += len(gt)
        total_pred += len(pred)

    agg_overlap = total_intersect / total_gt * 100 if total_gt else 0.0
    return {
        "per_layer": per_layer,
        "agg_overlap_pct": agg_overlap,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_intersect": total_intersect,
    }


def main():
    model_path = hf_repo_to_path(MODEL)
    shard_map = _build_shard_map(model_path)

    print(f"Loading {MODEL} with lazy=True...")
    model, tokenizer = mlx_lm.load(MODEL, lazy=True)

    # Install initial Phase 2 modules so mx.eval skips expert weights
    enable_lazy_experts(model, model_path, cache_capacity_per_layer=256)
    print("Evaluating non-expert parameters...")
    mx.eval(model.parameters())
    print(f"Base memory: {mx.get_active_memory() / 1e9:.2f} GB")

    methods = [
        ("router_only_forward", router_only_forward),
        ("speculative_router_probe", speculative_router_probe),
        ("speculative_router_cross_layer", speculative_router_cross_layer),
    ]

    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'='*70}")
        print(f"Prompt {pi+1}/{len(PROMPTS)}: {prompt[:60]}...")
        print(f"{'='*70}")

        # Collect ground truth
        print("  Collecting ground truth (Phase 2 cached generation)...")
        t0 = time.perf_counter()
        gt = collect_ground_truth(model, tokenizer, prompt, shard_map, capacity=256)
        t_gt = time.perf_counter() - t0
        gt_total = sum(len(v) for v in gt.values())
        print(f"  Ground truth: {gt_total} total experts across {len(gt)} layers ({t_gt:.1f}s)")

        prompt_results = {"prompt": prompt, "ground_truth_total": gt_total, "methods": {}}

        for method_name, method_fn in methods:
            # Reinstall fresh cached modules before each router-only method
            install_fresh_cached(model, shard_map, 256)

            print(f"  Running {method_name}...")
            t0 = time.perf_counter()
            predicted = method_fn(model, tokenizer, prompt, max_tokens=WARMUP_TOKENS)
            t_method = time.perf_counter() - t0

            overlap = compute_overlap(predicted, gt)
            pred_total = sum(len(v) for v in predicted.values())
            print(f"    -> {overlap['agg_overlap_pct']:.1f}% overlap "
                  f"({overlap['total_intersect']}/{overlap['total_gt']} experts) "
                  f"| predicted {pred_total} | {t_method:.1f}s")

            prompt_results["methods"][method_name] = {
                "overlap_pct": overlap["agg_overlap_pct"],
                "total_intersect": overlap["total_intersect"],
                "total_gt": overlap["total_gt"],
                "total_pred": overlap["total_pred"],
                "time_s": t_method,
                "per_layer": overlap["per_layer"],
            }

        all_results.append(prompt_results)
        mx.clear_cache()

    # Aggregate summary
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'Mean Overlap%':>13} {'Min':>6} {'Max':>6}")
    print("-" * 62)

    for method_name, _ in methods:
        overlaps = [r["methods"][method_name]["overlap_pct"] for r in all_results]
        mean_ov = np.mean(overlaps)
        min_ov = np.min(overlaps)
        max_ov = np.max(overlaps)
        print(f"{method_name:<35} {mean_ov:>12.1f}% {min_ov:>5.1f}% {max_ov:>5.1f}%")

    print()
    print("Per-prompt breakdown:")
    for r in all_results:
        short = r["prompt"][:50]
        parts = []
        for method_name, _ in methods:
            ov = r["methods"][method_name]["overlap_pct"]
            parts.append(f"{method_name.split('_')[-1]}={ov:.0f}%")
        print(f"  {short:<50} {' | '.join(parts)}")

    # Per-layer distribution for each method
    print()
    print("Per-layer overlap distribution (across all prompts):")
    for method_name, _ in methods:
        layer_overlaps: dict[int, list[float]] = {}
        for r in all_results:
            for linfo in r["methods"][method_name]["per_layer"]:
                layer_overlaps.setdefault(linfo["layer"], []).append(linfo["overlap_pct"])

        means = [np.mean(v) for v in layer_overlaps.values()]
        print(f"  {method_name}:")
        print(f"    Layer-mean overlap: min={np.min(means):.1f}% "
              f"median={np.median(means):.1f}% max={np.max(means):.1f}%")


if __name__ == "__main__":
    main()
