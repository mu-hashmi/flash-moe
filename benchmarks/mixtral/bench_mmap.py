"""Benchmark mmap vs mx.load for expert loading from safetensors.

Usage:
    uv run python benchmarks/mixtral/bench_mmap.py [--iters N]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from flash_moe.lazy_experts.loading import (
    SafetensorsMap,
    _load_proj_experts,
    _mmap_load_proj_experts,
)

MODEL_DIR = Path(
    "~/.cache/huggingface/hub/"
    "models--mlx-community--Mixtral-8x22B-Instruct-v0.1-4bit"
).expanduser()
SNAPSHOT = next(
    (MODEL_DIR / "snapshots").iterdir()
)

LAYER = 0
KEY_BASE = f"model.layers.{LAYER}.block_sparse_moe.switch_mlp"
EXPERT_IDS = mx.array([0, 1])


def find_shard_for_key(key: str) -> str:
    with open(SNAPSHOT / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    return str(SNAPSHOT / wm[key])


def bench_mx_load(shard_path: str, n: int) -> list[float]:
    times = []
    for _ in range(n):
        mx.clear_cache()
        t0 = time.perf_counter()
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shard = mx.load(shard_path)
            w, s, b = _load_proj_experts(shard, f"{KEY_BASE}.{proj}", EXPERT_IDS)
            mx.eval(w, s)
            if b is not None:
                mx.eval(b)
            del shard
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_mmap(st_map: SafetensorsMap, n: int) -> list[float]:
    times = []
    for _ in range(n):
        mx.clear_cache()
        t0 = time.perf_counter()
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w, s, b = _mmap_load_proj_experts(st_map, f"{KEY_BASE}.{proj}", EXPERT_IDS)
            mx.eval(w, s)
            if b is not None:
                mx.eval(b)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def stats(times: list[float]) -> dict:
    a = np.array(times)
    return {
        "median_ms": float(np.median(a) * 1000),
        "p95_ms": float(np.percentile(a, 95) * 1000),
        "min_ms": float(np.min(a) * 1000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    shard_path = find_shard_for_key(f"{KEY_BASE}.gate_proj.weight")
    print(f"Shard: {Path(shard_path).name}")
    print(f"Loading {len(np.asarray(EXPERT_IDS))} experts × 3 projections, {args.iters} iterations\n")

    # Warmup
    shard = mx.load(shard_path)
    del shard

    print("mx.load path...")
    mx_times = bench_mx_load(shard_path, args.iters)
    mx_stats = stats(mx_times)

    all_shards = sorted(str(p) for p in SNAPSHOT.glob("*.safetensors"))
    st_map = SafetensorsMap(all_shards)

    print("mmap path...")
    mm_times = bench_mmap(st_map, args.iters)
    mm_stats = stats(mm_times)
    st_map.close()

    print(f"\n{'Method':<12} {'Median':>10} {'P95':>10} {'Min':>10}")
    print("-" * 44)
    for name, s in [("mx.load", mx_stats), ("mmap", mm_stats)]:
        print(f"{name:<12} {s['median_ms']:>8.1f}ms {s['p95_ms']:>8.1f}ms {s['min_ms']:>8.1f}ms")

    speedup = mx_stats["median_ms"] / mm_stats["median_ms"]
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
