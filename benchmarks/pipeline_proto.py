#!/usr/bin/env python3
"""
Prototype prefetch pipeline to test compute/IO overlap.

Questions answered:
1. Can we overlap prefetch with compute?
2. What's the effective throughput with pipelining?
3. How far ahead do we need to prefetch?

Run with: sudo uv run python benchmarks/pipeline_proto.py <model.gguf>
"""

import argparse
import asyncio
import mmap
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gguf_parser import GGUFModel, ExpertInfo


def drop_caches():
    subprocess.run(["sudo", "purge"], capture_output=True)
    time.sleep(0.5)


def print_percentiles(latencies: list[float], prefix: str = ""):
    """Print P50, P95, P99, min, max statistics."""
    arr = np.array(latencies)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    print(f"{prefix}P50: {p50:.2f} ms, P95: {p95:.2f} ms, P99: {p99:.2f} ms")
    print(f"{prefix}Min: {np.min(arr):.2f} ms, Max: {np.max(arr):.2f} ms")


def simulate_compute(data: bytes, compute_time_ms: float = 1.0):
    """Simulate GPU compute time by sleeping."""
    time.sleep(compute_time_ms / 1000)


class SequentialPipeline:
    """Baseline: read then compute, no overlap."""

    def __init__(self, model: GGUFModel, compute_time_ms: float = 1.0):
        self.model = model
        self.compute_time_ms = compute_time_ms
        self.mm = None
        self.f = None

    def __enter__(self):
        self.f = open(self.model.path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        if self.mm:
            self.mm.close()
        if self.f:
            self.f.close()

    def run(self, infos: list[ExpertInfo]) -> float:
        """Run sequential read+compute, return total time in ms."""
        start = time.perf_counter()

        for info in infos:
            data = self.mm[info.offset:info.offset + info.size]
            data_bytes = bytes(data)
            simulate_compute(data_bytes, self.compute_time_ms)

        return (time.perf_counter() - start) * 1000


class ThreadedPipeline:
    """Threaded: prefetch next while computing current."""

    def __init__(self, model: GGUFModel, compute_time_ms: float = 1.0, lookahead: int = 1):
        self.model = model
        self.compute_time_ms = compute_time_ms
        self.lookahead = lookahead
        self.mm = None
        self.f = None

    def __enter__(self):
        self.f = open(self.model.path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        if self.mm:
            self.mm.close()
        if self.f:
            self.f.close()

    def prefetch(self, info: ExpertInfo):
        """Prefetch expert data into memory."""
        self.mm.madvise(mmap.MADV_WILLNEED, info.offset, info.size)

    def read(self, info: ExpertInfo) -> bytes:
        """Read expert data."""
        return bytes(self.mm[info.offset:info.offset + info.size])

    def run(self, infos: list[ExpertInfo]) -> float:
        """Run pipelined read+compute with prefetch."""
        start = time.perf_counter()

        # Prefetch initial batch
        for i in range(min(self.lookahead, len(infos))):
            self.prefetch(infos[i])

        for i, info in enumerate(infos):
            # Start prefetch for future experts
            for ahead in range(1, self.lookahead + 1):
                future_idx = i + ahead
                if future_idx < len(infos):
                    self.prefetch(infos[future_idx])

            # Read current
            data = self.read(info)
            # Compute
            simulate_compute(data, self.compute_time_ms)

        return (time.perf_counter() - start) * 1000


class AsyncPipeline:
    """Async I/O with compute overlap using executor."""

    def __init__(self, model: GGUFModel, compute_time_ms: float = 1.0):
        self.model = model
        self.compute_time_ms = compute_time_ms
        self.mm = None
        self.f = None

    def __enter__(self):
        self.f = open(self.model.path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        if self.mm:
            self.mm.close()
        if self.f:
            self.f.close()

    def read_sync(self, info: ExpertInfo) -> bytes:
        return bytes(self.mm[info.offset:info.offset + info.size])

    async def run(self, infos: list[ExpertInfo]) -> float:
        """Run async pipelined processing."""
        start = time.perf_counter()
        loop = asyncio.get_event_loop()

        # Use thread executor for blocking I/O
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending_read = None

            for i, info in enumerate(infos):
                # Start reading next expert while processing current
                next_read = None
                if i + 1 < len(infos):
                    next_read = loop.run_in_executor(
                        executor, self.read_sync, infos[i + 1]
                    )

                # Get current data (from previous prefetch or read now)
                if pending_read:
                    data = await pending_read
                else:
                    data = self.read_sync(info)

                # Compute while next read happens in background
                await loop.run_in_executor(executor, simulate_compute, data, self.compute_time_ms)

                pending_read = next_read

            # Process final item if pending
            if pending_read:
                data = await pending_read
                await loop.run_in_executor(executor, simulate_compute, data, self.compute_time_ms)

        return (time.perf_counter() - start) * 1000


class DoubleBufferPipeline:
    """Double-buffering: two buffers alternating between I/O and compute."""

    def __init__(self, model: GGUFModel, compute_time_ms: float = 1.0):
        self.model = model
        self.compute_time_ms = compute_time_ms
        self.mm = None
        self.f = None

    def __enter__(self):
        self.f = open(self.model.path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        if self.mm:
            self.mm.close()
        if self.f:
            self.f.close()

    def run(self, infos: list[ExpertInfo]) -> float:
        """Run double-buffered pipeline."""
        if len(infos) < 2:
            return 0

        start = time.perf_counter()

        read_queue = Queue(maxsize=2)
        done = threading.Event()

        def reader_thread():
            for info in infos:
                data = bytes(self.mm[info.offset:info.offset + info.size])
                read_queue.put(data)
            done.set()

        reader = threading.Thread(target=reader_thread)
        reader.start()

        processed = 0
        while not (done.is_set() and read_queue.empty()):
            try:
                data = read_queue.get_nowait()
                simulate_compute(data, self.compute_time_ms)
                processed += 1
            except Empty:
                time.sleep(0.0001)  # 0.1ms spin wait

        reader.join()

        return (time.perf_counter() - start) * 1000


def benchmark_pipelines(model: GGUFModel, n_experts: int = 100, compute_time_ms: float = 1.0,
                        n_trials: int = 5, warmup: int = 2):
    """Compare different pipeline strategies."""
    print("\n" + "=" * 60)
    print(f"PIPELINE BENCHMARK (n={n_experts}, compute={compute_time_ms}ms, trials={n_trials})")
    print("=" * 60)

    infos = [model.get_expert_offset(0, i % model.n_experts, 'down') for i in range(n_experts)]
    expert_size = infos[0].size

    results = {}
    all_latencies = {}

    def run_trials(name: str, pipeline_class, **kwargs):
        latencies = []
        # Warmup iterations (excluded from measurements)
        for _ in range(warmup):
            drop_caches()
            with pipeline_class(model, compute_time_ms, **kwargs) as pipeline:
                pipeline.run(infos)

        # Measured iterations
        for _ in range(n_trials):
            drop_caches()
            with pipeline_class(model, compute_time_ms, **kwargs) as pipeline:
                latencies.append(pipeline.run(infos))
        return latencies

    # Sequential (baseline)
    print("\n1. Sequential (no overlap)...")
    seq_latencies = run_trials('sequential', SequentialPipeline)
    seq_time = np.mean(seq_latencies)
    results['sequential'] = seq_time
    all_latencies['sequential'] = seq_latencies
    print(f"   Mean: {seq_time:.1f} ms ({seq_time/n_experts:.2f} ms/expert)")
    print_percentiles(seq_latencies, "   ")

    # Threaded with different lookahead
    for lookahead in [1, 2, 5]:
        print(f"\n2. Threaded (lookahead={lookahead})...")
        threaded_latencies = run_trials(f'threaded_la{lookahead}', ThreadedPipeline, lookahead=lookahead)
        threaded_time = np.mean(threaded_latencies)
        results[f'threaded_la{lookahead}'] = threaded_time
        all_latencies[f'threaded_la{lookahead}'] = threaded_latencies
        speedup = seq_time / threaded_time if threaded_time > 0 else 0
        print(f"   Mean: {threaded_time:.1f} ms ({speedup:.2f}x vs sequential)")
        print_percentiles(threaded_latencies, "   ")

    # Double buffer
    print("\n3. Double buffer...")
    db_latencies = run_trials('double_buffer', DoubleBufferPipeline)
    db_time = np.mean(db_latencies)
    results['double_buffer'] = db_time
    all_latencies['double_buffer'] = db_latencies
    speedup = seq_time / db_time if db_time > 0 else 0
    print(f"   Mean: {db_time:.1f} ms ({speedup:.2f}x vs sequential)")
    print_percentiles(db_latencies, "   ")

    # Calculate theoretical limits
    total_data = n_experts * expert_size
    io_time_ideal = total_data / (3e9) * 1000  # Assume 3 GB/s SSD
    compute_time_total = n_experts * compute_time_ms
    theoretical_min = max(io_time_ideal, compute_time_total)

    print(f"\n--- Theoretical Analysis ---")
    print(f"I/O bound time (3 GB/s): {io_time_ideal:.1f} ms")
    print(f"Compute bound time: {compute_time_total:.1f} ms")
    print(f"Theoretical min (perfect overlap): {theoretical_min:.1f} ms")
    print(f"Best achieved: {min(results.values()):.1f} ms")
    print(f"Efficiency: {theoretical_min / min(results.values()) * 100:.0f}%")

    return results


def benchmark_compute_variations(model: GGUFModel, n_experts: int = 50):
    """Test pipeline efficiency with different compute times."""
    print("\n" + "=" * 60)
    print("COMPUTE TIME VARIATION TEST")
    print("=" * 60)

    infos = [model.get_expert_offset(0, i % model.n_experts, 'down') for i in range(n_experts)]

    compute_times = [0.5, 1.0, 2.0, 5.0, 10.0]

    print("\nCompute Time | Sequential | Pipelined | Speedup | Efficiency")
    print("-" * 65)

    for ct in compute_times:
        drop_caches()
        with SequentialPipeline(model, ct) as seq:
            seq_time = seq.run(infos)

        drop_caches()
        with DoubleBufferPipeline(model, ct) as pipe:
            pipe_time = pipe.run(infos)

        speedup = seq_time / pipe_time if pipe_time > 0 else 0
        # Efficiency vs. ideal (all overlap)
        io_portion = seq_time - n_experts * ct
        ideal = max(io_portion, n_experts * ct)
        efficiency = ideal / pipe_time * 100 if pipe_time > 0 else 0

        print(f"  {ct:6.1f} ms  | {seq_time:8.1f} ms | {pipe_time:8.1f} ms | {speedup:6.2f}x | {efficiency:5.0f}%")


def trace_pipeline_with_fs_usage(model: GGUFModel, n_experts: int = 20):
    """Run pipeline with fs_usage to verify I/O overlaps with compute."""
    print("\n" + "=" * 60)
    print("FS_USAGE TRACE OF PIPELINE")
    print("=" * 60)

    trace_file = Path(__file__).parent.parent / "trace_output" / "pipeline_trace.txt"
    trace_file.parent.mkdir(exist_ok=True)

    infos = [model.get_expert_offset(0, i % model.n_experts, 'down') for i in range(n_experts)]

    drop_caches()

    fs_proc = subprocess.Popen(
        ["sudo", "fs_usage", "-f", "filesys", "-w"],
        stdout=open(trace_file, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    print("Running pipelined reads with fs_usage tracing...")
    with DoubleBufferPipeline(model, compute_time_ms=5.0) as pipeline:
        elapsed = pipeline.run(infos)

    time.sleep(0.5)
    fs_proc.terminate()
    fs_proc.wait(timeout=2)

    print(f"Pipeline completed in {elapsed:.1f} ms")
    print(f"\nTrace written to: {trace_file}")
    print("Look for interleaved RdData and sleep events to verify overlap")


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefetch pipeline")
    parser.add_argument("model", help="Path to GGUF model")
    parser.add_argument("--experts", type=int, default=100, help="Number of experts to process")
    parser.add_argument("--compute-ms", type=float, default=1.0, help="Simulated compute time")
    parser.add_argument("--trace", action="store_true", help="Run with fs_usage trace")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        sys.exit(1)

    # Note: requires passwordless sudo for purge and fs_usage
    # Add to /etc/sudoers.d/claude-benchmarks:
    # muhash ALL=(ALL) NOPASSWD: /usr/sbin/purge, /usr/bin/fs_usage

    model = GGUFModel(model_path)
    print(f"Model: {model_path.name}")
    print(f"Layers: {model.n_layers}, Experts: {model.n_experts}")
    print(f"Expert size: {model.get_expert_offset(0, 0, 'down').size / 1e6:.3f} MB")

    # Run benchmarks
    results = benchmark_pipelines(model, args.experts, args.compute_ms)
    benchmark_compute_variations(model)

    if args.trace:
        trace_pipeline_with_fs_usage(model)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE PROTOTYPE SUMMARY")
    print("=" * 60)

    best = min(results.values())
    best_name = min(results, key=results.get)
    seq = results['sequential']

    print(f"\nBest pipeline: {best_name}")
    print(f"Speedup vs sequential: {seq/best:.2f}x")
    print(f"Time per expert: {best/args.experts:.2f} ms")

    # Viability calculation
    print(f"\nViability estimate (480 experts/token):")
    per_expert_ms = best / args.experts
    token_time_ms = 480 * per_expert_ms
    tok_per_sec = 1000 / token_time_ms
    print(f"  Per-expert time: {per_expert_ms:.2f} ms")
    print(f"  Token time: {token_time_ms:.0f} ms")
    print(f"  Throughput: {tok_per_sec:.2f} tok/s")


if __name__ == "__main__":
    main()
