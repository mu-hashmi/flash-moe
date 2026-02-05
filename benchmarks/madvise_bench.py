#!/usr/bin/env python3
"""
Benchmark madvise(MADV_WILLNEED) effectiveness.

Questions answered:
1. Does MADV_WILLNEED trigger async page-in?
2. How long after madvise() is data actually in memory?
3. What's the minimum lead time needed?

Run with: sudo uv run python benchmarks/madvise_bench.py <model.gguf>
(sudo required for fs_usage and purge)
"""

import argparse
import mmap
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gguf_parser import GGUFModel


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


def measure_read_latency(mm: mmap.mmap, offset: int, size: int) -> float:
    start = time.perf_counter()
    data = mm[offset:offset + size]
    _ = bytes(data)
    return (time.perf_counter() - start) * 1000


def benchmark_madvise_willneed(model: GGUFModel, n_trials: int = 10, warmup: int = 2):
    """
    Test MADV_WILLNEED with varying delays to measure async prefetch behavior.
    """
    print("\n" + "=" * 60)
    print("MADVISE(MADV_WILLNEED) BENCHMARK")
    print("=" * 60)

    info = model.get_expert_offset(0, 0, 'down')
    expert_size = info.size
    expert_offset = info.offset

    print(f"Expert size: {expert_size / 1e6:.3f} MB")
    print(f"Expert offset: {expert_offset}")

    delays_ms = [0, 1, 2, 5, 10, 20, 50, 100]
    results = {d: [] for d in delays_ms}
    cold_reads = []

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        print("\n--- Cold Read Baseline (no madvise) ---")
        # Warmup iterations
        for _ in range(warmup):
            drop_caches()
            measure_read_latency(mm, info.offset, info.size)

        for trial in range(n_trials):
            layer = trial % model.n_layers
            expert = (trial * 37) % model.n_experts
            info = model.get_expert_offset(layer, expert, 'down')

            drop_caches()
            latency = measure_read_latency(mm, info.offset, info.size)
            cold_reads.append(latency)
            print(f"  Trial {trial+1}: {latency:.2f} ms")

        print(f"\nCold baseline: {np.mean(cold_reads):.2f} ± {np.std(cold_reads):.2f} ms")
        print_percentiles(cold_reads, "  ")

        print("\n--- MADV_WILLNEED with varying delays ---")
        for delay_ms in delays_ms:
            print(f"\nDelay: {delay_ms} ms")
            delay_results = []

            # Warmup for each delay setting
            for _ in range(warmup):
                drop_caches()
                mm.madvise(mmap.MADV_WILLNEED, info.offset, info.size)
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000)
                measure_read_latency(mm, info.offset, info.size)

            for trial in range(n_trials):
                layer = trial % model.n_layers
                expert = (trial * 37) % model.n_experts
                info = model.get_expert_offset(layer, expert, 'down')

                drop_caches()

                # Issue madvise WILLNEED
                mm.madvise(mmap.MADV_WILLNEED, info.offset, info.size)

                # Wait the specified delay
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000)

                # Now measure read latency
                latency = measure_read_latency(mm, info.offset, info.size)
                delay_results.append(latency)
                results[delay_ms].append(latency)

            avg = np.mean(delay_results)
            std = np.std(delay_results)
            speedup = np.mean(cold_reads) / avg if avg > 0 else 0
            print(f"  Avg: {avg:.2f} ± {std:.2f} ms (speedup: {speedup:.1f}x)")
            print_percentiles(delay_results, "  ")

        mm.close()

    return results, cold_reads


def benchmark_madvise_continuous_prefetch(model: GGUFModel, n_experts: int = 50, warmup: int = 2, n_trials: int = 5):
    """
    Simulate continuous prefetch: issue madvise for next expert while reading current.
    Measures if async prefetch provides any benefit.

    Uses non-consecutive experts across different layers to avoid measuring kernel
    sequential readahead instead of madvise effectiveness.
    """
    print("\n" + "=" * 60)
    print("CONTINUOUS PREFETCH SIMULATION")
    print("=" * 60)

    # Use non-consecutive experts across layers to avoid sequential readahead.
    # Spread across layers and use prime-step expert selection for non-locality.
    infos = []
    for i in range(n_experts):
        layer = (i * 7) % model.n_layers
        expert = (i * 37) % model.n_experts
        infos.append(model.get_expert_offset(layer, expert, 'down'))

    # Verify non-consecutive: check that offsets aren't sequential
    offsets = [info.offset for info in infos]
    consecutive_count = sum(1 for i in range(1, len(offsets)) if offsets[i] == offsets[i-1] + infos[i-1].size)
    print(f"Using {n_experts} experts spread across layers (consecutive pairs: {consecutive_count}/{n_experts-1})")

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        sequential_times = []
        prefetch_times = []

        # Warmup
        for _ in range(warmup):
            drop_caches()
            for info in infos:
                _ = bytes(mm[info.offset:info.offset + info.size])

        for trial in range(n_trials):
            # Sequential read (no prefetch)
            drop_caches()
            start = time.perf_counter()
            for info in infos:
                data = mm[info.offset:info.offset + info.size]
                _ = bytes(data)
            sequential_times.append((time.perf_counter() - start) * 1000)

            # With continuous prefetch
            drop_caches()
            start = time.perf_counter()
            for i, info in enumerate(infos):
                # Prefetch next expert while reading current
                if i + 1 < len(infos):
                    next_info = infos[i + 1]
                    mm.madvise(mmap.MADV_WILLNEED, next_info.offset, next_info.size)

                data = mm[info.offset:info.offset + info.size]
                _ = bytes(data)
            prefetch_times.append((time.perf_counter() - start) * 1000)

        mm.close()

    sequential_time = np.mean(sequential_times)
    prefetch_time = np.mean(prefetch_times)

    print(f"\nSequential ({n_experts} experts): {sequential_time:.1f} ms")
    print_percentiles(sequential_times, "  ")
    print(f"\nWith prefetch: {prefetch_time:.1f} ms")
    print_percentiles(prefetch_times, "  ")

    improvement = (sequential_time - prefetch_time) / sequential_time * 100
    print(f"\nImprovement: {improvement:.1f}%")

    per_expert_seq = sequential_time / n_experts
    per_expert_pre = prefetch_time / n_experts
    print(f"Per expert: {per_expert_seq:.2f} ms → {per_expert_pre:.2f} ms")

    return {
        'sequential_ms': sequential_time,
        'prefetch_ms': prefetch_time,
        'n_experts': n_experts,
    }


def benchmark_lookahead_prefetch(model: GGUFModel, n_experts: int = 50, warmup: int = 2, n_trials: int = 5):
    """
    Test different lookahead depths for prefetch.
    Uses non-consecutive experts to avoid measuring kernel sequential readahead.
    """
    print("\n" + "=" * 60)
    print("LOOKAHEAD PREFETCH DEPTH TEST")
    print("=" * 60)

    # Use non-consecutive experts across layers
    infos = []
    for i in range(n_experts):
        layer = (i * 7) % model.n_layers
        expert = (i * 37) % model.n_experts
        infos.append(model.get_expert_offset(layer, expert, 'down'))

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        results = {}
        all_latencies = {}

        for lookahead in [1, 2, 3, 5, 10]:
            latencies = []

            # Warmup
            for _ in range(warmup):
                drop_caches()
                for i, info in enumerate(infos):
                    for ahead in range(1, lookahead + 1):
                        if i + ahead < len(infos):
                            mm.madvise(mmap.MADV_WILLNEED, infos[i + ahead].offset, infos[i + ahead].size)
                    _ = bytes(mm[info.offset:info.offset + info.size])

            for _ in range(n_trials):
                drop_caches()
                start = time.perf_counter()

                for i, info in enumerate(infos):
                    # Prefetch multiple ahead
                    for ahead in range(1, lookahead + 1):
                        if i + ahead < len(infos):
                            next_info = infos[i + ahead]
                            mm.madvise(mmap.MADV_WILLNEED, next_info.offset, next_info.size)

                    data = mm[info.offset:info.offset + info.size]
                    _ = bytes(data)

                latencies.append((time.perf_counter() - start) * 1000)

            elapsed = np.mean(latencies)
            results[lookahead] = elapsed
            all_latencies[lookahead] = latencies
            print(f"\n  Lookahead {lookahead}: {elapsed:.1f} ms ({elapsed/n_experts:.2f} ms/expert)")
            print_percentiles(latencies, "    ")

        mm.close()

    return results


def verify_madvise_is_async(model: GGUFModel, n_trials: int = 5):
    """
    Verify if madvise(MADV_WILLNEED) is truly async by measuring:
    1. How long the madvise() call itself takes
    2. Whether I/O happens after madvise returns (via subsequent read speed)

    If madvise is blocking (sync), the call duration will be close to cold read time.
    If async, the call should return quickly and I/O happens in background.
    """
    print("\n" + "=" * 60)
    print("MADVISE ASYNC VERIFICATION")
    print("=" * 60)

    info = model.get_expert_offset(0, 0, 'down')
    expert_size_mb = info.size / 1e6

    madvise_durations = []
    cold_read_times = []
    post_madvise_read_times = []

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        for trial in range(n_trials):
            layer = trial % model.n_layers
            expert = (trial * 37) % model.n_experts
            info = model.get_expert_offset(layer, expert, 'down')

            # Measure cold read time (baseline)
            drop_caches()
            start = time.perf_counter()
            _ = bytes(mm[info.offset:info.offset + info.size])
            cold_read_times.append((time.perf_counter() - start) * 1000)

            # Measure madvise call duration
            drop_caches()
            start = time.perf_counter()
            mm.madvise(mmap.MADV_WILLNEED, info.offset, info.size)
            madvise_durations.append((time.perf_counter() - start) * 1000)

            # Immediately read after madvise (no wait)
            start = time.perf_counter()
            _ = bytes(mm[info.offset:info.offset + info.size])
            post_madvise_read_times.append((time.perf_counter() - start) * 1000)

        mm.close()

    cold_avg = np.mean(cold_read_times)
    madvise_avg = np.mean(madvise_durations)
    post_read_avg = np.mean(post_madvise_read_times)

    print(f"\nExpert size: {expert_size_mb:.2f} MB")
    print(f"\nCold read time (baseline): {cold_avg:.2f} ms")
    print(f"madvise() call duration: {madvise_avg:.3f} ms")
    print(f"Read immediately after madvise: {post_read_avg:.2f} ms")

    # Interpretation
    print("\n--- Interpretation ---")
    if madvise_avg < cold_avg * 0.1:
        print("✓ madvise() returns quickly (<10% of cold read time)")
        print("  This suggests madvise is async (just enqueues I/O request)")
    else:
        print("✗ madvise() takes significant time")
        print(f"  ({madvise_avg/cold_avg*100:.0f}% of cold read time)")
        print("  This suggests madvise may be partially or fully synchronous")

    total_time = madvise_avg + post_read_avg
    if total_time > cold_avg * 1.1:
        print(f"\n✗ madvise + immediate read ({total_time:.2f} ms) > cold read ({cold_avg:.2f} ms)")
        print("  No benefit from madvise without delay - I/O still happens on read")
    elif total_time < cold_avg * 0.9:
        print(f"\n✓ madvise + immediate read ({total_time:.2f} ms) < cold read ({cold_avg:.2f} ms)")
        print("  Some I/O completed during madvise call (partial prefetch)")
    else:
        print(f"\n≈ madvise + immediate read ({total_time:.2f} ms) ≈ cold read ({cold_avg:.2f} ms)")

    return {
        'madvise_duration_ms': madvise_avg,
        'cold_read_ms': cold_avg,
        'post_madvise_read_ms': post_read_avg,
        'is_async': madvise_avg < cold_avg * 0.1,
    }


def verify_with_fs_usage(model: GGUFModel):
    """
    Run a test with fs_usage to visually verify when page-in happens.
    """
    print("\n" + "=" * 60)
    print("FS_USAGE VERIFICATION")
    print("=" * 60)
    print("Starting fs_usage trace. Watch for page-in timing...")

    trace_file = Path(__file__).parent.parent / "trace_output" / "madvise_trace.txt"
    trace_file.parent.mkdir(exist_ok=True)

    info = model.get_expert_offset(0, 0, 'down')

    fs_proc = subprocess.Popen(
        ["sudo", "fs_usage", "-f", "filesys", "-w"],
        stdout=open(trace_file, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        drop_caches()
        print(f"[{time.strftime('%H:%M:%S')}] Cache dropped")
        time.sleep(0.2)

        print(f"[{time.strftime('%H:%M:%S')}] Issuing MADV_WILLNEED")
        madvise_start = time.perf_counter()
        mm.madvise(mmap.MADV_WILLNEED, info.offset, info.size)
        madvise_elapsed = (time.perf_counter() - madvise_start) * 1000
        print(f"[{time.strftime('%H:%M:%S')}] madvise returned in {madvise_elapsed:.3f} ms")

        print(f"[{time.strftime('%H:%M:%S')}] Waiting 50ms...")
        time.sleep(0.05)

        print(f"[{time.strftime('%H:%M:%S')}] Reading data")
        start = time.perf_counter()
        data = mm[info.offset:info.offset + info.size]
        _ = bytes(data)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[{time.strftime('%H:%M:%S')}] Read completed in {elapsed:.2f} ms")

        mm.close()

    time.sleep(0.5)
    fs_proc.terminate()
    fs_proc.wait(timeout=2)

    print(f"\nTrace written to: {trace_file}")
    print("Analyze with: grep -i 'page_in\\|pread\\|RdData' trace_output/madvise_trace.txt")


def main():
    parser = argparse.ArgumentParser(description="Benchmark madvise effectiveness")
    parser.add_argument("model", help="Path to GGUF model")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--verify", action="store_true", help="Run fs_usage verification")
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

    # Run benchmarks
    delay_results, cold_baseline = benchmark_madvise_willneed(model, n_trials=args.trials)
    async_results = verify_madvise_is_async(model)
    prefetch_results = benchmark_madvise_continuous_prefetch(model)
    lookahead_results = benchmark_lookahead_prefetch(model)

    if args.verify:
        verify_with_fs_usage(model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: MADVISE EFFECTIVENESS")
    print("=" * 60)

    cold_avg = np.mean(cold_baseline)
    print(f"\nCold read baseline: {cold_avg:.2f} ms")

    print("\nRead latency after MADV_WILLNEED + delay:")
    for delay_ms, latencies in delay_results.items():
        avg = np.mean(latencies)
        reduction = (cold_avg - avg) / cold_avg * 100 if cold_avg > avg else 0
        print(f"  {delay_ms:3d} ms delay: {avg:.2f} ms ({reduction:.0f}% reduction)")

    print("\nKey findings:")
    # madvise async status
    if async_results['is_async']:
        print(f"  - madvise is async (returns in {async_results['madvise_duration_ms']:.2f} ms)")
    else:
        print(f"  - madvise appears synchronous or partially blocking")

    # Determine minimum effective delay
    for delay_ms in sorted(delay_results.keys()):
        avg = np.mean(delay_results[delay_ms])
        if avg < cold_avg * 0.5:
            print(f"  - {delay_ms} ms lead time needed for >50% speedup")
            break
    else:
        print("  - No significant speedup observed (madvise may not help)")

    # Best lookahead
    best_lookahead = min(lookahead_results, key=lookahead_results.get)
    print(f"  - Best lookahead depth: {best_lookahead}")


if __name__ == "__main__":
    main()
