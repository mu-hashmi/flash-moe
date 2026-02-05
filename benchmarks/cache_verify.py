#!/usr/bin/env python3
"""
Verify warm cache speed with fs_usage tracing.

Questions answered:
1. Is the 10-15 GB/s warm cache number accurate?
2. What does fs_usage show for cached reads?
3. Is latency <0.1 ms per expert for cached data?

NOTE: All measurements include the memcpy from mmap to a new bytes buffer
(via `bytes(data)`). This is intentional as we'd need to copy data to GPU
anyway, so this measures the realistic workload of mmap + memcpy.

Run with: sudo uv run python benchmarks/cache_verify.py <model.gguf>
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


def print_percentiles(latencies: list[float], prefix: str = ""):
    """Print P50, P95, P99, min, max statistics."""
    arr = np.array(latencies)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    print(f"{prefix}P50: {p50:.3f} ms, P95: {p95:.3f} ms, P99: {p99:.3f} ms")
    print(f"{prefix}Min: {np.min(arr):.3f} ms, Max: {np.max(arr):.3f} ms")


def verify_warm_cache_speed(model: GGUFModel, n_trials: int = 100, warmup: int = 5):
    """
    Read the same expert multiple times to guarantee cached reads.
    Measure latency and throughput.
    """
    print("\n" + "=" * 60)
    print("WARM CACHE VERIFICATION")
    print("=" * 60)

    info = model.get_expert_offset(0, 0, 'down')
    expert_size = info.size

    print(f"Expert size: {expert_size / 1e6:.3f} MB")

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Warm up: read the data multiple times to ensure it's cached
        # and eliminate cold-start artifacts (TLB misses, etc.)
        print(f"\nWarming cache ({warmup} iterations)...")
        for _ in range(warmup):
            data = mm[info.offset:info.offset + info.size]
            _ = bytes(data)  # memcpy from mmap to new buffer

        # Now measure cached reads
        # Note: bytes(data) includes memcpy overhead, which is realistic
        # since we'd need to copy to GPU memory anyway
        print(f"\nMeasuring {n_trials} cached reads of same expert...")
        latencies = []
        throughputs = []

        for i in range(n_trials):
            start = time.perf_counter()
            data = mm[info.offset:info.offset + info.size]
            _ = bytes(data)
            elapsed = time.perf_counter() - start

            latencies.append(elapsed * 1000)
            throughputs.append(expert_size / elapsed / 1e9 if elapsed > 0 else 0)

        mm.close()

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    avg_throughput = np.mean(throughputs)

    print(f"\nResults (mmap slice + memcpy to bytes):")
    print(f"  Latency: {avg_latency:.3f} ± {std_latency:.3f} ms")
    print_percentiles(latencies, "  ")
    print(f"  Throughput: {avg_throughput:.2f} GB/s")

    target_met = avg_latency < 0.1
    print(f"\n  Target (<0.1 ms): {'PASS' if target_met else 'FAIL'}")

    return {
        'avg_ms': avg_latency,
        'std_ms': std_latency,
        'min_ms': min_latency,
        'max_ms': max_latency,
        'throughput_gbps': avg_throughput,
        'target_met': target_met,
    }


def verify_multiple_experts_cached(model: GGUFModel, n_experts: int = 50, warmup: int = 3, n_trials: int = 10):
    """
    Read multiple different experts (all cached) to verify throughput
    doesn't degrade with different offsets.
    """
    print("\n" + "=" * 60)
    print("MULTIPLE EXPERTS CACHED")
    print("=" * 60)

    infos = [model.get_expert_offset(0, i, 'down') for i in range(n_experts)]
    total_size = sum(info.size for info in infos)

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Warm up all experts (including warmup iterations for stable measurements)
        print(f"Warming cache for {n_experts} experts ({warmup} iterations)...")
        for _ in range(warmup):
            for info in infos:
                data = mm[info.offset:info.offset + info.size]
                _ = bytes(data)

        # Measure cached reads of all experts
        print(f"Measuring cached sequential read of all experts ({n_trials} trials)...")
        times = []

        for _ in range(n_trials):
            start = time.perf_counter()
            for info in infos:
                data = mm[info.offset:info.offset + info.size]
                _ = bytes(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        mm.close()

    avg_time = np.mean(times)
    per_expert = avg_time / n_experts
    throughput = total_size / (avg_time / 1000) / 1e9

    print(f"\nResults ({n_experts} experts):")
    print(f"  Total time: {avg_time:.2f} ms")
    print_percentiles(times, "  Total: ")
    print(f"  Per expert: {per_expert:.3f} ms")
    print(f"  Throughput: {throughput:.2f} GB/s")

    return {
        'n_experts': n_experts,
        'total_ms': avg_time,
        'per_expert_ms': per_expert,
        'throughput_gbps': throughput,
    }


def verify_with_fs_usage(model: GGUFModel):
    """
    Use fs_usage to confirm NO disk I/O for cached reads.
    """
    print("\n" + "=" * 60)
    print("FS_USAGE VERIFICATION (should show NO disk I/O)")
    print("=" * 60)

    trace_file = Path(__file__).parent.parent / "trace_output" / "cache_verify_trace.txt"
    trace_file.parent.mkdir(exist_ok=True)

    info = model.get_expert_offset(0, 0, 'down')

    # First warm the cache
    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for _ in range(3):
            data = mm[info.offset:info.offset + info.size]
            _ = bytes(data)
        mm.close()

    print("Cache warmed. Starting fs_usage trace...")
    time.sleep(0.5)

    fs_proc = subprocess.Popen(
        ["sudo", "fs_usage", "-f", "filesys", "-w"],
        stdout=open(trace_file, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    # Read from cache
    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        print(f"[{time.strftime('%H:%M:%S')}] Reading cached data (10 times)...")
        for i in range(10):
            start = time.perf_counter()
            data = mm[info.offset:info.offset + info.size]
            _ = bytes(data)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  Read {i+1}: {elapsed:.3f} ms")
            time.sleep(0.01)

        mm.close()

    time.sleep(0.5)
    fs_proc.terminate()
    fs_proc.wait(timeout=2)

    print(f"\nTrace written to: {trace_file}")
    print("\nExpected: NO RdData/PgIn events during the reads above")
    print("Check with: grep -c 'RdData\\|page_in' trace_output/cache_verify_trace.txt")

    # Analyze trace
    with open(trace_file) as f:
        content = f.read()
        model_name = model.path.name
        io_events = content.count('RdData') + content.count('page_in')
        if model_name in content:
            model_io = sum(1 for line in content.split('\n') if model_name in line)
            print(f"\nI/O events referencing model file: {model_io}")
        else:
            print(f"\nNo I/O events referencing model file found (GOOD)")

    return {'trace_file': str(trace_file)}


def verify_purge_effectiveness(model: GGUFModel):
    """
    Verify that purge actually evicts pages by checking that subsequent
    read shows significantly higher latency than warm cache.
    """
    print("\n" + "=" * 60)
    print("PURGE EFFECTIVENESS VERIFICATION")
    print("=" * 60)

    info = model.get_expert_offset(0, 0, 'down')

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Warm the cache
        for _ in range(3):
            _ = bytes(mm[info.offset:info.offset + info.size])

        # Measure warm read
        warm_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = bytes(mm[info.offset:info.offset + info.size])
            warm_times.append((time.perf_counter() - start) * 1000)

        warm_avg = np.mean(warm_times)
        print(f"Warm cache read: {warm_avg:.3f} ms")

        # Purge and measure cold read
        subprocess.run(["sudo", "purge"], capture_output=True)
        time.sleep(0.5)

        start = time.perf_counter()
        _ = bytes(mm[info.offset:info.offset + info.size])
        cold_time = (time.perf_counter() - start) * 1000

        print(f"Post-purge read: {cold_time:.2f} ms")

        mm.close()

    # Verify purge worked: cold read should be much slower than warm
    ratio = cold_time / warm_avg if warm_avg > 0 else 0
    purge_effective = ratio > 10  # Expect at least 10x difference

    if purge_effective:
        print(f"✓ Purge effective: post-purge read is {ratio:.0f}x slower than warm cache")
    else:
        print(f"✗ Purge may not be effective: only {ratio:.1f}x difference")
        print("  Consider increasing post-purge sleep or verifying with fs_usage")

    return {'purge_effective': purge_effective, 'cold_warm_ratio': ratio}


def compare_cold_vs_warm(model: GGUFModel, n_trials: int = 10, warmup: int = 2):
    """
    Side-by-side comparison of cold vs warm reads.
    """
    print("\n" + "=" * 60)
    print("COLD VS WARM COMPARISON")
    print("=" * 60)

    info = model.get_expert_offset(0, 0, 'down')

    cold_times = []
    warm_times = []

    with open(model.path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Warmup iterations (not counted)
        for _ in range(warmup):
            subprocess.run(["sudo", "purge"], capture_output=True)
            time.sleep(0.5)
            _ = bytes(mm[info.offset:info.offset + info.size])
            _ = bytes(mm[info.offset:info.offset + info.size])

        for trial in range(n_trials):
            # Cold read
            subprocess.run(["sudo", "purge"], capture_output=True)
            time.sleep(0.5)

            start = time.perf_counter()
            data = mm[info.offset:info.offset + info.size]
            _ = bytes(data)
            cold_times.append((time.perf_counter() - start) * 1000)

            # Warm read (same data just read)
            start = time.perf_counter()
            data = mm[info.offset:info.offset + info.size]
            _ = bytes(data)
            warm_times.append((time.perf_counter() - start) * 1000)

            print(f"  Trial {trial+1}: Cold={cold_times[-1]:.2f} ms, Warm={warm_times[-1]:.3f} ms")

        mm.close()

    cold_avg = np.mean(cold_times)
    warm_avg = np.mean(warm_times)
    speedup = cold_avg / warm_avg if warm_avg > 0 else 0

    print(f"\nSummary:")
    print(f"  Cold: {cold_avg:.2f} ± {np.std(cold_times):.2f} ms")
    print_percentiles(cold_times, "    ")
    print(f"  Warm: {warm_avg:.3f} ± {np.std(warm_times):.3f} ms")
    print_percentiles(warm_times, "    ")
    print(f"  Speedup: {speedup:.0f}x")

    return {
        'cold_avg_ms': cold_avg,
        'warm_avg_ms': warm_avg,
        'speedup': speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify warm cache performance")
    parser.add_argument("model", help="Path to GGUF model")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--skip-fs-usage", action="store_true")
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

    # Run verifications
    purge_check = verify_purge_effectiveness(model)
    single_results = verify_warm_cache_speed(model, n_trials=args.trials)
    multi_results = verify_multiple_experts_cached(model)
    comparison = compare_cold_vs_warm(model, n_trials=10)

    if not args.skip_fs_usage:
        verify_with_fs_usage(model)

    # Final summary
    print("\n" + "=" * 60)
    print("WARM CACHE VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"\nPurge effectiveness:")
    print(f"  Status: {'✓ VERIFIED' if purge_check['purge_effective'] else '✗ QUESTIONABLE'}")
    print(f"  Cold/warm ratio: {purge_check['cold_warm_ratio']:.0f}x")

    print(f"\nSingle expert cached read (mmap + memcpy):")
    print(f"  Latency: {single_results['avg_ms']:.3f} ms (target: <0.1 ms)")
    print(f"  Throughput: {single_results['throughput_gbps']:.2f} GB/s")
    print(f"  Target met: {'YES' if single_results['target_met'] else 'NO'}")

    print(f"\nMultiple experts cached:")
    print(f"  Per-expert: {multi_results['per_expert_ms']:.3f} ms")
    print(f"  Throughput: {multi_results['throughput_gbps']:.2f} GB/s")

    print(f"\nCold vs Warm:")
    print(f"  Cold: {comparison['cold_avg_ms']:.2f} ms")
    print(f"  Warm: {comparison['warm_avg_ms']:.3f} ms")
    print(f"  Speedup: {comparison['speedup']:.0f}x")


if __name__ == "__main__":
    main()
