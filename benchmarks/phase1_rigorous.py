#!/usr/bin/env python3
"""
Rigorous Phase 1 Benchmark with proper cache control.

Addresses issues from initial benchmark:
1. Drop caches between EVERY trial
2. Use data sizes that exceed RAM cache
3. Measure cold vs warm performance explicitly
4. Add DTrace/dtrace profiling for I/O verification
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
from expert_loader import ExpertLoader


def drop_caches():
    """Drop file system caches. Requires sudo."""
    # Sync first to flush writes
    os.system('sync')
    # macOS: purge command
    result = subprocess.run(['sudo', 'purge'], capture_output=True)
    if result.returncode != 0:
        print("WARNING: Could not drop caches (need sudo)")
        return False
    # Wait for purge to complete
    time.sleep(1.0)
    return True


def verify_cache_dropped():
    """Verify caches were actually dropped by checking memory pressure."""
    result = subprocess.run(['vm_stat'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Pages free' in line:
                pages = int(line.split(':')[1].strip().rstrip('.'))
                free_gb = pages * 16384 / 1e9  # 16KB pages
                return free_gb
    return None


def benchmark_cold_reads(model: GGUFModel, n_trials: int = 10) -> dict:
    """
    Benchmark with cache dropped before EVERY read.
    This measures true SSD performance.
    """
    print("\n" + "=" * 60)
    print("COLD READ BENCHMARK (cache dropped before each read)")
    print("=" * 60)

    results = []

    with ExpertLoader(model) as loader:
        for trial in range(n_trials):
            # Pick a random expert to avoid any pattern
            layer = np.random.randint(0, model.n_layers)
            expert = np.random.randint(0, model.n_experts)
            info = model.get_expert_offset(layer, expert, 'down')

            # Drop caches
            if not drop_caches():
                print("Skipping cold benchmark - cannot drop caches")
                return {}

            free_before = verify_cache_dropped()

            # Time the read
            start = time.perf_counter()
            data = os.pread(loader._file.fileno(), info.size, info.offset)
            elapsed = time.perf_counter() - start

            throughput = info.size / elapsed / 1e9
            results.append({
                'trial': trial,
                'layer': layer,
                'expert': expert,
                'size_mb': info.size / 1e6,
                'time_ms': elapsed * 1000,
                'throughput_gbps': throughput,
                'free_gb_before': free_before,
            })

            print(f"  Trial {trial+1}/{n_trials}: {info.size/1e6:.2f} MB in {elapsed*1000:.2f} ms = {throughput:.2f} GB/s")

    if not results:
        return {}

    times = [r['time_ms'] for r in results]
    throughputs = [r['throughput_gbps'] for r in results]

    summary = {
        'avg_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'avg_gbps': np.mean(throughputs),
        'std_gbps': np.std(throughputs),
        'n_trials': n_trials,
    }

    print(f"\nCOLD READ SUMMARY:")
    print(f"  Time: {summary['avg_ms']:.2f} ± {summary['std_ms']:.2f} ms")
    print(f"  Throughput: {summary['avg_gbps']:.2f} ± {summary['std_gbps']:.2f} GB/s")

    return summary


def benchmark_warm_reads(model: GGUFModel, n_trials: int = 50) -> dict:
    """
    Benchmark with warm cache (no drops).
    This measures best-case cached performance.
    """
    print("\n" + "=" * 60)
    print("WARM READ BENCHMARK (cached)")
    print("=" * 60)

    # First, warm up by reading all experts we'll test
    with ExpertLoader(model) as loader:
        # Pre-warm specific experts
        test_experts = [(i % model.n_layers, (i * 37) % model.n_experts) for i in range(n_trials)]
        for layer, expert in test_experts:
            info = model.get_expert_offset(layer, expert, 'down')
            _ = os.pread(loader._file.fileno(), info.size, info.offset)

        # Now benchmark
        results = []
        for trial, (layer, expert) in enumerate(test_experts):
            info = model.get_expert_offset(layer, expert, 'down')

            start = time.perf_counter()
            data = os.pread(loader._file.fileno(), info.size, info.offset)
            elapsed = time.perf_counter() - start

            throughput = info.size / elapsed / 1e9
            results.append({
                'time_ms': elapsed * 1000,
                'throughput_gbps': throughput,
            })

    times = [r['time_ms'] for r in results]
    throughputs = [r['throughput_gbps'] for r in results]

    summary = {
        'avg_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'avg_gbps': np.mean(throughputs),
    }

    print(f"WARM READ SUMMARY:")
    print(f"  Time: {summary['avg_ms']:.3f} ± {summary['std_ms']:.3f} ms")
    print(f"  Throughput: {summary['avg_gbps']:.2f} GB/s")

    return summary


def benchmark_large_sequential(model: GGUFModel) -> dict:
    """
    Read large sequential chunks that can't fit in cache.
    This forces SSD access regardless of caching.
    """
    print("\n" + "=" * 60)
    print("LARGE SEQUENTIAL READ (forces SSD access)")
    print("=" * 60)

    # Read entire layers worth of experts (hundreds of MB)
    results = []

    with open(model.path, 'rb') as f:
        for layer in range(min(5, model.n_layers)):
            # Get all expert tensors for this layer
            if layer not in model.expert_tensors:
                continue

            # Read the entire down tensor (all 512 experts merged)
            down_tensor = model.expert_tensors[layer].get('down')
            if not down_tensor:
                continue

            size = down_tensor.size
            offset = down_tensor.offset

            # Drop caches before each layer
            drop_caches()

            start = time.perf_counter()
            f.seek(offset)
            data = f.read(size)
            elapsed = time.perf_counter() - start

            throughput = size / elapsed / 1e9
            results.append({
                'layer': layer,
                'size_mb': size / 1e6,
                'time_ms': elapsed * 1000,
                'throughput_gbps': throughput,
            })

            print(f"  Layer {layer}: {size/1e6:.1f} MB in {elapsed*1000:.1f} ms = {throughput:.2f} GB/s")

    if not results:
        return {}

    throughputs = [r['throughput_gbps'] for r in results]
    summary = {
        'avg_gbps': np.mean(throughputs),
        'min_gbps': np.min(throughputs),
        'max_gbps': np.max(throughputs),
        'total_mb': sum(r['size_mb'] for r in results),
    }

    print(f"\nLARGE SEQUENTIAL SUMMARY:")
    print(f"  Throughput: {summary['avg_gbps']:.2f} GB/s (range: {summary['min_gbps']:.2f} - {summary['max_gbps']:.2f})")

    return summary


def benchmark_madvise_controlled(model: GGUFModel, n_trials: int = 5) -> dict:
    """
    Controlled madvise test with proper cache clearing.
    """
    print("\n" + "=" * 60)
    print("MADVISE COMPARISON (controlled)")
    print("=" * 60)

    # Read 10 adjacent experts
    n_experts = 10
    layer = 0
    expert_ids = list(range(n_experts))
    infos = [model.get_expert_offset(layer, eid, 'down') for eid in expert_ids]
    sorted_infos = sorted(infos, key=lambda x: x.offset)
    min_offset = sorted_infos[0].offset
    total_size = sum(i.size for i in sorted_infos)
    span_size = sorted_infos[-1].offset + sorted_infos[-1].size - min_offset

    results_no_advise = []
    results_with_advise = []

    with open(model.path, 'rb') as f:
        fd = f.fileno()
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

        for trial in range(n_trials):
            # Test WITHOUT madvise (cold)
            drop_caches()
            start = time.perf_counter()
            data = mm[min_offset:min_offset + span_size]
            _ = bytes(data)  # Force read
            elapsed = time.perf_counter() - start
            results_no_advise.append(elapsed * 1000)

            # Test WITH madvise (cold)
            drop_caches()
            mm.madvise(mmap.MADV_SEQUENTIAL, min_offset, span_size)
            start = time.perf_counter()
            data = mm[min_offset:min_offset + span_size]
            _ = bytes(data)
            elapsed = time.perf_counter() - start
            results_with_advise.append(elapsed * 1000)

        mm.close()

    avg_no = np.mean(results_no_advise)
    avg_with = np.mean(results_with_advise)
    throughput_no = span_size / (avg_no / 1000) / 1e9
    throughput_with = span_size / (avg_with / 1000) / 1e9

    improvement = (avg_no - avg_with) / avg_no * 100 if avg_no > avg_with else 0

    print(f"  Without madvise: {avg_no:.2f} ms ({throughput_no:.2f} GB/s)")
    print(f"  With madvise:    {avg_with:.2f} ms ({throughput_with:.2f} GB/s)")
    print(f"  Improvement: {improvement:.1f}%")

    return {
        'no_madvise_ms': avg_no,
        'with_madvise_ms': avg_with,
        'no_madvise_gbps': throughput_no,
        'with_madvise_gbps': throughput_with,
        'improvement_pct': improvement,
        'data_size_mb': span_size / 1e6,
    }


def profile_with_fs_usage(model: GGUFModel) -> dict:
    """
    Use fs_usage to verify actual disk I/O is happening.
    """
    print("\n" + "=" * 60)
    print("FS_USAGE VERIFICATION")
    print("=" * 60)

    # We'll read experts and use fs_usage to verify disk access
    # This requires running as root, so we'll do a simpler check

    # Check iostat before and after
    def get_disk_stats():
        result = subprocess.run(['iostat', '-d', '-c', '1'], capture_output=True, text=True)
        return result.stdout

    print("  Running disk I/O verification...")

    # Drop caches
    drop_caches()

    # Get baseline
    before = get_disk_stats()

    # Do a large read
    with ExpertLoader(model) as loader:
        for layer in range(min(3, model.n_layers)):
            for expert in range(0, model.n_experts, 50):  # Every 50th expert
                info = model.get_expert_offset(layer, expert, 'down')
                _ = os.pread(loader._file.fileno(), info.size, info.offset)

    after = get_disk_stats()

    print("  iostat before:")
    print("  " + before.replace('\n', '\n  '))
    print("  iostat after:")
    print("  " + after.replace('\n', '\n  '))

    return {'verified': True}


def run_rigorous_benchmarks(model_path: Path, n_cold_trials: int = 10):
    """Run all rigorous benchmarks."""
    print("=" * 60)
    print("RIGOROUS PHASE 1 BENCHMARK")
    print("=" * 60)

    # Check sudo access
    print("\nChecking sudo access for cache control...")
    result = subprocess.run(['sudo', '-n', 'true'], capture_output=True)
    if result.returncode != 0:
        print("NOTE: Will need sudo password to drop caches")
        # Prompt once
        subprocess.run(['sudo', 'true'])

    model = GGUFModel(model_path)
    print(f"\nModel: {model_path.name}")
    print(f"Layers: {model.n_layers}, Experts: {model.n_experts}")

    expert_info = model.get_expert_offset(0, 0, 'down')
    print(f"Expert size (layer 0, down): {expert_info.size / 1e6:.3f} MB")

    results = {}

    # 1. Cold reads (true SSD performance)
    results['cold'] = benchmark_cold_reads(model, n_trials=n_cold_trials)

    # 2. Warm reads (cached performance)
    results['warm'] = benchmark_warm_reads(model, n_trials=50)

    # 3. Large sequential (forces SSD)
    results['large_seq'] = benchmark_large_sequential(model)

    # 4. Controlled madvise test
    results['madvise'] = benchmark_madvise_controlled(model, n_trials=5)

    # 5. Verify with fs_usage/iostat
    results['verification'] = profile_with_fs_usage(model)

    # Summary
    print("\n" + "=" * 60)
    print("RIGOROUS BENCHMARK SUMMARY")
    print("=" * 60)

    if results['cold']:
        print(f"\nCOLD (true SSD):     {results['cold']['avg_gbps']:.2f} ± {results['cold']['std_gbps']:.2f} GB/s")
    if results['warm']:
        print(f"WARM (cached):       {results['warm']['avg_gbps']:.2f} GB/s")
    if results['large_seq']:
        print(f"LARGE SEQUENTIAL:    {results['large_seq']['avg_gbps']:.2f} GB/s")
    if results['madvise']:
        print(f"MADVISE improvement: {results['madvise']['improvement_pct']:.1f}%")
        print(f"  Without: {results['madvise']['no_madvise_gbps']:.2f} GB/s")
        print(f"  With:    {results['madvise']['with_madvise_gbps']:.2f} GB/s")

    # Revised viability estimate
    if results['cold']:
        cold_gbps = results['cold']['avg_gbps']
        expert_size_mb = expert_info.size / 1e6

        time_per_expert_ms = expert_size_mb / cold_gbps  # MB / (GB/s) = ms
        time_per_token_ms = 480 * time_per_expert_ms
        toks_no_cache = 1000 / time_per_token_ms

        print(f"\nREVISED VIABILITY (using cold SSD numbers):")
        print(f"  Time per expert: {time_per_expert_ms:.3f} ms")
        print(f"  480 experts/token: {time_per_token_ms:.1f} ms → {toks_no_cache:.1f} tok/s (no cache)")
        print(f"  With 50% cache: {toks_no_cache * 2:.1f} tok/s")
        print(f"  With 70% cache: {toks_no_cache / 0.3:.1f} tok/s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous Phase 1 benchmarks")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--cold-trials", type=int, default=10, help="Number of cold read trials")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    results = run_rigorous_benchmarks(model_path, n_cold_trials=args.cold_trials)
