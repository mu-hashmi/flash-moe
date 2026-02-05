#!/usr/bin/env python3
"""
Phase 1 Benchmark: Validate expert loading viability.

Questions to answer:
1. Can you achieve 2+ GB/s reading individual experts?
2. Does batching reads help?
3. Does madvise(MADV_SEQUENTIAL) improve throughput?
4. What's the Metal buffer creation overhead?
"""

import argparse
import ctypes
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gguf_parser import GGUFModel
from expert_loader import ExpertLoader, verify_expert_extraction
from metal_buffer import MetalBufferManager


def drop_caches():
    """Attempt to drop filesystem caches (macOS)."""
    os.system('sync; sudo purge 2>/dev/null')
    time.sleep(0.5)


def benchmark_single_expert_read(model: GGUFModel, n_trials: int = 50) -> dict:
    """Q1: Can we achieve 2+ GB/s reading individual experts?"""
    print("\n" + "=" * 60)
    print("Q1: Single Expert Read Performance")
    print("=" * 60)

    results = {}

    with ExpertLoader(model) as loader:
        for method in ['seek', 'mmap', 'pread']:
            times = []
            sizes = []

            for trial in range(n_trials):
                layer = trial % model.n_layers
                expert = (trial * 37) % model.n_experts

                info = model.get_expert_offset(layer, expert, 'down')
                result = getattr(loader, f'load_expert_{method}')(info)
                times.append(result.elapsed_ms)
                sizes.append(info.size)

            avg_size = np.mean(sizes)
            avg_time = np.mean(times)
            avg_throughput = avg_size / (avg_time / 1000) / 1e9

            results[method] = {
                'avg_ms': avg_time,
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'avg_gbps': avg_throughput,
                'size_mb': avg_size / 1e6,
            }

            print(f"\n{method.upper()}:")
            print(f"  Size: {avg_size/1e6:.3f} MB")
            print(f"  Time: {avg_time:.3f} ± {np.std(times):.3f} ms (min: {np.min(times):.3f}, max: {np.max(times):.3f})")
            print(f"  Throughput: {avg_throughput:.2f} GB/s")
            print(f"  Target (2 GB/s): {'✓ PASS' if avg_throughput >= 2 else '✗ FAIL'}")

    return results


def benchmark_batch_vs_individual(model: GGUFModel, batch_sizes: list[int] = None) -> dict:
    """Q2: Does batching reads help?"""
    print("\n" + "=" * 60)
    print("Q2: Batch vs Individual Reads")
    print("=" * 60)

    if batch_sizes is None:
        batch_sizes = [1, 5, 10, 20, 50]

    results = {}

    with ExpertLoader(model) as loader:
        for batch_size in batch_sizes:
            if batch_size > model.n_experts:
                continue

            times = []
            total_sizes = []
            n_trials = max(5, 50 // batch_size)

            for trial in range(n_trials):
                layer = trial % model.n_layers
                expert_ids = [(trial * 37 + i * 53) % model.n_experts for i in range(batch_size)]
                infos = [model.get_expert_offset(layer, eid, 'down') for eid in expert_ids]

                _, total_ms, _ = loader.load_experts_batch(infos, 'pread')
                times.append(total_ms)
                total_sizes.append(sum(i.size for i in infos))

            avg_size = np.mean(total_sizes)
            avg_time = np.mean(times)
            avg_throughput = avg_size / (avg_time / 1000) / 1e9
            per_expert_ms = avg_time / batch_size

            results[batch_size] = {
                'total_ms': avg_time,
                'per_expert_ms': per_expert_ms,
                'throughput_gbps': avg_throughput,
            }

            print(f"\nBatch size {batch_size}:")
            print(f"  Total time: {avg_time:.3f} ms")
            print(f"  Per-expert: {per_expert_ms:.3f} ms")
            print(f"  Throughput: {avg_throughput:.2f} GB/s")

    # Summary
    if 1 in results and 10 in results:
        single = results[1]['per_expert_ms']
        batch = results[10]['per_expert_ms']
        improvement = (single - batch) / single * 100
        print(f"\nBatch improvement: {improvement:.1f}% faster per-expert with batch=10 vs single")

    return results


def benchmark_madvise(model: GGUFModel, n_trials: int = 20) -> dict:
    """Q3: Does madvise(MADV_SEQUENTIAL) improve throughput?"""
    print("\n" + "=" * 60)
    print("Q3: madvise Effect on Sequential Reads")
    print("=" * 60)

    results = {}
    batch_size = 10

    # Get contiguous experts for sequential read test
    layer = 0
    expert_ids = list(range(batch_size))
    infos = [model.get_expert_offset(layer, eid, 'down') for eid in expert_ids]
    sorted_infos = sorted(infos, key=lambda x: x.offset)
    min_offset = sorted_infos[0].offset
    total_size = sum(i.size for i in sorted_infos)
    span_size = sorted_infos[-1].offset + sorted_infos[-1].size - min_offset

    with open(model.path, 'rb') as f:
        fd = f.fileno()

        # Test without madvise
        times_no_advise = []
        for _ in range(n_trials):
            start = time.perf_counter()
            data = os.pread(fd, span_size, min_offset)
            elapsed = (time.perf_counter() - start) * 1000
            times_no_advise.append(elapsed)

        # Test with mmap + madvise
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

        times_with_advise = []
        for _ in range(n_trials):
            # Apply madvise hint
            try:
                mm.madvise(mmap.MADV_SEQUENTIAL, min_offset, span_size)
            except Exception:
                pass  # May not be available

            start = time.perf_counter()
            data = mm[min_offset:min_offset + span_size]
            _ = bytes(data)  # Force page fault
            elapsed = (time.perf_counter() - start) * 1000
            times_with_advise.append(elapsed)

        mm.close()

    avg_no = np.mean(times_no_advise)
    avg_with = np.mean(times_with_advise)
    throughput_no = span_size / (avg_no / 1000) / 1e9
    throughput_with = span_size / (avg_with / 1000) / 1e9

    results = {
        'no_madvise': {'avg_ms': avg_no, 'throughput_gbps': throughput_no},
        'with_madvise': {'avg_ms': avg_with, 'throughput_gbps': throughput_with},
        'improvement_pct': (avg_no - avg_with) / avg_no * 100 if avg_no > avg_with else 0,
    }

    print(f"\nWithout madvise: {avg_no:.3f} ms, {throughput_no:.2f} GB/s")
    print(f"With madvise:    {avg_with:.3f} ms, {throughput_with:.2f} GB/s")
    print(f"Improvement: {results['improvement_pct']:.1f}%")

    return results


def benchmark_metal_overhead() -> dict:
    """Q4: What's the Metal buffer creation overhead?"""
    print("\n" + "=" * 60)
    print("Q4: Metal Buffer Creation Overhead")
    print("=" * 60)

    mgr = MetalBufferManager()
    info = mgr.get_device_info()

    print(f"\nDevice: {info['name']}")
    print(f"Working set: {info['recommended_working_set']:.1f} GB")
    print(f"Max buffer: {info['max_buffer_length']:.1f} GB")

    results = {}

    sizes = [
        (512 * 1024, "512 KB"),
        (1 * 1024 * 1024, "1 MB"),
        (4 * 1024 * 1024, "4 MB"),
        (16 * 1024 * 1024, "16 MB"),
    ]

    print("\nDirect bytes → Metal buffer timing:")
    print("-" * 60)
    print(f"{'Size':12} | {'Create':>12} | {'Kernel':>12} | {'Total':>12}")
    print("-" * 60)

    for size, label in sizes:
        stats = mgr.benchmark_direct_metal(size, n_trials=20)
        print(f"{label:12} | {stats.creation_ms:10.3f} ms | {stats.kernel_ms:10.3f} ms | {stats.total_ms:10.3f} ms")
        results[label] = {
            'creation_ms': stats.creation_ms,
            'kernel_ms': stats.kernel_ms,
            'total_ms': stats.total_ms,
        }

    return results


def run_end_to_end(model: GGUFModel, n_experts: int = 10) -> dict:
    """End-to-end test: Load experts from disk into Metal."""
    print("\n" + "=" * 60)
    print("End-to-End: Disk → CPU → Metal")
    print("=" * 60)

    mgr = MetalBufferManager()
    results = []

    with ExpertLoader(model) as loader:
        for trial in range(5):
            layer = trial % model.n_layers
            expert_ids = [(trial * 37 + i * 53) % model.n_experts for i in range(n_experts)]

            total_start = time.perf_counter()

            # Load from disk
            disk_start = time.perf_counter()
            loaded = []
            for eid in expert_ids:
                info = model.get_expert_offset(layer, eid, 'down')
                result = loader.load_expert_pread(info)
                loaded.append((info, result.data))
            disk_time = (time.perf_counter() - disk_start) * 1000

            # Create Metal buffers
            metal_start = time.perf_counter()
            metal_bufs = []
            for info, data in loaded:
                metal_buf = mgr.create_metal_buffer_direct(data)
                metal_bufs.append(metal_buf)
            metal_time = (time.perf_counter() - metal_start) * 1000

            # Run kernel on each
            kernel_start = time.perf_counter()
            for metal_buf in metal_bufs:
                _ = mgr.run_kernel_on_metal_buffer(metal_buf)
            kernel_time = (time.perf_counter() - kernel_start) * 1000

            # Cleanup (let GC handle Metal buffers)
            del metal_bufs

            total_time = (time.perf_counter() - total_start) * 1000
            total_size = sum(info.size for info, _ in loaded)

            results.append({
                'disk_ms': disk_time,
                'metal_ms': metal_time,
                'kernel_ms': kernel_time,
                'total_ms': total_time,
                'size_mb': total_size / 1e6,
            })

    avg = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}

    print(f"\nLoading {n_experts} experts:")
    print(f"  Disk I/O:     {avg['disk_ms']:.2f} ms")
    print(f"  Metal create: {avg['metal_ms']:.2f} ms")
    print(f"  Kernel exec:  {avg['kernel_ms']:.2f} ms")
    print(f"  Total:        {avg['total_ms']:.2f} ms")
    print(f"  Data size:    {avg['size_mb']:.2f} MB")
    print(f"  Throughput:   {avg['size_mb'] / (avg['total_ms'] / 1000) / 1000:.2f} GB/s")

    return avg


def main():
    parser = argparse.ArgumentParser(description="Phase 1 benchmarks")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--verify", action="store_true", help="Verify expert extraction correctness")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("Phase 1 Benchmark Suite")
    print("=" * 60)

    model = GGUFModel(model_path)
    print(f"\n{model.summary()}")

    if args.verify:
        print("\n" + "=" * 60)
        print("Verifying Expert Extraction")
        print("=" * 60)
        verify_expert_extraction(model, layer=0, expert_id=0)
        verify_expert_extraction(model, layer=0, expert_id=model.n_experts - 1)
        if model.n_layers > 1:
            verify_expert_extraction(model, layer=model.n_layers - 1, expert_id=0)

    # Run benchmarks
    n_trials = 10 if args.quick else 50

    results = {}
    results['q1_single'] = benchmark_single_expert_read(model, n_trials)
    results['q2_batch'] = benchmark_batch_vs_individual(model)
    results['q3_madvise'] = benchmark_madvise(model, n_trials // 2)
    results['q4_metal'] = benchmark_metal_overhead()
    results['e2e'] = run_end_to_end(model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    q1_pass = results['q1_single']['pread']['avg_gbps'] >= 2.0
    print(f"\nQ1: 2+ GB/s single expert read?  {'✓ YES' if q1_pass else '✗ NO'} ({results['q1_single']['pread']['avg_gbps']:.2f} GB/s)")

    if 1 in results['q2_batch'] and 10 in results['q2_batch']:
        batch_improvement = results['q2_batch'][1]['per_expert_ms'] - results['q2_batch'][10]['per_expert_ms']
        print(f"Q2: Batching helps?              {'✓ YES' if batch_improvement > 0 else '✗ NO'} ({batch_improvement:.3f} ms saved per expert)")

    madvise_improvement = results['q3_madvise']['improvement_pct']
    print(f"Q3: madvise helps?               {'✓ YES' if madvise_improvement > 5 else '✗ MINIMAL'} ({madvise_improvement:.1f}% improvement)")

    metal_overhead = results['q4_metal'].get('1 MB', {}).get('creation_ms', 0)
    print(f"Q4: Metal overhead?              {metal_overhead:.3f} ms per 1MB buffer")

    print(f"\nEnd-to-end (10 experts): {results['e2e']['total_ms']:.2f} ms")

    # Viability calculation
    e2e_time_per_expert = results['e2e']['total_ms'] / 10
    estimated_per_token_ms = e2e_time_per_expert * 480  # 480 experts per token
    estimated_toks = 1000 / estimated_per_token_ms if estimated_per_token_ms > 0 else 0

    print(f"\nViability estimate (no caching):")
    print(f"  Time per expert: {e2e_time_per_expert:.3f} ms")
    print(f"  480 experts/token → {estimated_per_token_ms:.0f} ms/token → {estimated_toks:.2f} tok/s")

    # With caching assumption
    cache_hit_rate = 0.5
    effective_loads = 480 * (1 - cache_hit_rate)
    cached_toks = 1000 / (effective_loads * e2e_time_per_expert)
    print(f"  With 50% cache hits → {effective_loads:.0f} loads/token → {cached_toks:.2f} tok/s")


if __name__ == "__main__":
    main()
