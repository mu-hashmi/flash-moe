"""Selective expert loading with I/O benchmarking."""

import mmap
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .gguf_parser import ExpertInfo, GGUFModel
except ImportError:
    from gguf_parser import ExpertInfo, GGUFModel


@dataclass
class LoadResult:
    expert_info: ExpertInfo
    data: bytes
    elapsed_ms: float
    throughput_gbps: float


class ExpertLoader:
    """Load individual experts from GGUF file with various strategies."""

    def __init__(self, model: GGUFModel):
        self.model = model
        self.path = model.path
        self._file = None
        self._mmap = None

    def __enter__(self):
        self._file = open(self.path, 'rb')
        return self

    def __exit__(self, *args):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

    def _ensure_mmap(self):
        if self._mmap is None:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def load_expert_seek(self, info: ExpertInfo) -> LoadResult:
        """Load expert using seek + read."""
        self._file.seek(info.offset)

        start = time.perf_counter()
        data = self._file.read(info.size)
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        throughput = info.size / elapsed / 1e9 if elapsed > 0 else 0

        return LoadResult(info, data, elapsed_ms, throughput)

    def load_expert_mmap(self, info: ExpertInfo) -> LoadResult:
        """Load expert using mmap slice."""
        self._ensure_mmap()

        start = time.perf_counter()
        data = self._mmap[info.offset:info.offset + info.size]
        # Force page fault by accessing the data
        _ = bytes(data)
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        throughput = info.size / elapsed / 1e9 if elapsed > 0 else 0

        return LoadResult(info, data, elapsed_ms, throughput)

    def load_expert_pread(self, info: ExpertInfo) -> LoadResult:
        """Load expert using pread (no seek needed)."""
        start = time.perf_counter()
        data = os.pread(self._file.fileno(), info.size, info.offset)
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        throughput = info.size / elapsed / 1e9 if elapsed > 0 else 0

        return LoadResult(info, data, elapsed_ms, throughput)

    def load_experts_batch(
        self, infos: list[ExpertInfo], method: str = 'pread'
    ) -> tuple[list[LoadResult], float, float]:
        """
        Load multiple experts and return individual + aggregate stats.

        Returns: (results, total_ms, aggregate_throughput_gbps)
        """
        results = []
        total_size = sum(i.size for i in infos)

        load_fn = {
            'seek': self.load_expert_seek,
            'mmap': self.load_expert_mmap,
            'pread': self.load_expert_pread,
        }[method]

        start = time.perf_counter()
        for info in infos:
            results.append(load_fn(info))
        total_elapsed = time.perf_counter() - start

        total_ms = total_elapsed * 1000
        throughput = total_size / total_elapsed / 1e9 if total_elapsed > 0 else 0

        return results, total_ms, throughput

    def load_experts_sequential(
        self, infos: list[ExpertInfo]
    ) -> tuple[bytes, float, float]:
        """
        Load multiple experts in one contiguous read (if they're adjacent).

        Returns: (data, total_ms, throughput_gbps)
        """
        if not infos:
            return b'', 0, 0

        # Sort by offset
        sorted_infos = sorted(infos, key=lambda x: x.offset)

        # Check if contiguous
        min_offset = sorted_infos[0].offset
        max_end = sorted_infos[-1].offset + sorted_infos[-1].size
        total_size = sum(i.size for i in sorted_infos)
        span_size = max_end - min_offset

        # Allow some gap (up to 10% overhead)
        if span_size > total_size * 1.1:
            raise ValueError(
                f"Experts not contiguous: span={span_size}, data={total_size}"
            )

        start = time.perf_counter()
        data = os.pread(self._file.fileno(), span_size, min_offset)
        elapsed = time.perf_counter() - start

        return data, elapsed * 1000, span_size / elapsed / 1e9


def benchmark_io(
    model: GGUFModel,
    n_trials: int = 10,
    experts_per_batch: int = 10,
) -> dict:
    """Run I/O benchmarks and return results."""
    results = {
        'single_expert': {},
        'batch_experts': {},
        'sequential_read': {},
    }

    with ExpertLoader(model) as loader:
        # Warm up filesystem cache
        info = model.get_expert_offset(0, 0, 'down')
        _ = loader.load_expert_pread(info)

        # Drop caches (requires sudo, skip if not available)
        try:
            os.system('sync; sudo purge 2>/dev/null')
            time.sleep(0.5)
        except Exception:
            pass

        # Benchmark single expert reads with different methods
        for method in ['seek', 'mmap', 'pread']:
            times = []
            throughputs = []

            for trial in range(n_trials):
                # Random layer and expert to avoid caching bias
                layer = trial % model.n_layers
                expert = (trial * 37) % model.n_experts  # Pseudo-random

                info = model.get_expert_offset(layer, expert, 'down')
                result = getattr(loader, f'load_expert_{method}')(info)
                times.append(result.elapsed_ms)
                throughputs.append(result.throughput_gbps)

            results['single_expert'][method] = {
                'avg_ms': np.mean(times),
                'std_ms': np.std(times),
                'avg_gbps': np.mean(throughputs),
                'expert_size_mb': info.size / 1e6,
            }

        # Benchmark batch reads (10 experts = one layer's active set)
        for method in ['seek', 'mmap', 'pread']:
            times = []
            throughputs = []

            for trial in range(n_trials):
                layer = trial % model.n_layers
                expert_ids = [(trial * 37 + i * 53) % model.n_experts for i in range(experts_per_batch)]
                infos = [model.get_expert_offset(layer, eid, 'down') for eid in expert_ids]

                _, total_ms, throughput = loader.load_experts_batch(infos, method)
                times.append(total_ms)
                throughputs.append(throughput)

            results['batch_experts'][method] = {
                'avg_ms': np.mean(times),
                'std_ms': np.std(times),
                'avg_gbps': np.mean(throughputs),
                'n_experts': experts_per_batch,
            }

        # Benchmark sequential read of adjacent experts
        times = []
        throughputs = []

        for trial in range(n_trials):
            layer = trial % model.n_layers
            # Adjacent experts
            start_expert = (trial * 37) % (model.n_experts - experts_per_batch)
            expert_ids = list(range(start_expert, start_expert + experts_per_batch))
            infos = [model.get_expert_offset(layer, eid, 'down') for eid in expert_ids]

            try:
                _, total_ms, throughput = loader.load_experts_sequential(infos)
                times.append(total_ms)
                throughputs.append(throughput)
            except ValueError:
                # Not contiguous, skip
                pass

        if times:
            results['sequential_read'] = {
                'avg_ms': np.mean(times),
                'std_ms': np.std(times),
                'avg_gbps': np.mean(throughputs),
                'n_experts': experts_per_batch,
            }

    return results


def verify_expert_extraction(model: GGUFModel, layer: int = 0, expert_id: int = 0):
    """
    Verify expert extraction by comparing to full tensor load.

    Loads the full merged tensor and extracts one expert,
    then compares to our calculated offset extraction.
    """
    from gguf import GGUFReader

    reader = GGUFReader(str(model.path))

    # Find the tensor
    tensor_name = f"blk.{layer}.ffn_down_exps.weight"
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break

    if tensor is None:
        raise ValueError(f"Tensor {tensor_name} not found")

    # Get our calculated expert info
    info = model.get_expert_offset(layer, expert_id, 'down')

    # Load via our method
    with ExpertLoader(model) as loader:
        result = loader.load_expert_pread(info)
        our_data = result.data

    # Load full tensor and extract expert slice
    full_data = tensor.data
    n_experts = tensor.shape[-1]
    expert_size = len(full_data) // n_experts
    expected_data = bytes(full_data[expert_id * expert_size:(expert_id + 1) * expert_size])

    # Compare
    match = our_data == expected_data
    print(f"Expert extraction verification:")
    print(f"  Layer: {layer}, Expert: {expert_id}")
    print(f"  Our size: {len(our_data)}, Expected: {len(expected_data)}")
    print(f"  Match: {match}")

    if not match and len(our_data) == len(expected_data):
        # Find first mismatch
        for i, (a, b) in enumerate(zip(our_data, expected_data)):
            if a != b:
                print(f"  First mismatch at byte {i}: {a} vs {b}")
                break

    return match


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python expert_loader.py <model.gguf> [--verify] [--bench]")
        sys.exit(1)

    model = GGUFModel(sys.argv[1])
    print(model.summary())
    print()

    if '--verify' in sys.argv:
        print("Verifying expert extraction...")
        verify_expert_extraction(model)
        print()

    if '--bench' in sys.argv or '--verify' not in sys.argv:
        print("Running I/O benchmarks...")
        results = benchmark_io(model)

        print("\n=== Single Expert Read ===")
        for method, stats in results['single_expert'].items():
            print(f"  {method:6}: {stats['avg_ms']:.2f} ± {stats['std_ms']:.2f} ms, "
                  f"{stats['avg_gbps']:.2f} GB/s ({stats['expert_size_mb']:.2f} MB)")

        print("\n=== Batch Read (10 experts) ===")
        for method, stats in results['batch_experts'].items():
            print(f"  {method:6}: {stats['avg_ms']:.2f} ± {stats['std_ms']:.2f} ms, "
                  f"{stats['avg_gbps']:.2f} GB/s")

        if results['sequential_read']:
            print("\n=== Sequential Read (adjacent experts) ===")
            stats = results['sequential_read']
            print(f"  {stats['avg_ms']:.2f} ± {stats['std_ms']:.2f} ms, "
                  f"{stats['avg_gbps']:.2f} GB/s")
