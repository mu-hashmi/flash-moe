#!/usr/bin/env python3
"""
End-to-end viability calculation for flash-moe.

Combines measurements from:
- madvise_bench.py (prefetch latency, lead time)
- cache_verify.py (warm cache speed)
- pipeline_proto.py (pipeline overhead)
- analyze_expert_log.py (cache hit rates)

Produces go/no-go recommendation.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResults:
    """Results from benchmark runs."""
    # Cold read (SSD) performance
    cold_read_gbps: float = 3.0  # Typical Apple SSD
    cold_read_latency_ms: float = 5.0  # Per expert

    # Warm read (cached) performance
    warm_read_gbps: float = 15.0
    warm_read_latency_ms: float = 0.05

    # madvise effectiveness
    madvise_effective: bool = True
    madvise_lead_time_ms: float = 10.0  # Time needed for prefetch

    # Pipeline efficiency
    pipeline_overhead: float = 1.2  # 1.0 = perfect overlap

    # Expert reuse (from llama.cpp logs)
    cache_hit_rate: float = 0.3  # 30% default assumption
    working_set_size: int = 64  # Experts to keep cached


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen3-Coder-Next-UD-IQ1_S"
    n_layers: int = 48
    n_experts: int = 512
    experts_per_token: int = 8  # Typical for MoE models
    expert_size_mb: float = 0.344  # Per expert tensor (down only) - measured
    total_expert_size_gb: float = 21.0  # Full model size


@dataclass
class TargetConfig:
    """Performance targets."""
    min_tok_per_sec: float = 4.0
    target_tok_per_sec: float = 10.0
    max_memory_gb: float = 8.0  # For expert caching


def calculate_viability(
    bench: BenchmarkResults,
    model: ModelConfig,
    target: TargetConfig,
) -> dict:
    """
    Calculate end-to-end viability.

    Returns detailed breakdown and go/no-go recommendation.
    """
    results = {}

    # Total experts needed per token
    total_experts_per_token = model.n_layers * model.experts_per_token
    results['total_experts_per_token'] = total_experts_per_token

    # Data loaded per token (assuming only down tensor, not gate/up)
    data_per_token_mb = total_experts_per_token * model.expert_size_mb
    results['data_per_token_mb'] = data_per_token_mb

    # === Scenario 1: No caching (worst case) ===
    # Time = size / throughput
    # expert_size_mb MB / cold_read_gbps GB/s = expert_size_mb / (cold_read_gbps * 1000) seconds
    # = expert_size_mb / cold_read_gbps ms
    cold_time_per_expert = model.expert_size_mb / bench.cold_read_gbps  # ms
    cold_time_per_token = total_experts_per_token * cold_time_per_expert
    cold_tok_per_sec = 1000 / cold_time_per_token

    results['no_cache'] = {
        'time_per_expert_ms': cold_time_per_expert,
        'time_per_token_ms': cold_time_per_token,
        'tok_per_sec': cold_tok_per_sec,
    }

    # === Scenario 2: With caching (based on measured hit rate) ===
    cache_hit_rate = bench.cache_hit_rate
    n_cold_experts = total_experts_per_token * (1 - cache_hit_rate)
    n_warm_experts = total_experts_per_token * cache_hit_rate

    warm_time_per_expert = model.expert_size_mb / bench.warm_read_gbps  # ms
    cached_time = n_cold_experts * cold_time_per_expert + n_warm_experts * warm_time_per_expert
    cached_tok_per_sec = 1000 / cached_time

    results['with_cache'] = {
        'hit_rate': cache_hit_rate,
        'cold_experts_per_token': n_cold_experts,
        'warm_experts_per_token': n_warm_experts,
        'time_per_token_ms': cached_time,
        'tok_per_sec': cached_tok_per_sec,
    }

    # === Scenario 3: With prefetch pipeline ===
    if bench.madvise_effective:
        # Assume prefetch hides most cold read latency
        # Effective time = max(compute_time, cold_read_time / overlap_factor)
        compute_time_per_token = 5.0  # Assume ~5ms GPU compute per token
        io_time = cached_time * bench.pipeline_overhead

        # Pipeline can overlap I/O with compute
        pipelined_time = max(compute_time_per_token, io_time)
        pipelined_tok_per_sec = 1000 / pipelined_time

        results['with_pipeline'] = {
            'compute_time_ms': compute_time_per_token,
            'io_time_ms': io_time,
            'pipeline_overhead': bench.pipeline_overhead,
            'time_per_token_ms': pipelined_time,
            'tok_per_sec': pipelined_tok_per_sec,
        }
    else:
        results['with_pipeline'] = None

    # === Memory requirements ===
    cache_memory_gb = bench.working_set_size * model.expert_size_mb / 1000 * model.n_layers
    results['memory'] = {
        'working_set_per_layer': bench.working_set_size,
        'total_cache_gb': cache_memory_gb,
        'within_budget': cache_memory_gb <= target.max_memory_gb,
    }

    # === Break-even analysis ===
    # What cache hit rate needed for target tok/s?
    required_time_per_token = 1000 / target.min_tok_per_sec
    # time = n_cold * cold_time + n_warm * warm_time
    # time = (1-hr) * n * cold + hr * n * warm
    # required_time = n * ((1-hr) * cold + hr * warm)
    # hr = (n * cold - required_time) / (n * (cold - warm))
    n = total_experts_per_token
    break_even_hr = (n * cold_time_per_expert - required_time_per_token) / \
                    (n * (cold_time_per_expert - warm_time_per_expert))
    break_even_hr = max(0, min(1, break_even_hr))

    results['break_even'] = {
        'target_tok_per_sec': target.min_tok_per_sec,
        'required_cache_hit_rate': break_even_hr,
        'achievable': bench.cache_hit_rate >= break_even_hr,
    }

    # === Go/No-Go Recommendation ===
    best_tok_per_sec = max(
        cached_tok_per_sec,
        results['with_pipeline']['tok_per_sec'] if results['with_pipeline'] else 0,
    )

    go = best_tok_per_sec >= target.min_tok_per_sec
    confidence = 'high' if best_tok_per_sec >= target.target_tok_per_sec else \
                 'medium' if best_tok_per_sec >= target.min_tok_per_sec else 'low'

    results['recommendation'] = {
        'go': go,
        'confidence': confidence,
        'best_tok_per_sec': best_tok_per_sec,
        'target_met': best_tok_per_sec >= target.target_tok_per_sec,
        'min_met': best_tok_per_sec >= target.min_tok_per_sec,
    }

    return results


def print_viability_report(results: dict, bench: BenchmarkResults, model: ModelConfig):
    """Print formatted viability report."""
    print("\n" + "=" * 70)
    print("FLASH-MOE VIABILITY CALCULATION")
    print("=" * 70)

    print(f"\n--- Model: {model.name} ---")
    print(f"Layers: {model.n_layers}")
    print(f"Experts/layer: {model.n_experts}")
    print(f"Active experts/token/layer: {model.experts_per_token}")
    print(f"Expert size: {model.expert_size_mb:.2f} MB")

    print(f"\n--- Measured Parameters ---")
    print(f"Cold read (SSD): {bench.cold_read_gbps:.1f} GB/s")
    print(f"Warm read (cache): {bench.warm_read_gbps:.1f} GB/s")
    print(f"madvise effective: {bench.madvise_effective}")
    print(f"Cache hit rate: {bench.cache_hit_rate*100:.0f}%")
    print(f"Pipeline overhead: {bench.pipeline_overhead:.2f}x")

    print(f"\n--- Scenario Analysis ---")

    print(f"\n1. No Caching (Worst Case):")
    nc = results['no_cache']
    print(f"   Time per expert: {nc['time_per_expert_ms']:.2f} ms")
    print(f"   Time per token: {nc['time_per_token_ms']:.1f} ms")
    print(f"   Throughput: {nc['tok_per_sec']:.2f} tok/s")

    print(f"\n2. With {bench.cache_hit_rate*100:.0f}% Cache Hit Rate:")
    wc = results['with_cache']
    print(f"   Cold experts/token: {wc['cold_experts_per_token']:.0f}")
    print(f"   Warm experts/token: {wc['warm_experts_per_token']:.0f}")
    print(f"   Time per token: {wc['time_per_token_ms']:.1f} ms")
    print(f"   Throughput: {wc['tok_per_sec']:.2f} tok/s")

    if results['with_pipeline']:
        print(f"\n3. With Prefetch Pipeline:")
        wp = results['with_pipeline']
        print(f"   Compute time: {wp['compute_time_ms']:.1f} ms")
        print(f"   I/O time (with pipeline): {wp['io_time_ms']:.1f} ms")
        print(f"   Effective time: {wp['time_per_token_ms']:.1f} ms")
        print(f"   Throughput: {wp['tok_per_sec']:.2f} tok/s")

    print(f"\n--- Memory Requirements ---")
    mem = results['memory']
    print(f"Working set per layer: {mem['working_set_per_layer']} experts")
    print(f"Total cache needed: {mem['total_cache_gb']:.1f} GB")
    print(f"Within 8GB budget: {'YES' if mem['within_budget'] else 'NO'}")

    print(f"\n--- Break-Even Analysis ---")
    be = results['break_even']
    print(f"For 4 tok/s minimum:")
    print(f"   Required cache hit rate: {be['required_cache_hit_rate']*100:.0f}%")
    print(f"   Achievable: {'YES' if be['achievable'] else 'NO'}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    rec = results['recommendation']
    status = "GO" if rec['go'] else "NO-GO"
    print(f"\n  Decision: {status}")
    print(f"  Confidence: {rec['confidence'].upper()}")
    print(f"  Best projected throughput: {rec['best_tok_per_sec']:.1f} tok/s")
    print(f"  Minimum target (4 tok/s): {'MET' if rec['min_met'] else 'NOT MET'}")
    print(f"  Stretch target (10 tok/s): {'MET' if rec['target_met'] else 'NOT MET'}")

    if rec['go']:
        print("\n  Proceed to Phase 2 implementation.")
    else:
        print("\n  Revisit assumptions or consider alternative approaches.")


def load_benchmark_results(results_dir: Path) -> BenchmarkResults:
    """
    Load benchmark results from JSON files in results directory.
    Falls back to defaults if files not found.
    """
    bench = BenchmarkResults()

    # Try loading from JSON files
    madvise_file = results_dir / "madvise_results.json"
    cache_file = results_dir / "cache_results.json"
    pipeline_file = results_dir / "pipeline_results.json"
    expert_file = results_dir / "expert_analysis.json"

    if madvise_file.exists():
        with open(madvise_file) as f:
            data = json.load(f)
            bench.madvise_effective = data.get('effective', True)
            bench.madvise_lead_time_ms = data.get('lead_time_ms', 10.0)

    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            bench.warm_read_gbps = data.get('warm_gbps', 15.0)
            bench.cold_read_gbps = data.get('cold_gbps', 3.0)

    if pipeline_file.exists():
        with open(pipeline_file) as f:
            data = json.load(f)
            bench.pipeline_overhead = data.get('overhead', 1.2)

    if expert_file.exists():
        with open(expert_file) as f:
            data = json.load(f)
            bench.cache_hit_rate = data.get('hit_rate', 0.3)
            bench.working_set_size = data.get('working_set', 64)

    return bench


def interactive_mode():
    """Run interactive viability calculation with user inputs."""
    print("=" * 70)
    print("FLASH-MOE VIABILITY CALCULATOR (Interactive)")
    print("=" * 70)
    print("\nEnter benchmark results (press Enter for defaults):\n")

    bench = BenchmarkResults()

    try:
        val = input(f"Cold read throughput (GB/s) [{bench.cold_read_gbps}]: ")
        if val:
            bench.cold_read_gbps = float(val)

        val = input(f"Warm read throughput (GB/s) [{bench.warm_read_gbps}]: ")
        if val:
            bench.warm_read_gbps = float(val)

        val = input(f"Cache hit rate (0-1) [{bench.cache_hit_rate}]: ")
        if val:
            bench.cache_hit_rate = float(val)

        val = input(f"Pipeline overhead multiplier [{bench.pipeline_overhead}]: ")
        if val:
            bench.pipeline_overhead = float(val)

        val = input(f"madvise effective (true/false) [{bench.madvise_effective}]: ")
        if val:
            bench.madvise_effective = val.lower() == 'true'

    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    model = ModelConfig()
    target = TargetConfig()

    results = calculate_viability(bench, model, target)
    print_viability_report(results, bench, model)


def main():
    parser = argparse.ArgumentParser(description="Calculate flash-moe viability")
    parser.add_argument("--results-dir", type=Path, help="Directory with benchmark JSON files")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    # Override parameters
    parser.add_argument("--cold-gbps", type=float, help="Cold read throughput")
    parser.add_argument("--warm-gbps", type=float, help="Warm read throughput")
    parser.add_argument("--cache-hit-rate", type=float, help="Cache hit rate (0-1)")
    parser.add_argument("--pipeline-overhead", type=float, help="Pipeline overhead")
    parser.add_argument("--madvise-effective", type=bool, help="madvise works")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    # Load or use defaults
    if args.results_dir and args.results_dir.exists():
        bench = load_benchmark_results(args.results_dir)
    else:
        bench = BenchmarkResults()

    # Apply overrides
    if args.cold_gbps:
        bench.cold_read_gbps = args.cold_gbps
    if args.warm_gbps:
        bench.warm_read_gbps = args.warm_gbps
    if args.cache_hit_rate is not None:
        bench.cache_hit_rate = args.cache_hit_rate
    if args.pipeline_overhead:
        bench.pipeline_overhead = args.pipeline_overhead
    if args.madvise_effective is not None:
        bench.madvise_effective = args.madvise_effective

    model = ModelConfig()
    target = TargetConfig()

    results = calculate_viability(bench, model, target)
    print_viability_report(results, bench, model)

    # Exit code based on go/no-go
    sys.exit(0 if results['recommendation']['go'] else 1)


if __name__ == "__main__":
    main()
