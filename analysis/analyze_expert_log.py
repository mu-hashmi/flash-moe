#!/usr/bin/env python3
"""
Analyze expert selection logs from instrumented llama.cpp.

Expected log format (CSV):
token_id,layer_id,expert_0,expert_1,...,expert_k

Example:
0,0,42,156,301,498,12,89,200,77
0,1,12,89,455,301,42,156,77,200
1,0,42,156,288,500,99,12,89,77
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_expert_log(log_path: Path) -> list[dict]:
    """
    Parse expert selection log file.
    Returns list of records: {token_id, layer_id, experts: [int, ...]}
    """
    records = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) < 3:
                continue

            try:
                token_id = int(parts[0])
                layer_id = int(parts[1])
                experts = [int(x) for x in parts[2:]]

                records.append({
                    'token_id': token_id,
                    'layer_id': layer_id,
                    'experts': experts,
                })
            except ValueError:
                continue

    return records


def analyze_expert_frequency(records: list[dict]) -> dict:
    """Analyze expert usage frequency per layer."""
    layer_expert_counts = defaultdict(Counter)

    for record in records:
        layer = record['layer_id']
        for expert in record['experts']:
            layer_expert_counts[layer][expert] += 1

    results = {}
    for layer, counter in sorted(layer_expert_counts.items()):
        total = sum(counter.values())
        n_experts = len(counter)
        top_10 = counter.most_common(10)

        # Calculate entropy (higher = more uniform)
        probs = np.array([c / total for c in counter.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(n_experts) if n_experts > 0 else 0

        results[layer] = {
            'total_activations': total,
            'unique_experts': n_experts,
            'top_10_experts': top_10,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'entropy_ratio': entropy / max_entropy if max_entropy > 0 else 0,
        }

    return results


def analyze_token_to_token_overlap(records: list[dict]) -> dict:
    """Analyze how often consecutive tokens use the same experts."""
    by_layer = defaultdict(list)
    for record in records:
        by_layer[record['layer_id']].append(
            (record['token_id'], set(record['experts']))
        )

    results = {}
    for layer, token_experts in sorted(by_layer.items()):
        # Sort by token_id
        token_experts.sort(key=lambda x: x[0])

        overlaps = []
        for i in range(1, len(token_experts)):
            prev_experts = token_experts[i - 1][1]
            curr_experts = token_experts[i][1]
            overlap = len(prev_experts & curr_experts)
            n_experts = len(curr_experts)
            overlap_ratio = overlap / n_experts if n_experts > 0 else 0
            overlaps.append(overlap_ratio)

        if overlaps:
            results[layer] = {
                'avg_overlap': np.mean(overlaps),
                'std_overlap': np.std(overlaps),
                'min_overlap': np.min(overlaps),
                'max_overlap': np.max(overlaps),
                'pct_full_overlap': sum(1 for o in overlaps if o == 1.0) / len(overlaps) * 100,
                'pct_any_overlap': sum(1 for o in overlaps if o > 0) / len(overlaps) * 100,
            }

    return results


def simulate_cache_hit_rate(records: list[dict], cache_sizes: list[int]) -> dict:
    """
    Simulate LRU cache hit rates with different cache sizes.
    Cache is per-layer (as each layer has its own experts).
    """
    by_layer = defaultdict(list)
    for record in records:
        layer = record['layer_id']
        by_layer[layer].extend(record['experts'])

    results = {}
    for cache_size in cache_sizes:
        layer_hit_rates = {}

        for layer, expert_sequence in sorted(by_layer.items()):
            # Simple LRU simulation
            cache = []
            hits = 0
            total = 0

            for expert in expert_sequence:
                total += 1
                if expert in cache:
                    hits += 1
                    cache.remove(expert)
                    cache.append(expert)
                else:
                    if len(cache) >= cache_size:
                        cache.pop(0)
                    cache.append(expert)

            layer_hit_rates[layer] = hits / total if total > 0 else 0

        avg_hit_rate = np.mean(list(layer_hit_rates.values()))
        results[cache_size] = {
            'avg_hit_rate': avg_hit_rate,
            'per_layer': layer_hit_rates,
        }

    return results


def find_hot_experts(records: list[dict], threshold: float = 0.1) -> dict:
    """
    Find "hot" experts that are activated in >threshold fraction of tokens.
    """
    layer_expert_counts = defaultdict(Counter)
    layer_token_counts = defaultdict(int)

    for record in records:
        layer = record['layer_id']
        layer_token_counts[layer] += 1
        for expert in record['experts']:
            layer_expert_counts[layer][expert] += 1

    results = {}
    for layer in sorted(layer_expert_counts.keys()):
        n_tokens = layer_token_counts[layer]
        hot_experts = []

        for expert, count in layer_expert_counts[layer].items():
            frequency = count / n_tokens
            if frequency >= threshold:
                hot_experts.append((expert, frequency))

        hot_experts.sort(key=lambda x: -x[1])
        results[layer] = hot_experts

    return results


def calculate_working_set_size(records: list[dict], target_hit_rates: list[float]) -> dict:
    """
    Calculate cache size needed to achieve target hit rates.
    """
    by_layer = defaultdict(list)
    for record in records:
        layer = record['layer_id']
        by_layer[layer].extend(record['experts'])

    results = {}
    for target in target_hit_rates:
        sizes_needed = []

        for layer, expert_sequence in by_layer.items():
            # Binary search for minimum cache size
            low, high = 1, 512
            while low < high:
                mid = (low + high) // 2

                # Simulate LRU
                cache = []
                hits = 0
                total = len(expert_sequence)

                for expert in expert_sequence:
                    if expert in cache:
                        hits += 1
                        cache.remove(expert)
                        cache.append(expert)
                    else:
                        if len(cache) >= mid:
                            cache.pop(0)
                        cache.append(expert)

                hit_rate = hits / total if total > 0 else 0

                if hit_rate >= target:
                    high = mid
                else:
                    low = mid + 1

            sizes_needed.append(low)

        results[target] = {
            'avg_cache_size': np.mean(sizes_needed),
            'max_cache_size': np.max(sizes_needed),
            'per_layer': dict(zip(sorted(by_layer.keys()), sizes_needed)),
        }

    return results


def print_analysis_report(
    records: list[dict],
    frequency_analysis: dict,
    overlap_analysis: dict,
    cache_simulation: dict,
    hot_experts: dict,
    working_set: dict,
):
    """Print comprehensive analysis report."""
    n_tokens = len(set(r['token_id'] for r in records))
    n_layers = len(set(r['layer_id'] for r in records))
    n_experts_per_token = len(records[0]['experts']) if records else 0

    print("\n" + "=" * 70)
    print("EXPERT REUSE ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n--- Dataset Overview ---")
    print(f"Total tokens: {n_tokens}")
    print(f"MoE layers: {n_layers}")
    print(f"Experts per token per layer: {n_experts_per_token}")
    print(f"Total expert activations: {len(records)}")

    print(f"\n--- Expert Frequency Distribution (Sample Layers) ---")
    sample_layers = list(frequency_analysis.keys())[:3]
    for layer in sample_layers:
        data = frequency_analysis[layer]
        print(f"\nLayer {layer}:")
        print(f"  Unique experts activated: {data['unique_experts']}")
        print(f"  Entropy ratio: {data['entropy_ratio']:.2f} (1.0 = uniform)")
        print(f"  Top 5 experts: {data['top_10_experts'][:5]}")

    print(f"\n--- Token-to-Token Overlap ---")
    for layer in sample_layers:
        if layer in overlap_analysis:
            data = overlap_analysis[layer]
            print(f"\nLayer {layer}:")
            print(f"  Average overlap: {data['avg_overlap']*100:.1f}%")
            print(f"  % with any overlap: {data['pct_any_overlap']:.1f}%")
            print(f"  % with full overlap: {data['pct_full_overlap']:.1f}%")

    print(f"\n--- Cache Hit Rate Simulation ---")
    for cache_size, data in sorted(cache_simulation.items()):
        print(f"  Cache size {cache_size:3d}: {data['avg_hit_rate']*100:.1f}% hit rate")

    print(f"\n--- Working Set Size for Target Hit Rates ---")
    for target, data in sorted(working_set.items()):
        print(f"  {target*100:.0f}% hit rate: need cache size ~{data['avg_cache_size']:.0f} (max: {data['max_cache_size']})")

    print(f"\n--- Hot Experts (>10% activation frequency) ---")
    for layer in sample_layers:
        if layer in hot_experts and hot_experts[layer]:
            print(f"\nLayer {layer}: {len(hot_experts[layer])} hot experts")
            for expert, freq in hot_experts[layer][:5]:
                print(f"    Expert {expert}: {freq*100:.1f}%")
        else:
            print(f"\nLayer {layer}: No hot experts (uniform distribution)")


def export_results(output_path: Path, **kwargs):
    """Export analysis results to markdown."""
    with open(output_path, 'w') as f:
        f.write("# Expert Reuse Analysis Results\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")

        if 'overlap_analysis' in kwargs:
            overlaps = kwargs['overlap_analysis']
            avg_overlap = np.mean([d['avg_overlap'] for d in overlaps.values()])
            f.write(f"| Avg token-to-token overlap | {avg_overlap*100:.1f}% |\n")

        if 'cache_simulation' in kwargs:
            sim = kwargs['cache_simulation']
            for size, data in sorted(sim.items()):
                f.write(f"| Cache hit rate (size={size}) | {data['avg_hit_rate']*100:.1f}% |\n")

        if 'working_set' in kwargs:
            ws = kwargs['working_set']
            for target, data in sorted(ws.items()):
                f.write(f"| Cache size for {target*100:.0f}% hits | {data['avg_cache_size']:.0f} |\n")

        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert selection logs")
    parser.add_argument("log_file", help="Path to expert selection log")
    parser.add_argument("--output", "-o", help="Output markdown report path")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    print(f"Parsing {log_path}...")
    records = parse_expert_log(log_path)

    if not records:
        print("Error: No valid records found in log file")
        sys.exit(1)

    print(f"Loaded {len(records)} expert selection records")

    # Run analyses
    print("Analyzing expert frequency...")
    frequency_analysis = analyze_expert_frequency(records)

    print("Analyzing token-to-token overlap...")
    overlap_analysis = analyze_token_to_token_overlap(records)

    print("Simulating cache hit rates...")
    cache_sizes = [8, 16, 32, 64, 128, 256]
    cache_simulation = simulate_cache_hit_rate(records, cache_sizes)

    print("Finding hot experts...")
    hot_experts = find_hot_experts(records, threshold=0.1)

    print("Calculating working set sizes...")
    working_set = calculate_working_set_size(records, [0.5, 0.7, 0.9])

    # Print report
    print_analysis_report(
        records,
        frequency_analysis,
        overlap_analysis,
        cache_simulation,
        hot_experts,
        working_set,
    )

    # Export if requested
    if args.output:
        output_path = Path(args.output)
        export_results(
            output_path,
            overlap_analysis=overlap_analysis,
            cache_simulation=cache_simulation,
            working_set=working_set,
        )
        print(f"\nResults exported to: {output_path}")


if __name__ == "__main__":
    main()
