#!/usr/bin/env python3
"""
Self-contained I/O tracing benchmark.
Captures fs_usage output alongside benchmark timing for correlation.

Run with: sudo uv run python benchmarks/io_trace.py <model.gguf>
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gguf_parser import GGUFModel


OUTPUT_DIR = Path(__file__).parent.parent / "trace_output"


def log(msg: str, f=None):
    """Print with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line)
    if f:
        f.write(line + "\n")
        f.flush()


def run_trace(model_path: Path, n_trials: int = 5):
    """Run benchmark with fs_usage tracing."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    trace_file = OUTPUT_DIR / "fs_usage.txt"
    bench_file = OUTPUT_DIR / "benchmark.txt"
    summary_file = OUTPUT_DIR / "summary.txt"

    model = GGUFModel(model_path)
    expert_info = model.get_expert_offset(0, 0, 'down')

    print(f"\n{'='*60}")
    print("I/O TRACE BENCHMARK")
    print(f"{'='*60}")
    print(f"Model: {model_path.name}")
    print(f"Expert size: {expert_info.size / 1e6:.3f} MB")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Start fs_usage in background
    print("Starting fs_usage trace...")
    fs_proc = subprocess.Popen(
        ["fs_usage", "-f", "filesys", "-w"],
        stdout=open(trace_file, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)  # Let fs_usage start

    results = []

    with open(bench_file, "w") as f:
        log("Benchmark started", f)
        log(f"Model: {model_path}", f)
        log(f"Expert size: {expert_info.size} bytes", f)
        log("", f)

        with open(model_path, "rb") as model_file:
            fd = model_file.fileno()

            for trial in range(n_trials):
                # Pick an expert
                layer = trial % model.n_layers
                expert_id = (trial * 37) % model.n_experts
                info = model.get_expert_offset(layer, expert_id, 'down')

                log(f"--- Trial {trial + 1}/{n_trials} ---", f)
                log(f"Layer={layer}, Expert={expert_id}, Offset={info.offset}, Size={info.size}", f)

                # Drop caches
                log("Dropping caches (sudo purge)...", f)
                purge_start = time.perf_counter()
                subprocess.run(["purge"], capture_output=True)
                purge_time = time.perf_counter() - purge_start
                log(f"Purge completed in {purge_time*1000:.1f} ms", f)

                # Small delay for purge to take effect
                time.sleep(0.2)

                # Mark the read start clearly
                log(f">>> READ START (offset={info.offset}) <<<", f)
                read_start = time.perf_counter()

                # Do the actual read
                data = os.pread(fd, info.size, info.offset)

                read_end = time.perf_counter()
                log(f">>> READ END <<<", f)

                read_time_ms = (read_end - read_start) * 1000
                throughput = info.size / (read_end - read_start) / 1e9

                log(f"Read {len(data)} bytes in {read_time_ms:.3f} ms = {throughput:.2f} GB/s", f)
                log("", f)

                results.append({
                    'trial': trial,
                    'layer': layer,
                    'expert': expert_id,
                    'size': info.size,
                    'time_ms': read_time_ms,
                    'throughput_gbps': throughput,
                })

                # Small delay between trials
                time.sleep(0.3)

        log("Benchmark completed", f)

    # Stop fs_usage
    print("Stopping fs_usage trace...")
    fs_proc.terminate()
    try:
        fs_proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        fs_proc.kill()

    # Write summary
    with open(summary_file, "w") as f:
        f.write("I/O TRACE BENCHMARK SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Model: {model_path.name}\n")
        f.write(f"Expert size: {expert_info.size / 1e6:.3f} MB\n")
        f.write(f"Trials: {n_trials}\n\n")

        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"Trial {r['trial']+1}: Layer {r['layer']}, Expert {r['expert']}: "
                   f"{r['time_ms']:.3f} ms, {r['throughput_gbps']:.2f} GB/s\n")

        f.write("-" * 60 + "\n")

        avg_time = sum(r['time_ms'] for r in results) / len(results)
        avg_throughput = sum(r['throughput_gbps'] for r in results) / len(results)

        f.write(f"\nAverage: {avg_time:.3f} ms, {avg_throughput:.2f} GB/s\n")

        # Viability estimate
        f.write(f"\nViability estimate:\n")
        f.write(f"  Time per expert: {avg_time:.3f} ms\n")
        f.write(f"  480 experts/token: {avg_time * 480:.1f} ms = {1000 / (avg_time * 480):.2f} tok/s\n")
        f.write(f"  With 50% cache: {1000 / (avg_time * 240):.2f} tok/s\n")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Average read time: {avg_time:.3f} ms")
    print(f"Average throughput: {avg_throughput:.2f} GB/s")
    print(f"\nOutput files:")
    print(f"  {trace_file}")
    print(f"  {bench_file}")
    print(f"  {summary_file}")
    print(f"\nTo analyze, run:")
    print(f"  grep 'Qwen3' {trace_file} | head -100")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to GGUF model")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        sys.exit(1)

    # Check if running as root
    if os.geteuid() != 0:
        print("Error: Must run with sudo for cache control and fs_usage")
        print(f"Run: sudo uv run python {__file__} {args.model}")
        sys.exit(1)

    run_trace(model_path, args.trials)


if __name__ == "__main__":
    main()
