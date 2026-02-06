"""Micro-benchmark: just the rebuild step, isolated from discovery and eval.

Creates a synthetic stacked tensor matching the actual cache shape,
then measures the three rebuild approaches WITHOUT per-layer eval,
to see the raw cost of each method.
"""

import time
import numpy as np
import mlx.core as mx

CAPACITY = 256
NUM_SWAPS = 30  # typical per-layer swap count
ITERATIONS = 10

# Actual tensor shapes from the model (gate_proj/up_proj)
WEIGHT_SHAPE = (CAPACITY, 512, 256)  # uint32
SCALE_SHAPE = (CAPACITY, 512, 32)    # bfloat16
BIAS_SHAPE = (CAPACITY, 512, 32)     # bfloat16

def make_tensors():
    w = mx.zeros(WEIGHT_SHAPE, dtype=mx.uint32)
    s = mx.zeros(SCALE_SHAPE, dtype=mx.bfloat16)
    b = mx.zeros(BIAS_SHAPE, dtype=mx.bfloat16)
    mx.eval(w, s, b)
    return w, s, b

def make_new_data():
    new_w = mx.ones((NUM_SWAPS,) + WEIGHT_SHAPE[1:], dtype=mx.uint32)
    new_s = mx.ones((NUM_SWAPS,) + SCALE_SHAPE[1:], dtype=mx.bfloat16)
    new_b = mx.ones((NUM_SWAPS,) + BIAS_SHAPE[1:], dtype=mx.bfloat16)
    mx.eval(new_w, new_s, new_b)
    return new_w, new_s, new_b

slots = list(range(0, NUM_SWAPS * 8, 8))[:NUM_SWAPS]  # spread-out slots
slot_indices_mx = mx.array(slots)


def bench_original(w, s, b, new_w, new_s, new_b):
    """List decompose + mx.stack (current method)."""
    w_list = [w[j] for j in range(CAPACITY)]
    s_list = [s[j] for j in range(CAPACITY)]
    b_list = [b[j] for j in range(CAPACITY)]

    for j, slot in enumerate(slots):
        w_list[slot] = new_w[j]
        s_list[slot] = new_s[j]
        b_list[slot] = new_b[j]

    w_out = mx.stack(w_list)
    s_out = mx.stack(s_list)
    b_out = mx.stack(b_list)
    return w_out, s_out, b_out


def bench_scatter(w, s, b, new_w, new_s, new_b):
    """MLX native scatter via __setitem__."""
    w[slot_indices_mx] = new_w
    s[slot_indices_mx] = new_s
    b[slot_indices_mx] = new_b
    return w, s, b


def bench_numpy(w, s, b, new_w, new_s, new_b):
    """Numpy round-trip scatter."""
    w_np = np.array(w)
    w_np[slots] = np.array(new_w)
    w_out = mx.array(w_np)

    s_np = np.frombuffer(memoryview(s), dtype=np.uint16).reshape(s.shape).copy()
    s_np[slots] = np.frombuffer(memoryview(new_s), dtype=np.uint16).reshape(new_s.shape)
    s_out = mx.array(s_np).view(mx.bfloat16)

    b_np = np.frombuffer(memoryview(b), dtype=np.uint16).reshape(b.shape).copy()
    b_np[slots] = np.frombuffer(memoryview(new_b), dtype=np.uint16).reshape(new_b.shape)
    b_out = mx.array(b_np).view(mx.bfloat16)
    return w_out, s_out, b_out


def run_bench(name, fn, with_eval):
    times = []
    for _ in range(ITERATIONS):
        w, s, b = make_tensors()
        new_w, new_s, new_b = make_new_data()

        t0 = time.perf_counter()
        w_out, s_out, b_out = fn(w, s, b, new_w, new_s, new_b)
        if with_eval:
            mx.eval(w_out, s_out, b_out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    mn = min(times)
    return avg, mn


print(f"Benchmark: {CAPACITY} capacity, {NUM_SWAPS} swaps, {ITERATIONS} iterations")
print(f"Weight: {WEIGHT_SHAPE} uint32 ({np.prod(WEIGHT_SHAPE) * 4 / 1e6:.1f} MB)")
print(f"Scale:  {SCALE_SHAPE} bfloat16 ({np.prod(SCALE_SHAPE) * 2 / 1e6:.1f} MB)")
print()

# Warmup
make_tensors()
make_new_data()

print("WITHOUT mx.eval (graph construction only):")
print(f"  {'Method':<20} {'Avg (ms)':>12} {'Min (ms)':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")
for name, fn in [("Original (stack)", bench_original),
                 ("Scatter (setitem)", bench_scatter),
                 ("Numpy (round-trip)", bench_numpy)]:
    avg, mn = run_bench(name, fn, with_eval=False)
    print(f"  {name:<20} {avg*1000:>12.1f} {mn*1000:>12.1f}")

print()
print("WITH mx.eval (full materialization):")
print(f"  {'Method':<20} {'Avg (ms)':>12} {'Min (ms)':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")
for name, fn in [("Original (stack)", bench_original),
                 ("Scatter (setitem)", bench_scatter),
                 ("Numpy (round-trip)", bench_numpy)]:
    avg, mn = run_bench(name, fn, with_eval=True)
    print(f"  {name:<20} {avg*1000:>12.1f} {mn*1000:>12.1f}")

# Also measure eval alone on a pre-built tensor
print()
print("Pure eval cost (pre-built tensor):")
w, s, b = make_tensors()
times = []
for _ in range(ITERATIONS):
    t0 = time.perf_counter()
    mx.eval(w, s, b)
    t1 = time.perf_counter()
    times.append(t1 - t0)
avg = sum(times) / len(times)
print(f"  Eval of 3 tensors: {avg*1000:.1f} ms avg")
