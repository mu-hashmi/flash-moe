"""Benchmark the per-layer eval cost that dominates delta_warmup rebuild."""

import time
import numpy as np
import mlx.core as mx

CAPACITY = 256

# Actual shapes from the model
SHAPES_DTYPES = [
    # gate_proj
    ((CAPACITY, 512, 256), mx.uint32),    # weight
    ((CAPACITY, 512, 32), mx.bfloat16),   # scales
    ((CAPACITY, 512, 32), mx.bfloat16),   # biases
    # up_proj
    ((CAPACITY, 512, 256), mx.uint32),
    ((CAPACITY, 512, 32), mx.bfloat16),
    ((CAPACITY, 512, 32), mx.bfloat16),
    # down_proj
    ((CAPACITY, 2048, 64), mx.uint32),
    ((CAPACITY, 2048, 8), mx.bfloat16),
    ((CAPACITY, 2048, 8), mx.bfloat16),
]

slots = list(range(30))
slot_indices = mx.array(slots)
ITERS = 5

# Scenario 1: eval of scatter results (graph contains scatter op)
print("Scenario 1: eval after scatter (single-ref donation path)")
times = []
for _ in range(ITERS):
    tensors = []
    for shape, dtype in SHAPES_DTYPES:
        t = mx.zeros(shape, dtype=dtype)
        mx.eval(t)
        new_data = mx.ones((30,) + shape[1:], dtype=dtype)
        mx.eval(new_data)
        t[slot_indices] = new_data
        tensors.append(t)

    t0 = time.perf_counter()
    mx.eval(*tensors)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  {(t1 - t0)*1000:.1f} ms")

avg = sum(times) / len(times)
print(f"  Average: {avg*1000:.1f} ms per layer eval (scatter path)")

# Scenario 2: eval of mx.stack results (graph contains stack op)
print("\nScenario 2: eval after mx.stack (current path)")
times = []
for _ in range(ITERS):
    tensors = []
    for shape, dtype in SHAPES_DTYPES:
        t = mx.zeros(shape, dtype=dtype)
        mx.eval(t)
        new_data = mx.ones((30,) + shape[1:], dtype=dtype)
        mx.eval(new_data)
        parts = [t[j] for j in range(CAPACITY)]
        for j, s in enumerate(slots):
            parts[s] = new_data[j]
        stacked = mx.stack(parts)
        tensors.append(stacked)

    t0 = time.perf_counter()
    mx.eval(*tensors)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  {(t1 - t0)*1000:.1f} ms")

avg = sum(times) / len(times)
print(f"  Average: {avg*1000:.1f} ms per layer eval (stack path)")

# Scenario 3: eval of already-evaluated tensors (no-op baseline)
print("\nScenario 3: eval of already-evaluated tensors (baseline)")
tensors = []
for shape, dtype in SHAPES_DTYPES:
    t = mx.zeros(shape, dtype=dtype)
    mx.eval(t)
    tensors.append(t)

times = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    mx.eval(*tensors)
    t1 = time.perf_counter()
    times.append(t1 - t0)
print(f"  Average: {sum(times)/len(times)*1000:.1f} ms (no-op)")

total_bytes = sum(np.prod(s) * (4 if d == mx.uint32 else 2)
                  for s, d in SHAPES_DTYPES)
print(f"\nTotal per-layer data: {total_bytes / 1e6:.1f} MB")
