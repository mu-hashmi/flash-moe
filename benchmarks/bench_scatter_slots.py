"""Benchmark: mx.fast.scatter_slots vs pop()+__setitem__ vs mx.stack rebuild.

Run with mlx-lm venv:
    PATH_REMOVED bench_scatter_slots.py
"""

import time
import mlx.core as mx


def bench_method(name, fn, warmup=3, trials=10):
    for _ in range(warmup):
        fn()
    mx.metal.reset_peak_memory()
    times = []
    for _ in range(trials):
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    peak = mx.metal.get_peak_memory() / 1e9
    print(f"  {name}: {avg*1000:.2f} ms avg ({min(times)*1000:.2f}-{max(times)*1000:.2f}), peak {peak:.2f} GB")
    return avg


def micro_benchmark():
    """Per-layer scatter micro-benchmark at low memory pressure."""
    print("\n=== Micro-benchmark (low memory) ===")
    capacity = 256
    num_swaps = 30

    # Realistic shapes for Qwen3-Coder-Next expert projections
    shapes = {
        "gate_w": (capacity, 512, 2048),
        "gate_s": (capacity, 8),
        "gate_b": (capacity, 512),
        "up_w": (capacity, 512, 2048),
        "up_s": (capacity, 8),
        "up_b": (capacity, 512),
        "down_w": (capacity, 2048, 512),
        "down_s": (capacity, 8),
        "down_b": (capacity, 2048),
    }

    # Use uint8 for weight tensors (quantized) and float32 for scales/biases
    dtypes = {}
    for k in shapes:
        dtypes[k] = mx.float32 if ("_s" in k or "_b" in k) else mx.uint8

    # Create target tensors (simulate cache)
    targets = {k: mx.random.uniform(shape=s).astype(dtypes[k]) for k, s in shapes.items()}
    mx.eval(*targets.values())

    indices = mx.array(list(range(num_swaps)), dtype=mx.int32)
    values = {k: mx.random.uniform(shape=(num_swaps, *s[1:])).astype(dtypes[k])
              for k, s in shapes.items()}
    mx.eval(*values.values())

    # Method 1: scatter_slots (if available)
    has_scatter_slots = hasattr(mx.fast, "scatter_slots")
    if has_scatter_slots:
        def run_scatter_slots():
            t_list = [targets[k] for k in shapes]
            v_list = [values[k] for k in shapes]
            results = mx.fast.scatter_slots(t_list, indices, v_list)
            mx.eval(*results)
            # Put results back
            for k, r in zip(shapes, results):
                targets[k] = r

        bench_method("scatter_slots", run_scatter_slots)
    else:
        print("  scatter_slots: NOT AVAILABLE (build custom MLX first)")

    # Method 2: pop() + __setitem__ (current approach)
    def run_donation():
        d = dict(targets)
        for k in shapes:
            w = d.pop(k)
            w[indices] = values[k]
            d[k] = w
        mx.eval(*d.values())
        targets.update(d)

    bench_method("pop+setitem", run_donation)

    # Method 3: list decompose + mx.stack (Phase 2 baseline)
    def run_stack():
        d = dict(targets)
        for k in shapes:
            t = d[k]
            parts = [t[i] for i in range(capacity)]
            for j in range(num_swaps):
                parts[indices[j].item()] = values[k][j]
            d[k] = mx.stack(parts)
        mx.eval(*d.values())
        targets.update(d)

    bench_method("mx.stack", run_stack, warmup=1, trials=3)


def correctness_test():
    """Verify scatter_slots produces correct results."""
    if not hasattr(mx.fast, "scatter_slots"):
        print("\n=== Correctness: SKIPPED (scatter_slots not available) ===")
        return

    print("\n=== Correctness test ===")
    capacity = 10
    target = mx.arange(capacity * 4, dtype=mx.float32).reshape(capacity, 4)
    mx.eval(target)

    indices = mx.array([2, 5, 8], dtype=mx.int32)
    new_values = mx.ones((3, 4), dtype=mx.float32) * -1
    mx.eval(new_values)

    results = mx.fast.scatter_slots([target], indices, [new_values])
    mx.eval(*results)
    out = results[0]

    # Check that slots 2, 5, 8 are -1 and others unchanged
    ok = True
    for i in range(capacity):
        expected = mx.ones(4) * -1 if i in [2, 5, 8] else mx.arange(i * 4, i * 4 + 4, dtype=mx.float32)
        actual = out[i]
        if not mx.allclose(actual, expected).item():
            print(f"  FAIL: slot {i} expected {expected.tolist()} got {actual.tolist()}")
            ok = False

    if ok:
        print("  PASS: all slots correct")
    else:
        print("  FAIL: some slots incorrect")


if __name__ == "__main__":
    print(f"MLX version: {mx.__version__}")
    print(f"Active memory: {mx.metal.get_active_memory() / 1e9:.2f} GB")
    print(f"scatter_slots available: {hasattr(mx.fast, 'scatter_slots')}")

    correctness_test()
    micro_benchmark()
