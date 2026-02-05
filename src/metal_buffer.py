"""Metal buffer management for expert caching."""

import struct
import time
from dataclasses import dataclass

import Metal
import numpy as np


@dataclass
class BufferStats:
    creation_ms: float
    kernel_ms: float
    total_ms: float
    size_mb: float


class MetalBufferManager:
    """Manage Metal buffers with CPU-backed memory."""

    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("No Metal device available")

        self.command_queue = self.device.newCommandQueue()
        self._compile_test_kernel()

        # Track allocations - store numpy arrays to keep them alive
        self._np_buffers: dict[int, np.ndarray] = {}
        self._metal_buffers: dict[int, object] = {}
        self._next_id = 0

    def _compile_test_kernel(self):
        """Compile a simple kernel for testing buffer validity."""
        source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void sum_reduce(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id == 0) {
                float sum = 0;
                for (int i = 0; i < 1024; i++) {
                    sum += input[i];
                }
                output[0] = sum;
            }
        }
        """
        options = Metal.MTLCompileOptions.new()
        library, error = self.device.newLibraryWithSource_options_error_(
            source, options, None
        )
        if error:
            raise RuntimeError(f"Shader compilation failed: {error}")

        self.test_function = library.newFunctionWithName_("sum_reduce")
        self.test_pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            self.test_function, None
        )
        if error:
            raise RuntimeError(f"Pipeline creation failed: {error}")

    def allocate_buffer(self, size: int) -> int:
        """Allocate a buffer and return its ID."""
        # Page-align for optimal performance
        page_size = 16384
        aligned_size = ((size + page_size - 1) // page_size) * page_size

        # Use numpy array - keeps memory alive and works well with PyObjC
        arr = np.zeros(aligned_size, dtype=np.uint8)

        buf_id = self._next_id
        self._next_id += 1
        self._np_buffers[buf_id] = arr

        return buf_id

    def fill_buffer(self, buf_id: int, data: bytes):
        """Fill buffer with data."""
        arr = self._np_buffers[buf_id]
        arr[:len(data)] = np.frombuffer(data, dtype=np.uint8)

    def get_buffer_size(self, buf_id: int) -> int:
        """Get buffer size."""
        return len(self._np_buffers[buf_id])

    def create_metal_buffer(self, buf_id: int) -> int:
        """
        Create Metal buffer from CPU data.

        Uses newBufferWithBytes which copies data into a Metal-managed buffer.
        For true zero-copy, we'd need to use IOSurface or similar.
        """
        if buf_id in self._metal_buffers:
            return buf_id

        arr = self._np_buffers[buf_id]
        data = arr.tobytes()

        # MTLResourceStorageModeShared = 0 (unified memory, no copy needed)
        metal_buf = self.device.newBufferWithBytes_length_options_(
            data,
            len(data),
            0,  # MTLResourceStorageModeShared
        )

        if not metal_buf:
            raise RuntimeError("Failed to create Metal buffer")

        self._metal_buffers[buf_id] = metal_buf
        return buf_id

    def create_metal_buffer_direct(self, data: bytes) -> object:
        """Create a Metal buffer directly from bytes (for simpler benchmarking)."""
        metal_buf = self.device.newBufferWithBytes_length_options_(
            data, len(data), 0
        )
        return metal_buf

    def release_metal_buffer(self, buf_id: int):
        """Release Metal buffer wrapper."""
        if buf_id in self._metal_buffers:
            del self._metal_buffers[buf_id]

    def free_buffer(self, buf_id: int):
        """Free both Metal and CPU buffer."""
        self.release_metal_buffer(buf_id)
        if buf_id in self._np_buffers:
            del self._np_buffers[buf_id]

    def _read_float_from_buffer(self, metal_buf: object) -> float:
        """Read a float from a Metal buffer's contents."""
        contents = metal_buf.contents()
        # objc.varlist returns single-byte bytes objects for each index
        raw_bytes = b''.join(contents[i] for i in range(4))
        return struct.unpack('f', raw_bytes)[0]

    def run_test_kernel(self, buf_id: int) -> float:
        """Run test kernel on buffer and return result."""
        if buf_id not in self._metal_buffers:
            raise ValueError("Buffer not in Metal")

        input_buf = self._metal_buffers[buf_id]
        output_buf = self.device.newBufferWithLength_options_(4, 0)

        cmd_buf = self.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        encoder.setComputePipelineState_(self.test_pipeline)
        encoder.setBuffer_offset_atIndex_(input_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(1, 1, 1),
            Metal.MTLSizeMake(1, 1, 1),
        )

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        return self._read_float_from_buffer(output_buf)

    def run_kernel_on_metal_buffer(self, metal_buf: object) -> float:
        """Run test kernel directly on a Metal buffer."""
        output_buf = self.device.newBufferWithLength_options_(4, 0)

        cmd_buf = self.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        encoder.setComputePipelineState_(self.test_pipeline)
        encoder.setBuffer_offset_atIndex_(metal_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(1, 1, 1),
            Metal.MTLSizeMake(1, 1, 1),
        )

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        return self._read_float_from_buffer(output_buf)

    def benchmark_buffer_lifecycle(self, size: int, n_trials: int = 10) -> BufferStats:
        """
        Benchmark the full buffer lifecycle:
        1. Allocate CPU buffer
        2. Fill with test data
        3. Create Metal buffer
        4. Run kernel
        5. Release and free
        """
        creation_times = []
        kernel_times = []
        total_times = []

        # Generate test data once
        test_data = np.random.randn(size // 4).astype(np.float32).tobytes()

        for _ in range(n_trials):
            total_start = time.perf_counter()

            # Allocation + Metal buffer creation
            create_start = time.perf_counter()
            buf_id = self.allocate_buffer(size)
            self.fill_buffer(buf_id, test_data)
            self.create_metal_buffer(buf_id)
            create_end = time.perf_counter()

            # Kernel
            kernel_start = time.perf_counter()
            _ = self.run_test_kernel(buf_id)
            kernel_end = time.perf_counter()

            # Cleanup
            self.free_buffer(buf_id)

            total_end = time.perf_counter()

            creation_times.append((create_end - create_start) * 1000)
            kernel_times.append((kernel_end - kernel_start) * 1000)
            total_times.append((total_end - total_start) * 1000)

        return BufferStats(
            creation_ms=np.mean(creation_times),
            kernel_ms=np.mean(kernel_times),
            total_ms=np.mean(total_times),
            size_mb=size / 1e6,
        )

    def benchmark_direct_metal(self, size: int, n_trials: int = 10) -> BufferStats:
        """Benchmark direct bytes -> Metal buffer (simpler path)."""
        creation_times = []
        kernel_times = []
        total_times = []

        test_data = np.random.randn(size // 4).astype(np.float32).tobytes()

        for _ in range(n_trials):
            total_start = time.perf_counter()

            create_start = time.perf_counter()
            metal_buf = self.create_metal_buffer_direct(test_data)
            create_end = time.perf_counter()

            kernel_start = time.perf_counter()
            _ = self.run_kernel_on_metal_buffer(metal_buf)
            kernel_end = time.perf_counter()

            # Let it be GC'd
            del metal_buf

            total_end = time.perf_counter()

            creation_times.append((create_end - create_start) * 1000)
            kernel_times.append((kernel_end - kernel_start) * 1000)
            total_times.append((total_end - total_start) * 1000)

        return BufferStats(
            creation_ms=np.mean(creation_times),
            kernel_ms=np.mean(kernel_times),
            total_ms=np.mean(total_times),
            size_mb=size / 1e6,
        )

    def get_device_info(self) -> dict:
        """Get Metal device information."""
        return {
            'name': self.device.name(),
            'recommended_working_set': self.device.recommendedMaxWorkingSetSize() / 1e9,
            'max_buffer_length': self.device.maxBufferLength() / 1e9,
            'unified_memory': self.device.hasUnifiedMemory(),
        }


def benchmark_metal_overhead():
    """Benchmark Metal buffer creation overhead at various sizes."""
    print("=== Metal Buffer Overhead Benchmark ===\n")

    mgr = MetalBufferManager()
    info = mgr.get_device_info()

    print(f"Device: {info['name']}")
    print(f"Recommended working set: {info['recommended_working_set']:.1f} GB")
    print(f"Max buffer length: {info['max_buffer_length']:.1f} GB")
    print(f"Unified memory: {info['unified_memory']}")
    print()

    sizes = [
        (512 * 1024, "512 KB"),
        (1 * 1024 * 1024, "1 MB"),
        (4 * 1024 * 1024, "4 MB"),
        (16 * 1024 * 1024, "16 MB"),
    ]

    print("Direct bytes → Metal (minimal path):")
    print("-" * 60)
    print(f"{'Size':12} | {'Create':>12} | {'Kernel':>12} | {'Total':>12}")
    print("-" * 60)

    for size, label in sizes:
        stats = mgr.benchmark_direct_metal(size, n_trials=20)
        print(f"{label:12} | {stats.creation_ms:10.3f} ms | {stats.kernel_ms:10.3f} ms | {stats.total_ms:10.3f} ms")

    print()


if __name__ == "__main__":
    benchmark_metal_overhead()
