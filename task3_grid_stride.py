"""Task 3: Grid-stride loop scaling with a fixed hardware grid."""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64  # 256 * 64 = 16,384 hardware threads, no more


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        d_arr, np.float32(factor), h_arr.size)
    cuda.synchronize()
    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 1_000_003  # far larger than 16,384 threads, not a multiple of the stride
    factor = 4.25
    res = run_grid_stride(np.ones(N, dtype=np.float32), factor)
    assert np.allclose(res, factor)
    print(f"TASK 3 PASSED: {N} elements, all equal {factor}")
