"""Task 1: Warp divergence microbenchmark."""
import time
import numpy as np
from numba import cuda

ITERS = 1000
N = 1 << 22  # 4,194,304 elements (multiple of 32, so warps are full)
TPB = 256


@cuda.jit
def kernel_uniform(arr, n):
    idx = cuda.grid(1)
    if idx < n:
        acc = arr[idx]
        for _ in range(ITERS):
            acc = acc * 1.0001 + 0.5  # multiply-accumulate
        arr[idx] = acc


@cuda.jit
def kernel_divergent(arr, n):
    idx = cuda.grid(1)
    if idx < n:
        acc = arr[idx]
        if idx % 2 == 0:
            for _ in range(ITERS):
                acc = acc * 1.0001 + 0.5  # Path 1: MAC
        else:
            for _ in range(ITERS):
                acc = (acc - 0.5) / 1.0001  # Path 2: subtract-divide
        arr[idx] = acc


@cuda.jit
def kernel_warp_aligned(arr, n):
    idx = cuda.grid(1)
    if idx < n:
        acc = arr[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(ITERS):
                acc = acc * 1.0001 + 0.5  # Path 1: MAC
        else:
            for _ in range(ITERS):
                acc = (acc - 0.5) / 1.0001  # Path 2: subtract-divide
        arr[idx] = acc


def bench(kernel, d_arr, n, trials=10):
    blocks = (n + TPB - 1) // TPB
    kernel[blocks, TPB](d_arr, n)  # warm-up (also triggers JIT compile)
    cuda.synchronize()
    times = []
    for _ in range(trials):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[blocks, TPB](d_arr, n)
        cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def main():
    h = np.random.rand(N).astype(np.float32)
    d = cuda.to_device(h)  # transfer excluded from timing
    t_a = bench(kernel_uniform, d, N)
    t_b = bench(kernel_divergent, d, N)
    t_c = bench(kernel_warp_aligned, d, N)

    print("| Kernel | Avg time (ms) | Slowdown vs A |")
    print("|---|--:|--:|")
    print(f"| A: Uniform path | {t_a:.3f} | 1.00x |")
    print(f"| B: Full divergence (idx % 2) | {t_b:.3f} | {t_b / t_a:.2f}x |")
    print(f"| C: Warp-aligned (warp_id % 2) | {t_c:.3f} | {t_c / t_a:.2f}x |")


if __name__ == "__main__":
    main()
