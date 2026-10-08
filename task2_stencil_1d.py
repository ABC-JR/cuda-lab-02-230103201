"""Task 2: 1D 3-point smoothing stencil with boundary clamping."""
import numpy as np
from numba import cuda

TPB = 256


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        left = d_in[idx - 1] if idx > 0 else d_in[0]          # clamp left
        right = d_in[idx + 1] if idx < N - 1 else d_in[N - 1]  # clamp right
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    N = h_in.size
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array(N, dtype=np.float32)
    blocks = (N + TPB - 1) // TPB
    stencil_1d[blocks, TPB](d_in, d_out, N)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


if __name__ == "__main__":
    N = 10007  # odd, non-power-of-two
    h_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)
    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)
    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4)
    delta = float(np.max(np.abs(h_out_gpu - cpu_ref)))
    print(f"TASK 2 PASSED: MAX DELTA = {delta}")
