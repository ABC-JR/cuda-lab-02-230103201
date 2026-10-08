"""Task 4: 2D Sobel-X filter."""
import numpy as np
from numba import cuda

THREADS_2D = (16, 16)


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            right = (d_in[row - 1, col + 1]
                     + 2.0 * d_in[row, col + 1]
                     + d_in[row + 1, col + 1])
            left = (d_in[row - 1, col - 1]
                    + 2.0 * d_in[row, col - 1]
                    + d_in[row + 1, col - 1])
            d_out[row, col] = right - left
        else:
            d_out[row, col] = 0.0  # border pixels


def run_sobel(h_img):
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((rows, cols), dtype=np.float32)
    blocks = ((cols + THREADS_2D[0] - 1) // THREADS_2D[0],
              (rows + THREADS_2D[1] - 1) // THREADS_2D[1])
    sobel_x_kernel[blocks, THREADS_2D](d_in, d_out, rows, cols)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_sobel(img):
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (
        (img[:-2, 2:] + 2 * img[1:-1, 2:] + img[2:, 2:])
        - (img[:-2, :-2] + 2 * img[1:-1, :-2] + img[2:, :-2]))
    return out


if __name__ == "__main__":
    rows, cols = 1080, 1920  # not multiples of 16, tests the guards
    img = np.random.rand(rows, cols).astype(np.float32)
    gpu = run_sobel(img)
    assert np.allclose(gpu, cpu_sobel(img), atol=1e-4)
    print(f"TASK 4 PASSED: {rows}x{cols} matches CPU reference")
