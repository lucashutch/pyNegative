from ._numba_base import njit, prange

@njit(fastmath=True, cache=True, parallel=True)
def sharpen_kernel(img, blurred, percent):
    """
    Highly optimized fused unsharp mask kernel.
    """
    rows, cols, _ = img.shape
    factor = percent * 0.01

    for r in prange(rows):
        for c in range(cols):
            # Explicitly unroll channels for better ILP
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]

            r_res = r_val + (r_val - blurred[r, c, 0]) * factor
            g_res = g_val + (g_val - blurred[r, c, 1]) * factor
            b_res = b_val + (b_val - blurred[r, c, 2]) * factor

            img[r, c, 0] = max(0.0, min(1.0, r_res))
            img[r, c, 1] = max(0.0, min(1.0, g_res))
            img[r, c, 2] = max(0.0, min(1.0, b_res))
