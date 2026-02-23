import numpy as np

from ..processing.constants import LUMA_B, LUMA_G, LUMA_R
from ._numba_base import njit


@njit(fastmath=True, cache=True)
def numba_histogram_kernel(img, stride):
    """
    Calculates R, G, B, and YUV histograms in a single pass with striding.
    Input img must be uint8 (H, W, 3).
    Returns 6 arrays of size 256.
    """
    rows, cols, _ = img.shape

    hist_r = np.zeros(256, dtype=np.float32)
    hist_g = np.zeros(256, dtype=np.float32)
    hist_b = np.zeros(256, dtype=np.float32)
    hist_y = np.zeros(256, dtype=np.float32)
    hist_u = np.zeros(256, dtype=np.float32)
    hist_v = np.zeros(256, dtype=np.float32)

    for r in range(0, rows, stride):
        for c in range(0, cols, stride):
            r_val = int(img[r, c, 0])
            g_val = int(img[r, c, 1])
            b_val = int(img[r, c, 2])

            hist_r[r_val] += 1
            hist_g[g_val] += 1
            hist_b[b_val] += 1

            # RGB to YUV (Standardized Luma Rec. 709)
            # Y = LUMA_R*R + LUMA_G*G + LUMA_B*B
            # U/V still use BT.601-like coefficients for now as they are less critical for luma-focused tools

            y = int(LUMA_R * r_val + LUMA_G * g_val + LUMA_B * b_val)
            u = ((-150 * r_val - 296 * g_val + 446 * b_val) >> 10) + 128
            v = ((630 * r_val - 527 * g_val - 102 * b_val) >> 10) + 128

            # Clamp YUV
            if y < 0:
                y = 0
            elif y > 255:
                y = 255
            if u < 0:
                u = 0
            elif u > 255:
                u = 255
            if v < 0:
                v = 0
            elif v > 255:
                v = 255

            hist_y[y] += 1
            hist_u[u] += 1
            hist_v[v] += 1

    return hist_r, hist_g, hist_b, hist_y, hist_u, hist_v
