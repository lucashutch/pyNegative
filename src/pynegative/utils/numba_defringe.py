import numpy as np

from ._numba_base import njit, prange
from ..processing.constants import LUMA_B, LUMA_G, LUMA_R


@njit(inline="always")
def _get_luma(r, g, b):
    return LUMA_R * r + LUMA_G * g + LUMA_B * b


@njit(fastmath=True, cache=True, parallel=True)
def defringe_kernel(img, out, purple_thresh, green_thresh, edge_thresh, radius):
    h, w, channels = img.shape

    # Pre-calculate luma for edge detection
    luma = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            luma[y, x] = _get_luma(img[y, x, 0], img[y, x, 1], img[y, x, 2])

    r_int = int(radius)
    search_r = max(1, r_int)

    for y in prange(h):
        for x in range(w):
            # 1. Edge Detection (Local Contrast)
            # Use clamped bounds to avoid per-pixel branch inside inner loop
            y0 = max(0, y - search_r)
            y1 = min(h, y + search_r + 1)
            x0 = max(0, x - search_r)
            x1 = min(w, x + search_r + 1)

            max_l = luma[y, x]
            min_l = max_l

            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    val = luma[ny, nx]
                    if val > max_l:
                        max_l = val
                    if val < min_l:
                        min_l = val

            contrast = max_l - min_l

            # 2. Color Analysis
            r_val = img[y, x, 0]
            g_val = img[y, x, 1]
            b_val = img[y, x, 2]
            luma_val = luma[y, x]

            # Cr and Cb approx (Red-Green and Blue-Yellow scales)
            # Purple: Cr > 0, Cb > 0
            # Green: Cr < 0, Cb < 0
            cr = r_val - luma_val
            cb = b_val - luma_val

            suppression = 0.0
            if contrast > edge_thresh:
                is_purple = cr > 0.01 and cb > 0.01
                is_green = cr < -0.01 and cb < -0.01

                if is_purple and purple_thresh > 0:
                    suppression = purple_thresh
                elif is_green and green_thresh > 0:
                    suppression = green_thresh

            if suppression > 0.0:
                # Move towards luma (desaturate)
                out[y, x, 0] = r_val - cr * suppression
                out[y, x, 1] = g_val + (luma_val - g_val) * suppression
                out[y, x, 2] = b_val - cb * suppression
            else:
                out[y, x, 0] = r_val
                out[y, x, 1] = g_val
                out[y, x, 2] = b_val
