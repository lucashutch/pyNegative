import numpy as np
from numba import njit, prange


@njit(inline="always")
def _get_luma(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


@njit(parallel=True)
def defringe_kernel(img, out, purple_thresh, green_thresh, edge_thresh, radius):
    h, w, channels = img.shape

    # Pre-calculate luma for edge detection
    luma = np.zeros((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            luma[y, x] = _get_luma(img[y, x, 0], img[y, x, 1], img[y, x, 2])

    for y in prange(h):
        for x in range(w):
            # 1. Edge Detection (Local Contrast)
            # Check neighborhood defined by radius
            r_int = int(radius)
            max_l = luma[y, x]
            min_l = luma[y, x]

            # If radius is 0, we still check 1x1 (current pixel)
            # If radius is 1, we check 3x3
            search_r = max(1, r_int)

            for dy in range(-search_r, search_r + 1):
                for dx in range(-search_r, search_r + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        val = luma[ny, nx]
                        if val > max_l:
                            max_l = val
                        if val < min_l:
                            min_l = val

            contrast = max_l - min_l

            # 2. Color Analysis
            r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            l = luma[y, x]

            # Cr and Cb approx (Red-Green and Blue-Yellow scales)
            # Purple: Cr > 0, Cb > 0
            # Green: Cr < 0, Cb < 0
            cr = r - l
            cb = b - l

            is_purple = cr > 0.01 and cb > 0.01
            is_green = cr < -0.01 and cb < -0.01

            suppression = 0.0
            if contrast > edge_thresh:
                if is_purple and purple_thresh > 0:
                    # Scale suppression by how much purple is there and the threshold
                    # We want higher threshold to mean MORE aggressive removal
                    suppression = purple_thresh
                elif is_green and green_thresh > 0:
                    suppression = green_thresh

            if suppression > 0:
                # Move towards luma (desaturate)
                out[y, x, 0] = r - cr * suppression
                out[y, x, 1] = g + (l - g) * suppression
                out[y, x, 2] = b - cb * suppression
            else:
                out[y, x, 0] = r
                out[y, x, 1] = g
                out[y, x, 2] = b
