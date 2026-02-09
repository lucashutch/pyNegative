import numpy as np
from ._numba_base import njit, prange


@njit(fastmath=True, cache=True, parallel=True)
def dark_channel_kernel(img):
    """
    Calculates the dark channel of an image: min(R, G, B) for each pixel.
    """
    rows, cols, _ = img.shape
    dark = np.empty((rows, cols), dtype=np.float32)

    for r in prange(rows):
        for c in range(cols):
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]

            # min(R, G, B)
            m = r_val
            if g_val < m:
                m = g_val
            if b_val < m:
                m = b_val
            dark[r, c] = m

    return dark


@njit(fastmath=True, cache=True, parallel=True)
def dehaze_recovery_kernel(img, transmission, atmospheric_light):
    """
    Highly optimized radiance recovery: J(x) = (I(x) - A) / max(t(x), t0) + A
    """
    rows, cols, _ = img.shape
    out = np.empty_like(img)

    a_r = atmospheric_light[0]
    a_g = atmospheric_light[1]
    a_b = atmospheric_light[2]

    for r in prange(rows):
        for c in range(cols):
            # Reciprocal is much faster than 3 divisions
            inv_t = 1.0 / max(transmission[r, c], 0.1)

            r_val = (img[r, c, 0] - a_r) * inv_t + a_r
            g_val = (img[r, c, 1] - a_g) * inv_t + a_g
            b_val = (img[r, c, 2] - a_b) * inv_t + a_b

            out[r, c, 0] = max(0.0, min(1.0, r_val))
            out[r, c, 1] = max(0.0, min(1.0, g_val))
            out[r, c, 2] = max(0.0, min(1.0, b_val))
    return out
