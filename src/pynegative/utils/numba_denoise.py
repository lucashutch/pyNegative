import numpy as np

from ._numba_base import njit, prange


@njit(fastmath=True, cache=True, inline="always")
def _fast_exp(x):
    """Fast exponential approximation for negative x values.

    Uses a rational polynomial that is accurate to ~0.1% for x in [-10, 0].
    For bilateral weights we only need negative exponents so this is ideal.
    Much faster than np.exp() in tight Numba loops.
    """
    # Clamp to avoid underflow — exp(-10) ≈ 4.5e-5, negligible weight
    if x < -10.0:
        return 0.0
    # 6th-degree minimax polynomial approximation of exp(x) for x in [-10, 0]
    # Coefficients via Remez/Chebyshev fitting
    a = 1.0 + x * (
        1.0
        + x
        * (
            0.5
            + x * (0.16666667 + x * (0.04166667 + x * (0.00833333 + x * 0.00138889)))
        )
    )
    # Clamp to [0, 1] to prevent any negative blips from the polynomial
    if a < 0.0:
        return 0.0
    return a


@njit(fastmath=True, cache=True, parallel=True)
def bilateral_kernel_yuv(
    img_yuv, strength, sigma_color_y, sigma_space_y, sigma_color_uv, sigma_space_uv
):
    """
    Joint-bilateral filter for YUV image.
    Luma is self-guided (standard bilateral). Chroma is **Y-guided**
    (joint/cross bilateral): the colour-similarity weight for U and V
    is derived from the luma (Y) channel, not from the chroma values
    themselves. This dramatically improves chroma denoising because:
      - Luma carries the real edge structure; chroma is too noisy to
        reliably detect its own edges.
      - Flat-luma regions get aggressively smoothed in chroma.
      - Luma edges prevent chroma bleeding across real boundaries.

    Chroma uses a 7×7 window (was 5×5) because chroma noise is
    lower-frequency and our eyes are less sensitive to chroma detail.

    Both U and V share the same Y-derived weight per neighbour, saving
    one _fast_exp call per inner iteration compared to self-guided.
    """
    rows, cols, _ = img_yuv.shape
    out = np.empty_like(img_yuv)

    # --- Pre-extract contiguous channel planes for better cache locality ---
    plane_y = np.ascontiguousarray(img_yuv[:, :, 0])
    plane_u = np.ascontiguousarray(img_yuv[:, :, 1])
    plane_v = np.ascontiguousarray(img_yuv[:, :, 2])

    # --- Pre-calculate spatial weights (only done once, not per-pixel) ---
    # Luma 3×3
    s_weights_y = np.zeros((3, 3), dtype=np.float64)
    s_inv_y = -1.0 / (2.0 * sigma_space_y * sigma_space_y)
    for i in range(-1, 2):
        for j in range(-1, 2):
            s_weights_y[i + 1, j + 1] = _fast_exp((i * i + j * j) * s_inv_y)

    # Chroma 7×7 (larger window captures spatially-spread chroma noise)
    chroma_half = 3  # 7×7 kernel
    chroma_size = 2 * chroma_half + 1  # 7
    s_weights_uv = np.zeros((chroma_size, chroma_size), dtype=np.float64)
    s_inv_uv = -1.0 / (2.0 * sigma_space_uv * sigma_space_uv)
    for i in range(-chroma_half, chroma_half + 1):
        for j in range(-chroma_half, chroma_half + 1):
            s_weights_uv[i + chroma_half, j + chroma_half] = _fast_exp(
                (i * i + j * j) * s_inv_uv
            )

    c_inv_y = -1.0 / (2.0 * sigma_color_y * sigma_color_y)
    c_inv_uv = -1.0 / (2.0 * sigma_color_uv * sigma_color_uv)

    # Border widths for the two kernel sizes
    border_uv = chroma_half  # 3 for 7×7 chroma kernel

    # =====================================================================
    #  INTERIOR: no bounds checks needed  (vast majority of pixels)
    # =====================================================================
    for r in prange(border_uv, rows - border_uv):
        for c in range(border_uv, cols - border_uv):
            # --- Luma (Y) — 3×3 window, no bounds checks ---
            y_val = plane_y[r, c]
            sum_y = 0.0
            w_sum_y = 0.0

            for i in range(-1, 2):
                rr = r + i
                for j in range(-1, 2):
                    cc = c + j
                    p_val = plane_y[rr, cc]
                    diff = p_val - y_val
                    w = s_weights_y[i + 1, j + 1] * _fast_exp(diff * diff * c_inv_y)
                    sum_y += p_val * w
                    w_sum_y += w

            out[r, c, 0] = sum_y / w_sum_y if w_sum_y > 0 else y_val

            # --- Chroma (U+V fused, Y-guided) — 7×7 window, no bounds checks ---
            u_val = plane_u[r, c]
            v_val = plane_v[r, c]
            sum_u = 0.0
            sum_v = 0.0
            w_sum_uv = 0.0

            for i in range(-chroma_half, chroma_half + 1):
                rr = r + i
                for j in range(-chroma_half, chroma_half + 1):
                    cc = c + j
                    sw = s_weights_uv[i + chroma_half, j + chroma_half]

                    # Y-guided: colour weight from luma differences
                    dy = plane_y[rr, cc] - y_val
                    w = sw * _fast_exp(dy * dy * c_inv_uv)

                    # Same weight for both U and V
                    sum_u += plane_u[rr, cc] * w
                    sum_v += plane_v[rr, cc] * w
                    w_sum_uv += w

            out[r, c, 1] = sum_u / w_sum_uv if w_sum_uv > 0 else u_val
            out[r, c, 2] = sum_v / w_sum_uv if w_sum_uv > 0 else v_val

    # =====================================================================
    #  BORDER: needs bounds checks  (thin strip around the edge)
    # =====================================================================
    for r in prange(rows):
        # Skip interior rows — they were already handled above
        if border_uv <= r < rows - border_uv:
            continue

        for c in range(cols):
            # --- Luma (Y) — 3×3 with bounds checks ---
            y_val = plane_y[r, c]
            sum_y = 0.0
            w_sum_y = 0.0

            for i in range(-1, 2):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-1, 2):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    p_val = plane_y[rr, cc]
                    diff = p_val - y_val
                    w = s_weights_y[i + 1, j + 1] * _fast_exp(diff * diff * c_inv_y)
                    sum_y += p_val * w
                    w_sum_y += w

            out[r, c, 0] = sum_y / w_sum_y if w_sum_y > 0 else y_val

            # --- Chroma (U+V fused, Y-guided) — 7×7 with bounds checks ---
            u_val = plane_u[r, c]
            v_val = plane_v[r, c]
            sum_u = 0.0
            sum_v = 0.0
            w_sum_uv = 0.0

            for i in range(-chroma_half, chroma_half + 1):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-chroma_half, chroma_half + 1):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    sw = s_weights_uv[i + chroma_half, j + chroma_half]

                    dy = plane_y[rr, cc] - y_val
                    w = sw * _fast_exp(dy * dy * c_inv_uv)

                    sum_u += plane_u[rr, cc] * w
                    sum_v += plane_v[rr, cc] * w
                    w_sum_uv += w

            out[r, c, 1] = sum_u / w_sum_uv if w_sum_uv > 0 else u_val
            out[r, c, 2] = sum_v / w_sum_uv if w_sum_uv > 0 else v_val

    # Handle left/right border columns of interior rows
    for r in prange(border_uv, rows - border_uv):
        for c in range(cols):
            # Only process border columns
            if border_uv <= c < cols - border_uv:
                continue

            # --- Luma (Y) — 3×3 with bounds checks ---
            y_val = plane_y[r, c]
            sum_y = 0.0
            w_sum_y = 0.0

            for i in range(-1, 2):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-1, 2):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    p_val = plane_y[rr, cc]
                    diff = p_val - y_val
                    w = s_weights_y[i + 1, j + 1] * _fast_exp(diff * diff * c_inv_y)
                    sum_y += p_val * w
                    w_sum_y += w

            out[r, c, 0] = sum_y / w_sum_y if w_sum_y > 0 else y_val

            # --- Chroma (U+V fused, Y-guided) — 7×7 with bounds checks ---
            u_val = plane_u[r, c]
            v_val = plane_v[r, c]
            sum_u = 0.0
            sum_v = 0.0
            w_sum_uv = 0.0

            for i in range(-chroma_half, chroma_half + 1):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-chroma_half, chroma_half + 1):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    sw = s_weights_uv[i + chroma_half, j + chroma_half]

                    dy = plane_y[rr, cc] - y_val
                    w = sw * _fast_exp(dy * dy * c_inv_uv)

                    sum_u += plane_u[rr, cc] * w
                    sum_v += plane_v[rr, cc] * w
                    w_sum_uv += w

            out[r, c, 1] = sum_u / w_sum_uv if w_sum_uv > 0 else u_val
            out[r, c, 2] = sum_v / w_sum_uv if w_sum_uv > 0 else v_val

    return out


@njit(fastmath=True, cache=True, parallel=True)
def bilateral_kernel_luma(img_yuv, strength, sigma_color_y, sigma_space_y):
    """
    Optimized bilateral filter for Luma (Y channel) only.
    Uses a 3×3 window and skips chroma processing entirely.
    Used when only luma denoising is requested.
    """
    rows, cols, _ = img_yuv.shape
    out = np.empty_like(img_yuv)

    plane_y = np.ascontiguousarray(img_yuv[:, :, 0])
    plane_u = np.ascontiguousarray(img_yuv[:, :, 1])
    plane_v = np.ascontiguousarray(img_yuv[:, :, 2])

    # Pre-calculate spatial weights for 3×3 kernel
    s_weights_y = np.zeros((3, 3), dtype=np.float64)
    s_inv_y = -1.0 / (2.0 * sigma_space_y * sigma_space_y)
    for i in range(-1, 2):
        for j in range(-1, 2):
            s_weights_y[i + 1, j + 1] = _fast_exp((i * i + j * j) * s_inv_y)

    c_inv_y = -1.0 / (2.0 * sigma_color_y * sigma_color_y)

    # Interior: no bounds checks
    for r in prange(1, rows - 1):
        for c in range(1, cols - 1):
            y_val = plane_y[r, c]
            sum_y = 0.0
            w_sum_y = 0.0

            for i in range(-1, 2):
                rr = r + i
                for j in range(-1, 2):
                    cc = c + j
                    p_val = plane_y[rr, cc]
                    diff = p_val - y_val
                    w = s_weights_y[i + 1, j + 1] * _fast_exp(diff * diff * c_inv_y)
                    sum_y += p_val * w
                    w_sum_y += w

            out[r, c, 0] = sum_y / w_sum_y if w_sum_y > 0 else y_val
            out[r, c, 1] = plane_u[r, c]
            out[r, c, 2] = plane_v[r, c]

    # Border handling with bounds checks
    for r in prange(rows):
        if 1 <= r < rows - 1:
            continue
        for c in range(cols):
            y_val = plane_y[r, c]
            sum_y = 0.0
            w_sum_y = 0.0

            for i in range(-1, 2):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-1, 2):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    p_val = plane_y[rr, cc]
                    diff = p_val - y_val
                    w = s_weights_y[i + 1, j + 1] * _fast_exp(diff * diff * c_inv_y)
                    sum_y += p_val * w
                    w_sum_y += w

            out[r, c, 0] = sum_y / w_sum_y if w_sum_y > 0 else y_val
            out[r, c, 1] = plane_u[r, c]
            out[r, c, 2] = plane_v[r, c]

    # Side borders for interior rows
    for r in prange(1, rows - 1):
        for c in range(cols):
            if 1 <= c < cols - 1:
                continue

            y_val = plane_y[r, c]
            sum_y = 0.0
            w_sum_y = 0.0

            for i in range(-1, 2):
                rr = r + i
                for j in range(-1, 2):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    p_val = plane_y[rr, cc]
                    diff = p_val - y_val
                    w = s_weights_y[i + 1, j + 1] * _fast_exp(diff * diff * c_inv_y)
                    sum_y += p_val * w
                    w_sum_y += w

            out[r, c, 0] = sum_y / w_sum_y if w_sum_y > 0 else y_val
            out[r, c, 1] = plane_u[r, c]
            out[r, c, 2] = plane_v[r, c]

    return out


@njit(fastmath=True, cache=True, parallel=True)
def bilateral_kernel_chroma(img_yuv, plane_y, strength, sigma_color_uv, sigma_space_uv):
    """
    Optimized bilateral filter for Chroma (U+V channels) only, Y-guided.
    Uses a 7×7 window with Y-guidance. Assumes plane_y (pre-computed luma)
    is passed in to avoid recomputation.
    Used when only chroma denoising is requested, or after luma in fused mode.
    """
    rows, cols, _ = img_yuv.shape
    out = np.empty_like(img_yuv)

    plane_u = np.ascontiguousarray(img_yuv[:, :, 1])
    plane_v = np.ascontiguousarray(img_yuv[:, :, 2])

    # Pre-calculate spatial weights for 7×7 kernel
    chroma_half = 3
    chroma_size = 2 * chroma_half + 1
    s_weights_uv = np.zeros((chroma_size, chroma_size), dtype=np.float64)
    s_inv_uv = -1.0 / (2.0 * sigma_space_uv * sigma_space_uv)
    for i in range(-chroma_half, chroma_half + 1):
        for j in range(-chroma_half, chroma_half + 1):
            s_weights_uv[i + chroma_half, j + chroma_half] = _fast_exp(
                (i * i + j * j) * s_inv_uv
            )

    c_inv_uv = -1.0 / (2.0 * sigma_color_uv * sigma_color_uv)
    border_uv = chroma_half

    # Interior: no bounds checks
    for r in prange(border_uv, rows - border_uv):
        for c in range(border_uv, cols - border_uv):
            y_val = plane_y[r, c]
            u_val = plane_u[r, c]
            v_val = plane_v[r, c]
            sum_u = 0.0
            sum_v = 0.0
            w_sum_uv = 0.0

            for i in range(-chroma_half, chroma_half + 1):
                rr = r + i
                for j in range(-chroma_half, chroma_half + 1):
                    cc = c + j
                    sw = s_weights_uv[i + chroma_half, j + chroma_half]
                    dy = plane_y[rr, cc] - y_val
                    w = sw * _fast_exp(dy * dy * c_inv_uv)
                    sum_u += plane_u[rr, cc] * w
                    sum_v += plane_v[rr, cc] * w
                    w_sum_uv += w

            out[r, c, 0] = plane_y[r, c]
            out[r, c, 1] = sum_u / w_sum_uv if w_sum_uv > 0 else u_val
            out[r, c, 2] = sum_v / w_sum_uv if w_sum_uv > 0 else v_val

    # Border rows with bounds checks
    for r in prange(rows):
        if border_uv <= r < rows - border_uv:
            continue

        for c in range(cols):
            y_val = plane_y[r, c]
            u_val = plane_u[r, c]
            v_val = plane_v[r, c]
            sum_u = 0.0
            sum_v = 0.0
            w_sum_uv = 0.0

            for i in range(-chroma_half, chroma_half + 1):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-chroma_half, chroma_half + 1):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    sw = s_weights_uv[i + chroma_half, j + chroma_half]
                    dy = plane_y[rr, cc] - y_val
                    w = sw * _fast_exp(dy * dy * c_inv_uv)
                    sum_u += plane_u[rr, cc] * w
                    sum_v += plane_v[rr, cc] * w
                    w_sum_uv += w

            out[r, c, 0] = plane_y[r, c]
            out[r, c, 1] = sum_u / w_sum_uv if w_sum_uv > 0 else u_val
            out[r, c, 2] = sum_v / w_sum_uv if w_sum_uv > 0 else v_val

    # Side borders for interior rows with bounds checks
    for r in prange(border_uv, rows - border_uv):
        for c in range(cols):
            if border_uv <= c < cols - border_uv:
                continue

            y_val = plane_y[r, c]
            u_val = plane_u[r, c]
            v_val = plane_v[r, c]
            sum_u = 0.0
            sum_v = 0.0
            w_sum_uv = 0.0

            for i in range(-chroma_half, chroma_half + 1):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-chroma_half, chroma_half + 1):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue
                    sw = s_weights_uv[i + chroma_half, j + chroma_half]
                    dy = plane_y[rr, cc] - y_val
                    w = sw * _fast_exp(dy * dy * c_inv_uv)
                    sum_u += plane_u[rr, cc] * w
                    sum_v += plane_v[rr, cc] * w
                    w_sum_uv += w

            out[r, c, 0] = plane_y[r, c]
            out[r, c, 1] = sum_u / w_sum_uv if w_sum_uv > 0 else u_val
            out[r, c, 2] = sum_v / w_sum_uv if w_sum_uv > 0 else v_val

    return out
