from ._numba_base import njit, prange


@njit(fastmath=True, cache=True, parallel=True)
def tone_map_kernel(
    img,  # float32 array (H, W, 3) - MODIFIED IN PLACE
    exposure,  # float
    contrast,  # float
    blacks,  # float (offset)
    whites,  # float (scale)
    shadows,  # float
    highlights,  # float
    saturation,  # float
    r_mult,  # float
    g_mult,  # float
    b_mult,  # float
):
    """
    Highly optimized fused kernel for tone mapping.
    Uses analytical luminance tracking to avoid redundant calculations.
    """
    rows, cols, _ = img.shape

    # Pre-calculate constants
    inv_denom = 1.0 / (whites - blacks) if abs(whites - blacks) > 1e-6 else 1000000.0
    abs_h = abs(highlights)

    # Statistics tracking
    clipped_shadows = 0
    clipped_highlights = 0
    pixel_sum = 0.0

    for r in prange(rows):
        for c in range(cols):
            r_val = img[r, c, 0] * r_mult * exposure
            g_val = img[r, c, 1] * g_mult * exposure
            b_val = img[r, c, 2] * b_mult * exposure

            if contrast != 1.0:
                r_val = (r_val - 0.5) * contrast + 0.5
                g_val = (g_val - 0.5) * contrast + 0.5
                b_val = (b_val - 0.5) * contrast + 0.5

            # Initial luminance for masks (Rec 709)
            lum = 0.2126 * r_val + 0.7152 * g_val + 0.0722 * b_val

            r_val = (r_val - blacks) * inv_denom
            g_val = (g_val - blacks) * inv_denom
            b_val = (b_val - blacks) * inv_denom
            # Track luminance through blacks/whites offset
            lum = (lum - blacks) * inv_denom

            if shadows != 0.0 or highlights != 0.0 or saturation != 1.0:
                # 1. Shadows
                if shadows != 0.0:
                    clim = max(0.0, min(1.0, lum))
                    s_mask = (1.0 - clim) * (1.0 - clim)
                    factor = 1.0 + shadows * s_mask
                    r_val *= factor
                    g_val *= factor
                    b_val *= factor
                    lum *= factor  # Analytical lum update

                # 2. Highlights
                if highlights != 0.0:
                    if highlights < 0.0:
                        # Recovery (Division)
                        h_mask = max(0.0, lum) ** 2
                        factor = 1.0 + abs_h * h_mask
                        inv_f = 1.0 / factor
                        r_val *= inv_f
                        g_val *= inv_f
                        b_val *= inv_f
                        lum *= inv_f  # Analytical lum update
                    else:
                        # Boost (Blend)
                        clim = max(0.0, min(1.0, lum))
                        h_term = highlights * clim * clim
                        inv_h = 1.0 - h_term
                        r_val = r_val * inv_h + h_term
                        g_val = g_val * inv_h + h_term
                        b_val = b_val * inv_h + h_term
                        lum = lum * inv_h + h_term  # Analytical lum update

                # 3. Saturation (uses tracked 'lum' directly)
                if saturation != 1.0:
                    c_lum = max(0.0, min(1.0, lum))
                    r_val = c_lum + (r_val - c_lum) * saturation
                    g_val = c_lum + (g_val - c_lum) * saturation
                    b_val = c_lum + (b_val - c_lum) * saturation

            # Clipping & Stats
            if r_val < 0.0:
                clipped_shadows += 1
            elif r_val > 1.0:
                clipped_highlights += 1
            if g_val < 0.0:
                clipped_shadows += 1
            elif g_val > 1.0:
                clipped_highlights += 1
            if b_val < 0.0:
                clipped_shadows += 1
            elif b_val > 1.0:
                clipped_highlights += 1

            r_val = max(0.0, min(1.0, r_val))
            g_val = max(0.0, min(1.0, g_val))
            b_val = max(0.0, min(1.0, b_val))

            img[r, c, 0] = r_val
            img[r, c, 1] = g_val
            img[r, c, 2] = b_val
            pixel_sum += (r_val + g_val + b_val) * 0.33333333

    return clipped_shadows, clipped_highlights, pixel_sum
