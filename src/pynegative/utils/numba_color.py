import numpy as np

from ..processing.constants import LUMA_B, LUMA_G, LUMA_R
from ._numba_base import njit, prange


@njit(fastmath=True, cache=True, parallel=True)
def preprocess_kernel(
    img,
    r_mult,
    g_mult,
    b_mult,
    exposure,
    vig_k1=0.0,
    vig_k2=0.0,
    vig_k3=0.0,
    vig_cx=0.0,
    vig_cy=0.0,
    full_w=1.0,
    full_h=1.0,
):
    """
    Apply vignette, WB multipliers, and exposure in linear space.
    Operates in-place on the input image.
    Vignette is applied first, then WB+Exposure.
    """
    rows, cols, _ = img.shape
    exp_mult = 2.0**exposure

    has_vignette = abs(vig_k1) > 1e-6 or abs(vig_k2) > 1e-6 or abs(vig_k3) > 1e-6
    if has_vignette:
        inv_max_r2 = 1.0 / ((full_w / 2.0) ** 2 + (full_h / 2.0) ** 2)

    for r in prange(rows):
        for c in range(cols):
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]

            if has_vignette:
                dx = c - vig_cx
                dy = r - vig_cy
                r2 = dx * dx + dy * dy
                rn2 = r2 * inv_max_r2
                rn4 = rn2 * rn2
                rn6 = rn4 * rn2
                vignette_gain = 1.0 + vig_k1 * rn2 + vig_k2 * rn4 + vig_k3 * rn6
                r_val *= vignette_gain
                g_val *= vignette_gain
                b_val *= vignette_gain

            img[r, c, 0] = r_val * r_mult * exp_mult
            img[r, c, 1] = g_val * g_mult * exp_mult
            img[r, c, 2] = b_val * b_mult * exp_mult


@njit(inline="always")
def _linear_to_srgb(x):
    """Applies sRGB gamma curve."""
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * (x ** (1.0 / 2.4)) - 0.055


@njit(fastmath=True, cache=True, parallel=True)
def tone_map_kernel(
    img,
    contrast,
    blacks,
    whites,
    shadows,
    highlights,
    saturation,
    apply_gamma,
):
    """
    Highly optimized fused kernel for tone mapping.
    Operates on pre-processed data (WB, exposure, vignette already applied).
    Applies: Contrast → Levels → Shadows/Highlights → Saturation → Gamma
    """
    rows, cols, _ = img.shape

    # Map Whites logically: 0.0 is neutral (1.0), 1.0 is soft (1.5), -1.0 is aggressive (0.5)
    # User expects 'negative' behavior (softening) when sliding Right (positive)
    w_mapped = 1.0 + (whites * 0.5)

    # Map Blacks: user expects 'negative' behavior (lifting) when sliding Right (positive)
    # +1.0 slider should equal old +0.2 behavior.
    b_mapped = -blacks * 0.2

    inv_denom = (
        1.0 / (w_mapped - b_mapped) if abs(w_mapped - b_mapped) > 1e-6 else 1000000.0
    )
    abs_h = abs(highlights)

    if contrast >= 0.0:
        c_mult = 1.0 + contrast
    else:
        c_mult = 1.0 + (contrast * 0.5)

    s_mult = 1.0 + saturation

    clipped_shadows = 0
    clipped_highlights = 0
    pixel_sum = 0.0

    for r in prange(rows):
        for c in range(cols):
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]

            if c_mult != 1.0:
                r_val = (r_val - 0.18) * c_mult + 0.18
                g_val = (g_val - 0.18) * c_mult + 0.18
                b_val = (b_val - 0.18) * c_mult + 0.18

            lum = LUMA_R * r_val + LUMA_G * g_val + LUMA_B * b_val

            r_val = (r_val - b_mapped) * inv_denom
            g_val = (g_val - b_mapped) * inv_denom
            b_val = (b_val - b_mapped) * inv_denom
            lum = (lum - b_mapped) * inv_denom

            if shadows != 0.0 or highlights != 0.0 or s_mult != 1.0:
                if shadows != 0.0:
                    clim = max(0.0, min(1.0, lum))
                    s_mask = (1.0 - clim) * (1.0 - clim)
                    factor = 1.0 + shadows * s_mask
                    r_val *= factor
                    g_val *= factor
                    b_val *= factor
                    lum *= factor

                if highlights != 0.0:
                    if highlights < 0.0:
                        h_mask = max(0.0, lum) ** 2
                        factor = 1.0 + abs_h * h_mask
                        inv_f = 1.0 / factor
                        r_val *= inv_f
                        g_val *= inv_f
                        b_val *= inv_f
                        lum *= inv_f
                    else:
                        clim = max(0.0, min(1.0, lum))
                        h_term = highlights * clim * clim
                        inv_h = 1.0 - h_term
                        r_val = r_val * inv_h + h_term
                        g_val = g_val * inv_h + h_term
                        b_val = b_val * inv_h + h_term
                        lum = lum * inv_h + h_term

                if s_mult != 1.0:
                    c_lum = max(0.0, min(1.0, lum))
                    r_val = c_lum + (r_val - c_lum) * s_mult
                    g_val = c_lum + (g_val - c_lum) * s_mult
                    b_val = c_lum + (b_val - c_lum) * s_mult

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

            if apply_gamma:
                r_val = _linear_to_srgb(max(0.0, r_val))
                g_val = _linear_to_srgb(max(0.0, g_val))
                b_val = _linear_to_srgb(max(0.0, b_val))

            r_val = max(0.0, min(1.0, r_val))
            g_val = max(0.0, min(1.0, g_val))
            b_val = max(0.0, min(1.0, b_val))

            img[r, c, 0] = r_val
            img[r, c, 1] = g_val
            img[r, c, 2] = b_val
            pixel_sum += (r_val + g_val + b_val) * 0.33333333

    return clipped_shadows, clipped_highlights, pixel_sum


@njit(fastmath=True, cache=True, parallel=True)
def float32_to_uint8(img):
    """
    Fused clip + scale + cast: converts float32 [0,1] image to uint8 [0,255].
    Avoids two intermediate array allocations from np.clip() and *255.
    """
    rows, cols, channels = img.shape
    out = np.empty((rows, cols, channels), dtype=np.uint8)

    for r in prange(rows):
        for c in range(cols):
            for ch in range(channels):
                val = img[r, c, ch]
                # Clamp to [0, 1] and scale to [0, 255]
                if val <= 0.0:
                    out[r, c, ch] = 0
                elif val >= 1.0:
                    out[r, c, ch] = 255
                else:
                    out[r, c, ch] = np.uint8(val * 255.0 + 0.5)

    return out
