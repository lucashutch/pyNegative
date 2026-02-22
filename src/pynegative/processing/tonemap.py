import logging
import math
import time

import numpy as np

from ..utils.numba_kernels import preprocess_kernel, tone_map_kernel
from .constants import LUMA_B, LUMA_G, LUMA_R

logger = logging.getLogger(__name__)


def apply_preprocess(
    img,
    temperature=0.0,
    tint=0.0,
    exposure=0.0,
    vignette_k1=0.0,
    vignette_k2=0.0,
    vignette_k3=0.0,
    vignette_cx=0.0,
    vignette_cy=0.0,
    full_width=1.0,
    full_height=1.0,
):
    """
    Applies Vignette -> White Balance -> Exposure in linear space.
    This should be called BEFORE denoising for linear-aware noise reduction.
    Operates in-place on a copy of the input array.
    """
    if img is None:
        return None

    def safe_float(val, default):
        try:
            if val is None:
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    temperature = safe_float(temperature, 0.0)
    tint = safe_float(tint, 0.0)
    exposure = safe_float(exposure, 0.0)
    vignette_k1 = safe_float(vignette_k1, 0.0)
    vignette_k2 = safe_float(vignette_k2, 0.0)
    vignette_k3 = safe_float(vignette_k3, 0.0)
    vignette_cx = safe_float(vignette_cx, 0.0)
    vignette_cy = safe_float(vignette_cy, 0.0)
    full_width = safe_float(full_width, 1.0)
    full_height = safe_float(full_height, 1.0)

    start_time = time.perf_counter()
    img = img.copy()

    t_scale = 0.4
    tint_scale = 0.2
    r_mult = np.exp(temperature * t_scale - tint * (tint_scale / 2))
    g_mult = np.exp(tint * tint_scale)
    b_mult = np.exp(-temperature * t_scale - tint * (tint_scale / 2))

    preprocess_kernel(
        img,
        r_mult,
        g_mult,
        b_mult,
        exposure,
        vignette_k1,
        vignette_k2,
        vignette_k3,
        vignette_cx,
        vignette_cy,
        full_width,
        full_height,
    )

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(
        f"Preprocess (WB+Exp+Vig): Size: {img.shape[1]}x{img.shape[0]} | Time: {elapsed:.2f}ms"
    )
    return img


def apply_tone_map(
    img,
    contrast=0.0,
    blacks=0.0,
    whites=0.0,
    shadows=0.0,
    highlights=0.0,
    saturation=0.0,
    calculate_stats=True,
    apply_gamma=True,
):
    """
    Applies Contrast -> Levels -> Shadows/Highlights -> Saturation -> Gamma.
    Expects pre-processed data (WB, exposure, vignette already applied).
    Optimized for performance with in-place operations and minimal allocations.
    """
    try:
        if img is None:
            return None, None

        def safe_float(val, default):
            try:
                if val is None:
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default

        contrast = safe_float(contrast, 0.0)
        blacks = safe_float(blacks, 0.0)
        whites = safe_float(whites, 0.0)
        shadows = safe_float(shadows, 0.0)
        highlights = safe_float(highlights, 0.0)
        saturation = safe_float(saturation, 0.0)

        calculate_stats = bool(calculate_stats)
        apply_gamma = bool(apply_gamma)

    except Exception as e:
        logger.error(f"Error sanitizing apply_tone_map parameters: {e}")
        return img, {}

    start_time = time.perf_counter()
    img = img.copy()
    total_pixels = img.shape[0] * img.shape[1]

    clipped_shadows, clipped_highlights, pixel_sum = tone_map_kernel(
        img,
        contrast,
        blacks,
        whites,
        shadows,
        highlights,
        saturation,
        apply_gamma,
    )

    if calculate_stats:
        stats = {
            "pct_shadows_clipped": clipped_shadows / total_pixels * 100,
            "pct_highlights_clipped": clipped_highlights / total_pixels * 100,
            "mean": pixel_sum / total_pixels,
        }
    else:
        stats = {}

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(
        f"Tone Map: Size: {img.shape[1]}x{img.shape[0]} | Time: {elapsed:.2f}ms"
    )
    return img, stats


def calculate_auto_exposure(img):
    """
    Analyzes image histogram to determine auto-exposure and contrast settings.
    Returns a dict with recommended {exposure, blacks, whites}.
    """
    h, w = img.shape[:2]
    stride = max(1, int(np.sqrt(h * w / (1000 * 1000))))
    img_small = img[::stride, ::stride]

    lum = (
        LUMA_R * img_small[:, :, 0]
        + LUMA_G * img_small[:, :, 1]
        + LUMA_B * img_small[:, :, 2]
    )

    p98 = np.percentile(lum, 98)
    if p98 < 0.001:
        p98 = 0.001

    exposure = math.log2(0.85 / p98)
    exposure = float(np.clip(exposure, 0.5, 4.0))

    base_blacks = 0.01
    base_whites = 0.0  # Neutral

    return {
        "exposure": exposure,
        "blacks": float(base_blacks),
        "whites": float(base_whites),
        "highlights": 0.0,
        "shadows": 0.0,
        "saturation": 0.05,  # 1.05 -> 0.05
    }


def calculate_auto_wb(img):
    """
    Calculates relative temperature and tint to neutralize the image (Gray World).
    """
    r_avg = np.mean(img[:, :, 0])
    g_avg = np.mean(img[:, :, 1])
    b_avg = np.mean(img[:, :, 2])

    if r_avg < 1e-6:
        r_avg = 1e-6
    if g_avg < 1e-6:
        g_avg = 1e-6
    if b_avg < 1e-6:
        b_avg = 1e-6

    temp = np.log(b_avg / r_avg) / 0.8
    tint = np.log((r_avg * b_avg) / (g_avg**2)) / 0.6

    return {
        "temperature": float(np.clip(temp, -1.0, 1.0)),
        "tint": float(np.clip(tint, -1.0, 1.0)),
    }
