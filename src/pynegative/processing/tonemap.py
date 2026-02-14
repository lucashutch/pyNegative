import logging
import time
import math
import numpy as np
from .constants import LUMA_R, LUMA_G, LUMA_B
from ..utils.numba_kernels import tone_map_kernel

logger = logging.getLogger(__name__)


def apply_tone_map(
    img,
    temperature=0.0,
    tint=0.0,
    exposure=0.0,
    contrast=1.0,
    blacks=0.0,
    whites=1.0,
    shadows=0.0,
    highlights=0.0,
    saturation=1.0,
    calculate_stats=True,
    apply_gamma=True,  # New parameter (Pillar A)
):
    """
    Applies White Balance -> Exposure -> Levels -> Tone EQ -> Saturation -> Base Curve
    Optimized for performance with in-place operations and minimal allocations.
    """
    try:
        if img is None:
            return None, None

        # Sanitize all inputs to floats with safe defaults
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
        contrast = safe_float(contrast, 1.0)
        blacks = safe_float(blacks, 0.0)
        whites = safe_float(whites, 1.0)
        shadows = safe_float(shadows, 0.0)
        highlights = safe_float(highlights, 0.0)
        saturation = safe_float(saturation, 1.0)

        # Ensure calculate_stats and apply_gamma are boolean
        calculate_stats = bool(calculate_stats)
        apply_gamma = bool(apply_gamma)

    except Exception as e:
        logger.error(f"Error sanitizing apply_tone_map parameters: {e}")
        # Re-raise or return default if parameters are critical
        return img, {}  # Or raise e, depending on desired error handling

    start_time = time.perf_counter()
    # Create a single copy at the start to protect the input array
    img = img.copy()
    total_pixels = img.shape[0] * img.shape[1]

    # --- NUMBA OPTIMIZATION ---
    # float32 and C-contiguous are guaranteed by the pipeline contract

    # Prepare WB multipliers
    t_scale = 0.4
    tint_scale = 0.2
    r_mult = np.exp(temperature * t_scale - tint * (tint_scale / 2))
    g_mult = np.exp(tint * tint_scale)
    b_mult = np.exp(-temperature * t_scale - tint * (tint_scale / 2))

    # Exposure multiplier
    exp_mult = 2.0**exposure

    # Call kernel (in-place)
    clipped_shadows, clipped_highlights, pixel_sum = tone_map_kernel(
        img,
        exp_mult,
        contrast,
        blacks,
        whites,
        shadows,
        highlights,
        saturation,
        r_mult,
        g_mult,
        b_mult,
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
    # Use a downsampled version for performance
    h, w = img.shape[:2]
    stride = max(1, int(np.sqrt(h * w / (1000 * 1000))))  # Target ~1MP for analysis
    img_small = img[::stride, ::stride]

    # 1. Calculate luminance
    lum = (
        LUMA_R * img_small[:, :, 0]
        + LUMA_G * img_small[:, :, 1]
        + LUMA_B * img_small[:, :, 2]
    )

    # Target: 98th percentile should be at ~0.85 (bright but not clipped)
    # This works well for linear RAW data.
    p98 = np.percentile(lum, 98)
    if p98 < 0.001:
        p98 = 0.001

    exposure = math.log2(0.85 / p98)
    # Clamp to reasonable range for auto-start
    exposure = float(np.clip(exposure, 0.5, 4.0))

    # Standard Contrast & Levels
    base_blacks = 0.01
    base_whites = 0.95  # Slight boost to peaks in our new curve logic

    return {
        "exposure": exposure,
        "blacks": float(base_blacks),
        "whites": float(base_whites),
        "highlights": 0.0,
        "shadows": 0.0,
        "saturation": 1.05,  # 5% Saturation boost (Standard Profile)
    }


def calculate_auto_wb(img):
    """
    Calculates relative temperature and tint to neutralize the image (Gray World).
    """
    # Calculate channel means
    r_avg = np.mean(img[:, :, 0])
    g_avg = np.mean(img[:, :, 1])
    b_avg = np.mean(img[:, :, 2])

    # Avoid division by zero
    if r_avg < 1e-6:
        r_avg = 1e-6
    if g_avg < 1e-6:
        g_avg = 1e-6
    if b_avg < 1e-6:
        b_avg = 1e-6

    # Log space calculation for relative offsets
    # target_g = (r_avg * r_mult + g_avg * g_mult + b_avg * b_mult) / 3

    # Simple Gray World: find multipliers to make R=G=B
    # In our model:
    # r_mult = exp(temp * 0.4 - tint * 0.1)
    # g_mult = exp(tint * 0.2)
    # b_mult = exp(-temp * 0.4 - tint * 0.1)

    # log(g_mult) = tint * 0.2  => tint = log(g_mult) / 0.2
    # log(r_mult/b_mult) = 0.8 * temp => temp = log(r_mult/b_mult) / 0.8

    # Target: make R = G = B
    # log(r) + temp * 0.4 - tint * 0.1 = log(g) + tint * 0.2
    # log(b) - temp * 0.4 - tint * 0.1 = log(g) + tint * 0.2

    # temp * 0.8 = log(b/r)
    # tint * 0.6 = log(r*b / g^2)

    temp = np.log(b_avg / r_avg) / 0.8
    tint = np.log((r_avg * b_avg) / (g_avg**2)) / 0.6

    # Clamp to slider range
    return {
        "temperature": float(np.clip(temp, -1.0, 1.0)),
        "tint": float(np.clip(tint, -1.0, 1.0)),
    }
