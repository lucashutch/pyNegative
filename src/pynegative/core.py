#!/usr/bin/env python3
import json
import time
import math
import logging
import re
from pathlib import Path

from datetime import datetime
from functools import lru_cache
from io import BytesIO

import numpy as np
import rawpy
from PIL import Image, ImageOps

import cv2

# Import Numba kernels (required)
from .utils.numba_kernels import (
    tone_map_kernel,
    sharpen_kernel,
    bilateral_kernel_yuv,
    dehaze_recovery_kernel,
    dark_channel_kernel,
    nl_means_numba,
    nl_means_numba_multichannel,
)

# Configure logger for this module
logger = logging.getLogger(__name__)

RAW_EXTS = {
    ".cr2",
    ".cr3",
    ".dng",
    ".arw",
    ".nef",
    ".nrw",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
}
STD_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".heic", ".heif"}
SUPPORTED_EXTS = tuple(RAW_EXTS | STD_EXTS)

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False


# ---------------- Tone Mapping ----------------
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
):
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

        # Ensure calculate_stats is boolean
        calculate_stats = bool(calculate_stats)

    except Exception as e:
        logger.error(f"Error sanitizing apply_tone_map parameters: {e}")
        # Re-raise or return default if parameters are critical
        return img, {}  # Or raise e, depending on desired error handling

    """
    Applies White Balance -> Exposure -> Levels -> Tone EQ -> Saturation -> Base Curve
    Optimized for performance with in-place operations and minimal allocations.
    """
    start_time = time.perf_counter()
    # Create a single copy at the start to protect the input array
    img = img.copy()
    total_pixels = img.shape[0] * img.shape[1]

    # --- NUMBA OPTIMIZATION ---
    # Ensure float32 and C-contiguous for Numba
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

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
        f"Tone Map (Numba): Size: {img.shape[1]}x{img.shape[0]} | Time: {elapsed:.2f}ms"
    )
    return img, stats


def calculate_auto_exposure(img):
    """
    Analyzes image histogram to determine auto-exposure and contrast settings.
    Returns a dict with recommended {exposure, blacks, whites}.
    """
    # 1. Calculate luminance
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

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


def apply_geometry(img, rotate=0.0, crop=None, flip_h=False, flip_v=False):
    """
    Applies geometric transformations: Flip -> Rotation -> Crop.

    Args:
        pil_img: PIL Image
        rotate: float (degrees, CCW. Negative values rotate clockwise)
        crop: tuple (left, top, right, bottom) as normalized coordinates (0.0-1.0).
              The crop coordinates are relative to the FLIPPED and ROTATED image.
        flip_h: bool, mirror horizontally
        flip_v: bool, mirror vertically
    """
    if img is None:
        return None

    # 1. Apply Flip
    if flip_h or flip_v:
        # flipCode: 0 for x-axis, 1 for y-axis, -1 for both
        flip_code = -1 if (flip_h and flip_v) else (1 if flip_h else 0)
        img = cv2.flip(img, flip_code)

    # 2. Apply Rotation
    if abs(rotate) > 0.01:
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        # Use INTER_CUBIC for rotation quality
        M = cv2.getRotationMatrix2D(center, rotate, 1.0)
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        img = cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    # 3. Apply Crop
    if crop is not None:
        h, w = img.shape[:2]
        c_left, c_top, c_right, c_bottom = crop
        x1, y1 = int(c_left * w), int(c_top * h)
        x2, y2 = int(c_right * w), int(c_bottom * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]

    return img


def calculate_max_safe_crop(w, h, angle_deg, aspect_ratio=None):
    """
    Calculates the maximum normalized crop (l, t, r, b) that fits inside
    a rotated rectangle of size (w, h) rotated by angle_deg.

    If aspect_ratio is provided, the result will respect it.
    Otherwise, it uses the original image aspect ratio (w/h).

    Returns (l, t, r, b) as normalized coordinates relative to
    the EXPANDED rotated canvas.
    """
    phi = abs(math.radians(angle_deg))

    if phi < 1e-4:
        return (0.0, 0.0, 1.0, 1.0)

    if aspect_ratio is None:
        aspect_ratio = w / h

    # Formula for largest axis-aligned rectangle of aspect ratio 'AR'
    # inside a rotated rectangle of size (w, h) and angle 'phi'.

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # We need to satisfy:
    # 1. w_prime * cos + h_prime * sin <= w
    # 2. w_prime * sin + h_prime * cos <= h
    # and w_prime = h_prime * aspect_ratio

    h_prime_1 = w / (aspect_ratio * cos_phi + sin_phi)
    h_prime_2 = h / (aspect_ratio * sin_phi + cos_phi)

    h_prime = min(h_prime_1, h_prime_2)
    w_prime = h_prime * aspect_ratio

    # Expanded canvas size
    W = w * cos_phi + h * sin_phi
    H = w * sin_phi + h * cos_phi

    # Normalized dimensions relative to expanded canvas
    nw = w_prime / W
    nh = h_prime / H

    # Center it
    c_left = (1.0 - nw) / 2
    c_top = (1.0 - nh) / 2
    c_right = c_left + nw
    c_bottom = c_top + nh

    # Clamp to safe range just in case of float errors
    return (
        float(max(0.0, min(1.0, c_left))),
        float(max(0.0, min(1.0, c_top))),
        float(max(0.0, min(1.0, c_right))),
        float(max(0.0, min(1.0, c_bottom))),
    )


@lru_cache(maxsize=4)
def open_raw(path, half_size=False, output_bps=8):
    """
    Opens a RAW or standard image file.
    Args:
        path: File path (str or Path)
        half_size: If True, decodes at 1/2 resolution (1/4 pixels) for speed.
        output_bps: Bit depth of the output image (8 or 16).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in STD_EXTS:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            if half_size:
                img.thumbnail((img.width // 2, img.height // 2))
            rgb = np.array(img)
            return rgb.astype(np.float32) / 255.0

    path_str = str(path)
    with rawpy.imread(path_str) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=half_size,
            no_auto_bright=True,  # Disable auto-brighten to allow manual recovery
            bright=1.0,
            output_bps=output_bps,
        )

    # Normalize to 0.0-1.0 range
    if output_bps == 16:
        return rgb.astype(np.float32) / 65535.0
    return rgb.astype(np.float32) / 255.0


def extract_thumbnail(path):
    """
    Attempts to extract an embedded thumbnail.
    Falls back to a fast, half-size RAW conversion if no thumbnail exists.
    Returns a PIL Image or None on failure.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in STD_EXTS:
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Error opening standard image thumbnail for {path}: {e}")
            return None

    path_str = str(path)
    try:
        with rawpy.imread(path_str) as raw:
            try:
                thumb = raw.extract_thumb()
            except rawpy.LibRawNoThumbnailError:
                thumb = None

            # If we found a JPEG thumbnail
            if thumb and thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(BytesIO(thumb.data))
                return ImageOps.exif_transpose(img)

            # Fallback: fast postprocess (half_size=True is very fast)
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # 1/4 resolution
                no_auto_bright=False,
                output_bps=8,
            )
            return Image.fromarray(rgb)

    except Exception as e:
        logger.error(f"Error extracting thumbnail for {path}: {e}")
        return None


def sharpen_image(img, radius, percent, method="High Quality"):
    """Advanced sharpening that operates on NumPy float32 arrays."""
    if img is None:
        return None

    try:
        # Sanitize radius and percent
        radius = float(radius if radius is not None else 0.5)
        percent = float(percent if percent is not None else 0.0)

        if percent <= 0 or radius <= 0:
            return img
    except (ValueError, TypeError):
        return img

    start_time = time.perf_counter()

    # Ensure float32
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0

    h, w = img.shape[:2]
    size_str = f" | Size: {w}x{h}"

    try:
        # Setup: Blur and Edges (CPU)
        blur = cv2.GaussianBlur(img, (0, 0), radius)
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 1. Numba JIT (Primary)
        # Numba kernel is in-place, so we copy for the 'sharpened' version
        sharpened = img.copy()
        sharpen_kernel(sharpened, blur, percent)
        result = np.where(edges[:, :, np.newaxis] > 0, sharpened, img)
        backend = "Numba JIT"

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Sharpen: High Quality ({backend}) | Radius: {radius:.2f} | Percent: {percent:.1f}%{size_str} | Time: {elapsed:.2f}ms"
        )
        return np.clip(result, 0, 1.0)
    except Exception as e:
        logger.error(f"High Quality Sharpen failed: {e}")
        return img


def _apply_nl_means_path(img_array, l_str, c_str, method):
    """Internal helper to apply Numba NL-Means denoising path."""
    yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)

    # Map strength 0-50 to h parameters
    h_y = (l_str / 50.0) * 0.01
    h_uv = (c_str / 50.0) * 50.0

    p_size, s_size = 5, 13
    if "Fast+" in method:
        p_size, s_size = 3, 5
    elif "Fast" in method:
        p_size, s_size = 5, 9

    if "Hybrid" in method:
        # Hybrid: Fast+ for Luma (3, 5), Fast for Chroma (5, 9)
        y_denoised = (
            nl_means_numba(y, h=h_y, patch_size=3, search_size=5) if l_str > 0 else y
        )
        if c_str > 0:
            uv_stack = np.ascontiguousarray(yuv[:, :, 1:])
            uv_denoised = nl_means_numba_multichannel(
                uv_stack, h=(h_uv, h_uv), patch_size=5, search_size=9
            )
            denoised_yuv = cv2.merge(
                [y_denoised, uv_denoised[:, :, 0], uv_denoised[:, :, 1]]
            )
        else:
            denoised_yuv = cv2.merge([y_denoised, u, v])
    elif l_str > 0 and c_str > 0:
        denoised_yuv = nl_means_numba_multichannel(
            yuv, h=(h_y, h_uv, h_uv), patch_size=p_size, search_size=s_size
        )
    elif l_str > 0:
        y_denoised = nl_means_numba(y, h=h_y, patch_size=p_size, search_size=s_size)
        denoised_yuv = cv2.merge([y_denoised, u, v])
    elif c_str > 0:
        uv_stack = np.ascontiguousarray(yuv[:, :, 1:])
        uv_denoised = nl_means_numba_multichannel(
            uv_stack, h=(h_uv, h_uv), patch_size=p_size, search_size=s_size
        )
        denoised_yuv = cv2.merge([y, uv_denoised[:, :, 0], uv_denoised[:, :, 1]])
    else:
        denoised_yuv = yuv

    return cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2RGB)


def _apply_bilateral_path(img_array, l_str, c_str):
    """Internal helper to apply Numba Bilateral denoising path."""
    s_scale = 1.0 / 255.0
    yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)

    sigma_color_y = max(1e-6, l_str * 0.4 * s_scale)
    sigma_space_y = 0.5 + (l_str / 100.0)
    sigma_color_uv = max(1e-6, c_str * 4.5 * s_scale)
    sigma_space_uv = 2.0 + (c_str / 10.0)

    img_yuv_denoised = bilateral_kernel_yuv(
        yuv,
        l_str,
        sigma_color_y,
        sigma_space_y,
        sigma_color_uv,
        sigma_space_uv,
    )
    return cv2.cvtColor(img_yuv_denoised, cv2.COLOR_YUV2RGB)


def de_noise_image(
    img, luma_strength=0, chroma_strength=0, method="High Quality", zoom=None
):
    """
    Numba-accelerated de-noising strictly for NumPy float32 arrays.
    """
    if img is None:
        return None

    l_str = float(luma_strength)
    c_str = float(chroma_strength)

    if l_str <= 0 and c_str <= 0:
        return img

    # Ensure float32
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0

    h, w = img.shape[:2]
    zoom_str = f" | Zoom: {zoom * 100:.0f}%" if zoom is not None else ""
    size_str = f" | Size: {w}x{h}"

    start_time = time.perf_counter()
    denoised = None
    backend = "Unknown"
    method_name = method

    try:
        if method.startswith("NLMeans (Numba"):
            backend = "Numba JIT"
            # Get clean variant name for logging
            parts = method.split(" ")
            variant = parts[-1].replace(")", "") if len(parts) > 1 else "Full"
            method_name = f"NL-Means ({variant})"

            denoised = _apply_nl_means_path(img, l_str, c_str, method)
        else:
            backend = "Numba JIT"
            method_name = "Bilateral"
            denoised = _apply_bilateral_path(img, l_str, c_str)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Denoise: {method_name} ({backend}) | Luma: {l_str:.2f} Chroma: {c_str:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )

    except Exception as e:
        logger.error(f"Core Denoise failed: {e}")

    if denoised is not None:
        return np.clip(denoised, 0, 1)

    return img


def de_haze_image(img, strength, zoom=None, fixed_atmospheric_light=None):
    """
    Applies a fast dehazing algorithm based on Dark Channel Prior.
    Strength: 0.0 to 1.0.
    Values > 1.0 are automatically normalized (divided by 50.0) for backward compatibility,
    but callers should prefer passing normalized 0.0-1.0 values.
    Returns (result, atmospheric_light)
    """
    if img is None:
        return None, None

    try:
        # Sanitize strength
        if strength is None:
            strength = 0.0
        strength = float(strength)
        if strength <= 0:
            return img, fixed_atmospheric_light
        # Legacy: UI used to pass 0-50, normalize to 0-1 for core
        if strength > 1.0:
            strength /= 50.0
        strength = min(1.0, max(0.0, strength))
    except (ValueError, TypeError):
        return img, fixed_atmospheric_light

    # Ensure float32
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0

    img_array = img

    h_img, w_img = img_array.shape[:2]
    zoom_str = f" | Zoom: {zoom * 100:.0f}%" if zoom is not None else ""
    size_str = f" | Size: {w_img}x{h_img}"
    start_time = time.perf_counter()

    try:
        # 1. Dark Channel estimation
        # Scale kernel size relative to image width for consistency across resolutions
        # Base kernel size 15 for a ~2048px preview
        kernel_size = max(3, int(15 * (w_img / 2048.0)))
        # Ensure kernel size is odd for GaussianBlur later
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Atmospheric light estimation
        if fixed_atmospheric_light is not None:
            atmospheric_light = fixed_atmospheric_light
        else:
            # 1. Dark Channel estimation
            dark_channel = dark_channel_kernel(img_array)

            # Morphology (Erode) - OpenCV CPU is very fast for this
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_size, kernel_size)
            )
            dark_channel = cv2.erode(dark_channel, kernel)

            # 2. Atmospheric light estimation
            # Top 0.1% brightest pixels in the dark channel
            num_pixels = dark_channel.size
            num_brightest = max(1, num_pixels // 1000)
            indices = np.argpartition(dark_channel.flatten(), -num_brightest)[
                -num_brightest:
            ]

            # Of these pixels, pick the brightest in the original image
            brightest_pixels = img_array.reshape(-1, 3)[indices]
            atmospheric_light = np.max(brightest_pixels, axis=0)

        # 3. Transmission map estimation
        # t(x) = 1 - omega * min_c(I_c(x) / A_c)
        omega = 0.95 * (strength / 50.0 if strength > 1.0 else strength)

        # Avoid division by zero in A
        a_safe = np.maximum(atmospheric_light, 0.001)

        normalized_img = img_array / a_safe
        dark_normalized = dark_channel_kernel(normalized_img)

        # Morphology (Erode) - OpenCV CPU is very fast for this
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_normalized = cv2.erode(dark_normalized, kernel)
        transmission = 1.0 - omega * dark_normalized

        # Refine transmission map
        transmission = cv2.GaussianBlur(
            transmission, (kernel_size * 2 + 1, kernel_size * 2 + 1), 0
        )

        transmission = np.maximum(transmission, 0.1)  # Lower bound for transmission

        # 4. Recover radiance
        # J(x) = (I(x) - A) / max(t(x), t0) + A
        backend_rec = "Numba JIT"
        result = dehaze_recovery_kernel(img_array, transmission, atmospheric_light)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Dehaze: ({backend_rec}) | "
            f"Strength: {strength:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )

        return result, atmospheric_light

    except Exception as e:
        logger.error(f"Dehaze failed: {e}")
        return img, None


def save_image(pil_img, output_path, quality=95):
    output_path = Path(output_path)
    fmt = output_path.suffix.lower()
    if fmt in (".jpeg", ".jpg"):
        pil_img.save(output_path, quality=quality)
    elif fmt in (".heif", ".heic"):
        if not HEIF_SUPPORTED:
            raise RuntimeError("HEIF requested but pillow-heif not installed.")
        pil_img.save(output_path, format="HEIF", quality=quality)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# ---------------- Sidecar Files ----------------
SIDECAR_DIR = ".pyNegative"


def get_sidecar_path(raw_path: str | Path) -> Path:
    """
    Returns the Path object to the sidecar JSON file for a given RAW file.
    Sidecars are stored in a hidden .pyNegative directory local to the image.
    """
    raw_path = Path(raw_path)
    return raw_path.parent / SIDECAR_DIR / f"{raw_path.name}.json"


def save_sidecar(raw_path: str | Path, settings: dict) -> None:
    """
    Saves edit settings to a JSON sidecar file.
    """
    sidecar_path = get_sidecar_path(raw_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure rating is present
    if "rating" not in settings:
        settings["rating"] = 0

    data = {
        "version": "1.0",
        "last_modified": time.time(),
        "raw_path": str(raw_path),
        "settings": settings,
    }

    with open(sidecar_path, "w") as f:
        json.dump(data, f, indent=4)


def load_sidecar(raw_path: str | Path) -> dict | None:
    """
    Loads edit settings from a JSON sidecar file if it exists.
    Returns the settings dict or None.
    """
    sidecar_path = get_sidecar_path(raw_path)
    if not sidecar_path.exists():
        return None

    try:
        with open(sidecar_path, "r") as f:
            data = json.load(f)
            settings = data.get("settings")
            if settings:
                if "rating" not in settings:
                    settings["rating"] = 0
            return settings
    except Exception as e:
        logger.error(f"Error loading sidecar {sidecar_path}: {e}")
        return None


def rename_sidecar(old_raw_path: str | Path, new_raw_path: str | Path) -> None:
    """
    Renames a sidecar file when the original RAW is moved/renamed.
    """
    old_sidecar = get_sidecar_path(old_raw_path)
    new_sidecar = get_sidecar_path(new_raw_path)

    if old_sidecar.exists():
        new_sidecar.parent.mkdir(parents=True, exist_ok=True)
        old_sidecar.rename(new_sidecar)


def get_sidecar_mtime(raw_path: str | Path) -> float | None:
    """
    Returns the last modified time of the sidecar file if it exists.
    """
    sidecar_path = get_sidecar_path(raw_path)
    if sidecar_path.exists():
        return sidecar_path.stat().st_mtime
    return None


def get_exif_capture_date(raw_path: str | Path) -> str | None:
    """
    Extracts the capture date from RAW or standard image file EXIF data.

    Returns the date as a string in YYYY-MM-DD format, or None if unavailable.
    Falls back to file modification date if EXIF date is not found.
    """

    raw_path = Path(raw_path)
    ext = raw_path.suffix.lower()

    try:
        if ext in STD_EXTS:
            with Image.open(raw_path) as img:
                exif = img.getexif()
                if exif:
                    # 306 = DateTime, 36867 = DateTimeOriginal
                    for tag in (36867, 306):
                        date_str = exif.get(tag)
                        if date_str and isinstance(date_str, str):
                            # Format: "YYYY:MM:DD HH:MM:SS"
                            try:
                                parts = date_str.split(" ")[0].split(":")
                                if len(parts) == 3:
                                    return f"{parts[0]}-{parts[1]}-{parts[2]}"
                            except Exception:
                                pass

        with rawpy.imread(str(raw_path)) as raw:
            # Try to extract EXIF DateTimeOriginal
            # rawpy stores EXIF data that we can parse
            try:
                # Access the raw data structure
                if hasattr(raw, "raw_image") and hasattr(raw, "extract_exif"):
                    exif_data = raw.extract_exif()
                    if exif_data:
                        # Parse DateTimeOriginal from EXIF
                        # Format in EXIF is typically: "2024:01:15 14:30:00"
                        exif_str = exif_data.decode("utf-8", errors="ignore")

                        # Search for date patterns in EXIF
                        date_patterns = [
                            r"DateTimeOriginal\s*\x00*\s*(\d{4}):(\d{2}):(\d{2})",
                            r"DateTime\s*\x00*\s*(\d{4}):(\d{2}):(\d{2})",
                            r"(\d{4}):(\d{2}):(\d{2})\s+(\d{2}):(\d{2}):(\d{2})",
                        ]

                        for pattern in date_patterns:
                            match = re.search(pattern, exif_str)
                            if match:
                                year, month, day = match.groups()[:3]
                                return f"{year}-{month}-{day}"

            except Exception as e:
                logger.debug(f"Error extracting EXIF from {raw_path}: {e}")

        # Fallback: use file modification time
        mtime = raw_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    except Exception as e:
        logger.error(f"Error reading file {raw_path}: {e}")
        return None


def format_date(timestamp: float) -> str:
    """Formats a timestamp as YYYY-MM-DD."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
