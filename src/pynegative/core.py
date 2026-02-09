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
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2

    # Note: OpenCV cache env var should be set in pynegative/__init__.py
    # before this module is imported
except ImportError:
    cv2 = None

# Configure logger for this module
logger = logging.getLogger(__name__)

# Try importing Numba kernels
try:
    from .utils.numba_kernels import (
        tone_map_kernel,
        sharpen_kernel,
        bilateral_kernel_yuv,
        dehaze_recovery_kernel,
        dark_channel_kernel,
        nl_means_numba,
        nl_means_numba_multichannel,
    )

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba kernels not available, falling back to NumPy")

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

    # --- NUMBA OPTIMIZATION START ---
    if NUMBA_AVAILABLE and img.dtype == np.float32 and img.flags["C_CONTIGUOUS"]:
        try:
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

            # Skip NumPy Implementation
            if calculate_stats:
                stats = {
                    "pct_shadows_clipped": clipped_shadows / total_pixels * 100,
                    "pct_highlights_clipped": clipped_highlights / total_pixels * 100,
                    "mean": pixel_sum
                    / total_pixels,  # pixel_sum is already averaged per channel in kernel
                }
            else:
                stats = {}

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Tone Map (Numba): Size: {img.shape[1]}x{img.shape[0]} | Time: {elapsed:.2f}ms"
            )
            return img, stats

        except Exception as e:
            logger.warning(f"Numba kernel failed, falling back to NumPy: {e}")
            # Fall through to NumPy implementation

    # --- NUMPY FALLBACK (Original Implementation) ---

    # 0. White Balance (Relative Scaling)
    if temperature != 0.0 or tint != 0.0:
        t_scale = 0.4
        tint_scale = 0.2

        r_mult = np.exp(temperature * t_scale - tint * (tint_scale / 2))
        g_mult = np.exp(tint * tint_scale)
        b_mult = np.exp(-temperature * t_scale - tint * (tint_scale / 2))

        img[:, :, 0] *= r_mult
        img[:, :, 1] *= g_mult
        img[:, :, 2] *= b_mult

    # 1. Exposure (2^stops)
    if exposure != 0.0:
        img *= 2**exposure

    # 1.5 Contrast (Symmetric around 0.5)
    if contrast != 1.0:
        img -= 0.5
        img *= contrast
        img += 0.5

    # 2. Tone EQ (Blacks, Whites, Shadows & Highlights) and 3. Saturation
    # We calculate luminance once and reuse it.
    # IMPORTANT: We use unclipped luminance to allow for highlight recovery of >1.0 values.
    if (
        blacks != 0.0
        or whites != 1.0
        or shadows != 0.0
        or highlights != 0.0
        or saturation != 1.0
    ):
        # Calculate luminance (Rec. 709)
        lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        lum_3d = lum[:, :, np.newaxis]

        # 2.1 Blacks (Linear Offset/Crush)
        if blacks != 0.0:
            img -= blacks

        # 2.2 Whites (Linear Level Adjustment)
        if whites != 1.0:
            denom = whites - blacks
            if abs(denom) < 1e-6:
                denom = 1e-6
            img /= denom

        # 2.3 Shadows & Highlights
        if shadows != 0.0:
            s_mask = (1.0 - np.clip(lum_3d, 0, 1)) ** 2
            img *= 1.0 + shadows * s_mask

        if highlights != 0.0:
            if highlights < 0:
                # RECOVERY: Compress over-exposed highlights
                # Use unclipped luminance for the mask to distinguish clipped areas
                h_mask = np.maximum(lum_3d, 0) ** 2
                img /= 1.0 + abs(highlights) * h_mask
            else:
                # BOOST: Brighten highlights
                h_mask = np.clip(lum_3d, 0, 1) ** 2
                h_term = highlights * h_mask
                # Use a blend that caps at 1.0
                img = img * (1.0 - h_term) + h_term

        if saturation != 1.0:
            # Re-calculate luminance after tone adjustments for accurate saturation
            # Use clipped luminance for saturation to avoid color shifts in over-exposed areas
            curr_lum = (
                0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
            )
            np.clip(curr_lum, 0, 1, out=curr_lum)
            curr_lum_3d = curr_lum[:, :, np.newaxis]

            img -= curr_lum_3d
            img *= saturation
            img += curr_lum_3d

    # Stats and Clipping
    if calculate_stats:
        clipped_shadows = np.sum(img < 0.0)
        clipped_highlights = np.sum(img > 1.0)
        stats = {
            "pct_shadows_clipped": clipped_shadows / total_pixels * 100,
            "pct_highlights_clipped": clipped_highlights / total_pixels * 100,
            "mean": img.mean(),
        }
    else:
        stats = {}

    # Final Clip in-place
    np.clip(img, 0.0, 1.0, out=img)

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(f"Tone Map: (Group) | Time: {elapsed:.2f}ms")

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


def apply_geometry(pil_img, rotate=0.0, crop=None, flip_h=False, flip_v=False):
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
    # 0. Apply Flip
    if flip_h:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

    # 1. Apply Rotation
    if rotate != 0.0:
        # expand=True changes the image size to fit the rotated image
        # PIL rotates CCW by default. The user wants negative to be CW, so
        # positive is CCW. This matches PIL's behavior.
        pil_img = pil_img.rotate(rotate, resample=Image.BICUBIC, expand=True)

    # 2. Apply Crop
    if crop is not None:
        w, h = pil_img.size
        c_left, c_top, c_right, c_bottom = crop

        # Convert to pixels
        left = int(c_left * w)
        top = int(c_top * h)
        right = int(c_right * w)
        bottom = int(c_bottom * h)

        # Clamp
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        if right > left and bottom > top:
            pil_img = pil_img.crop((left, top, right, bottom))

    return pil_img


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
    """Advanced sharpening with support for both PIL and Numpy float32."""
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
    if isinstance(img, Image.Image):
        # Convert PIL to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        img_float = img_array.astype(np.float32) / 255.0
        was_pil = True
    else:
        # Assume Numpy array
        img_float = img
        if img_float.dtype != np.float32:
            img_float = img_float.astype(np.float32) / 255.0
        was_pil = False

    h, w = img_float.shape[:2]
    size_str = f" | Size: {w}x{h}"

    if method == "High Quality":
        try:
            if cv2 is None:
                raise ImportError("OpenCV not available")

            # Setup: Blur and Edges (CPU)
            blur = cv2.GaussianBlur(img_float, (0, 0), radius)
            gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # 1. Try Numba JIT
            if NUMBA_AVAILABLE:
                try:
                    # Numba kernel is in-place, so we copy for the 'sharpened' version
                    sharpened = img_float.copy()
                    sharpen_kernel(sharpened, blur, percent)
                    result = np.where(edges[:, :, np.newaxis] > 0, sharpened, img_float)
                    backend = "Numba JIT"
                except Exception as e:
                    logger.warning(f"Numba Sharpen failed: {e}")
                    sharpened = img_float + (img_float - blur) * (percent / 100.0)
                    result = np.where(edges[:, :, np.newaxis] > 0, sharpened, img_float)
                    backend = "NumPy"
            else:
                sharpened = img_float + (img_float - blur) * (percent / 100.0)
                result = np.where(edges[:, :, np.newaxis] > 0, sharpened, img_float)
                backend = "NumPy"

            if was_pil:
                result_array = np.clip(result * 255, 0, 255).astype(np.uint8)
                res = Image.fromarray(result_array)
            else:
                res = np.clip(result, 0, 1.0)

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Sharpen: High Quality ({backend}) | Radius: {radius:.2f} | Percent: {percent:.1f}%{size_str} | Time: {elapsed:.2f}ms"
            )
            return res
        except Exception as e:
            logger.error(f"High Quality Sharpen failed: {e}")

    # Fallback for PIL
    if was_pil:
        return img.filter(
            ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent))
        )

    # Fallback for Numpy (Basic Unsharp Mask)
    try:
        if cv2 is None:
            raise ImportError("OpenCV not available")

        # Convert radius to kernel size (must be odd)
        k_size = int(2 * math.ceil(radius * 2) + 1)
        if k_size % 2 == 0:
            k_size += 1
        blur = cv2.GaussianBlur(img_float, (k_size, k_size), radius)
        result = img_float + (img_float - blur) * (percent / 100.0)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Sharpen: Basic CPU Fallback | Radius: {radius:.2f} | Percent: {percent:.1f}%{size_str} | Time: {elapsed:.2f}ms"
        )
        return np.clip(result, 0, 1.0)
    except Exception:
        return img_float


def de_noise_image(
    img, luma_strength=0, chroma_strength=0, method="High Quality", zoom=None
):
    """
    Numba-accelerated de-noising with separate Luma/Chroma control.
    """
    if img is None:
        return None

    l_str = float(luma_strength)
    c_str = float(chroma_strength)

    if l_str <= 0 and c_str <= 0:
        return img

    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        was_pil = True
    else:
        img_array = img
        was_pil = False

    h, w = img_array.shape[:2]
    zoom_str = f" | Zoom: {zoom * 100:.0f}%" if zoom is not None else ""
    size_str = f" | Size: {w}x{h}"

    start_time = time.perf_counter()
    denoised = None
    try:
        if cv2 is None:
            raise ImportError("OpenCV not available")

        # Scaling factor for sigmaColor based on 0-1 range
        s_scale = 1.0 / 255.0

        if not isinstance(method, str):
            method = "High Quality"

        if method.startswith("NLMeans (Numba") and NUMBA_AVAILABLE:
            backend = "Numba JIT"
            # Safe split for method name
            parts = method.split(" ")
            variant = parts[-1].replace(")", "") if len(parts) > 1 else "Full"
            method_name = f"NL-Means ({variant})"
            yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv)

            # Map strength 0-50 to h parameters
            # Luma scaling: 0.01 max
            h_y = (l_str / 50.0) * 0.01
            # Chroma scaling: 50.0 max
            h_uv = (c_str / 50.0) * 50.0

            p_size, s_size = 5, 13
            if "Fast+" in method:
                p_size, s_size = 3, 5
            elif "Fast" in method:
                p_size, s_size = 5, 9

            # Even if method says "Y" or "UV", we now respect individual strengths.
            # However, if both > 0, we can use multichannel for speed.
            if "Hybrid" in method:
                # Hybrid: Fast+ for Luma (3, 5), Fast for Chroma (5, 9)
                y_denoised = y
                if l_str > 0:
                    y_denoised = nl_means_numba(
                        y, h=h_y, patch_size=3, search_size=5
                    )

                u_denoised, v_denoised = u, v
                if c_str > 0:
                    uv_stack = np.ascontiguousarray(yuv[:, :, 1:])
                    uv_denoised = nl_means_numba_multichannel(
                        uv_stack, h=(h_uv, h_uv), patch_size=5, search_size=9
                    )
                    u_denoised = uv_denoised[:, :, 0]
                    v_denoised = uv_denoised[:, :, 1]

                denoised_yuv = cv2.merge([y_denoised, u_denoised, v_denoised])

            elif l_str > 0 and c_str > 0:
                denoised_yuv = nl_means_numba_multichannel(
                    yuv, h=(h_y, h_uv, h_uv), patch_size=p_size, search_size=s_size
                )
            elif l_str > 0:
                y_denoised = nl_means_numba(
                    y, h=h_y, patch_size=p_size, search_size=s_size
                )
                denoised_yuv = cv2.merge([y_denoised, u, v])
            elif c_str > 0:
                uv_stack = np.ascontiguousarray(yuv[:, :, 1:])
                uv_denoised = nl_means_numba_multichannel(
                    uv_stack, h=(h_uv, h_uv), patch_size=p_size, search_size=s_size
                )
                denoised_yuv = cv2.merge(
                    [y, uv_denoised[:, :, 0], uv_denoised[:, :, 1]]
                )
            else:
                denoised_yuv = yuv

            denoised = cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2RGB)


        else:  # Default to High Quality (YUV Bilateral)
            method_name = "Bilateral"
            if NUMBA_AVAILABLE:
                backend = "Numba JIT"
                try:
                    # Prepare YUV for Numba
                    yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
                    sigma_color_y = max(1e-6, l_str * 0.4 * s_scale)
                    sigma_space_y = 0.5 + (l_str / 100.0)
                    sigma_color_uv = max(1e-6, c_str * 4.5 * s_scale)
                    sigma_space_uv = 2.0 + (c_str / 10.0)

                    img_yuv_denoised = bilateral_kernel_yuv(
                        yuv,
                        l_str,  # Passing luma strength
                        sigma_color_y,
                        sigma_space_y,
                        sigma_color_uv,
                        sigma_space_uv,
                    )
                    denoised = cv2.cvtColor(img_yuv_denoised, cv2.COLOR_YUV2RGB)
                except Exception as e:
                    logger.warning(f"Numba Bilateral failed: {e}")
                    # denoised is still None, fall through
            else:
                backend = "OpenCV CPU"
                # Fallback to OpenCV CPU
                yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
                y, u, v = cv2.split(yuv)

                if c_str > 0:
                    chroma_sigma_color = c_str * 4.5 * s_scale
                    chroma_sigma_space = 2.0 + (c_str / 10.0)
                    u = cv2.bilateralFilter(
                        u, 11, chroma_sigma_color, chroma_sigma_space
                    )
                    v = cv2.bilateralFilter(
                        v, 11, chroma_sigma_color, chroma_sigma_space
                    )

                if l_str > 0:
                    luma_sigma_color = l_str * 0.4 * s_scale
                    luma_sigma_space = 0.5 + (l_str / 100.0)
                    y = cv2.bilateralFilter(y, 3, luma_sigma_color, luma_sigma_space)

                denoised_yuv = cv2.merge([y, u, v])
                denoised = cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2RGB)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Denoise: {method_name} ({backend}) | Luma: {l_str:.2f} Chroma: {c_str:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )

        if denoised is not None:
            if was_pil:
                return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))
            else:
                return np.clip(denoised, 0, 1)

    except Exception as e:
        logger.error(f"OpenCV Denoise failed: {e}")

    # Fallback to PIL or Median
    if was_pil:
        size = int(strength / 5.0)  # Scale down strength for median
        if size < 3:
            size = 3 if strength > 0 else 0
        if size == 0:
            return img
        if size % 2 == 0:
            size += 1

        fallback_start = time.perf_counter()
        # Convert back to PIL for the filter
        pil_img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
        result = pil_img.filter(ImageFilter.MedianFilter(size=size))
        elapsed = (time.perf_counter() - fallback_start) * 1000
        logger.debug(
            f"Denoise: Fallback (PIL MedianFilter) | Strength: {strength:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )
        return result

    # Fallback for Numpy (Median Filter)
    try:
        if cv2 is None:
            raise ImportError("OpenCV not available")

        size = int(strength / 5.0)
        if size < 3:
            size = 3 if strength > 0 else 0
        if size == 0:
            return img_array
        if size % 2 == 0:
            size += 1

        fallback_start = time.perf_counter()
        # medianBlur expects uint8 or float32. We can use float32.
        denoised = cv2.medianBlur(img_array, size)
        elapsed = (time.perf_counter() - fallback_start) * 1000
        logger.debug(
            f"Denoise: Fallback (OpenCV MedianBlur) | Strength: {strength:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )
        return np.clip(denoised, 0, 1)
    except Exception:
        return img_array


def de_haze_image(img, strength, zoom=None, fixed_atmospheric_light=None):
    """
    Applies a fast dehazing algorithm based on Dark Channel Prior.
    Strength: 0.0 to 1.0 (though UI might pass 0-50).
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
        # UI might pass 0-50, normalize to 0-1 for core
        if strength > 1.0:
            strength /= 50.0
        strength = min(1.0, max(0.0, strength))
    except (ValueError, TypeError):
        return img, fixed_atmospheric_light

    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        was_pil = True
    else:
        img_array = img
        was_pil = False

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
            if NUMBA_AVAILABLE:
                dark_channel = dark_channel_kernel(img_array)
            else:
                dark_channel = np.min(img_array, axis=2)

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

        if NUMBA_AVAILABLE:
            normalized_img = img_array / a_safe
            dark_normalized = dark_channel_kernel(normalized_img)
        else:
            normalized_img = img_array / a_safe
            dark_normalized = np.min(normalized_img, axis=2)

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
        backend_rec = "NumPy"
        if NUMBA_AVAILABLE:
            try:
                result = dehaze_recovery_kernel(
                    img_array, transmission, atmospheric_light
                )
                backend_rec = "Numba JIT"
            except Exception as e:
                logger.warning(f"Numba Dehaze recovery failed: {e}")
                transmission_3d = transmission[:, :, np.newaxis]
                result = (
                    img_array - atmospheric_light
                ) / transmission_3d + atmospheric_light
                result = np.clip(result, 0, 1)
        else:
            transmission_3d = transmission[:, :, np.newaxis]
            result = (
                img_array - atmospheric_light
            ) / transmission_3d + atmospheric_light
            result = np.clip(result, 0, 1)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Dehaze: ({backend_rec}) | "
            f"Strength: {strength:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )

        if was_pil:
            return Image.fromarray((result * 255).astype(np.uint8)), atmospheric_light
        else:
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
