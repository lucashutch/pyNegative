import logging
import time

import cv2
import numpy as np

from ..utils.numba_kernels import (
    bilateral_kernel_yuv,
    dark_channel_kernel,
    dehaze_recovery_kernel,
    nl_means_numba,
    nl_means_numba_multichannel,
    sharpen_kernel,
)

logger = logging.getLogger(__name__)


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

    # Pipeline ensures float32 and C-contiguous

    h, w = img.shape[:2]
    size_str = f" | Size: {w}x{h}"

    try:
        # Setup: Blur and Edges (CPU)
        blur = cv2.GaussianBlur(img, (0, 0), radius)
        gray = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Kernel is in-place, so we copy for the 'sharpened' version
        sharpened = img.copy()
        sharpen_kernel(sharpened, blur, percent)
        result = np.where(edges[:, :, np.newaxis] > 0, sharpened, img)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Sharpen: Radius: {radius:.2f} | Percent: {percent:.1f}%{size_str} | Time: {elapsed:.2f}ms"
        )
        return np.clip(result, 0, 1.0)
    except Exception as e:
        logger.error(f"High Quality Sharpen failed: {e}")
        return img


def _apply_nl_means_path(img_array, l_str, c_str, method):
    """Internal helper to apply NL-Means denoising path."""
    yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)

    # Map strength 0-50 to h parameters
    h_y = (l_str / 50.0) * 0.01
    h_uv = (c_str / 50.0) * 50.0

    p_size, s_size = 5, 13
    if "Ultra Fast" in method:
        p_size, s_size = 3, 3
    elif "Fast+" in method:
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
    """Internal helper to apply Bilateral denoising path."""
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
    img,
    luma_strength=0,
    chroma_strength=0,
    method="High Quality",
    zoom=None,
    tier=None,
):
    """
    De-noising for NumPy float32 arrays.
    """
    if img is None:
        return None

    l_str = float(luma_strength)
    c_str = float(chroma_strength)

    if l_str <= 0 and c_str <= 0:
        return img

    # Pipeline ensures float32 and C-contiguous

    h, w = img.shape[:2]
    tier_str = f" | Tier: {tier}" if tier is not None else ""

    start_time = time.perf_counter()
    denoised = None
    method_name = method

    try:
        if method.startswith("NLMeans (Numba"):
            # Get clean variant name for logging (strip "YUV" suffix)
            inner = method.split("(", 1)[1].rstrip(")")  # e.g. "Numba Fast+ YUV"
            tokens = inner.split()
            tokens = [t for t in tokens if t not in ("Numba", "YUV")]
            variant = " ".join(tokens) if tokens else "Full"
            method_name = f"NL-Means ({variant})"

            denoised = _apply_nl_means_path(img, l_str, c_str, method)
        else:
            method_name = "Bilateral"
            denoised = _apply_bilateral_path(img, l_str, c_str)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Denoise: {method_name} | Size: {w}x{h}{tier_str} | Time: {elapsed:.2f}ms"
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

    # Pipeline ensures float32 and C-contiguous

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
        result = dehaze_recovery_kernel(img_array, transmission, atmospheric_light)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Dehaze: Strength: {strength:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )

        return result, atmospheric_light

    except Exception as e:
        logger.error(f"Dehaze failed: {e}")
        return img, None
