import logging
import time

import cv2
import numpy as np

from ..utils.numba_kernels import (
    bilateral_kernel_yuv,
    bilateral_kernel_luma,
    bilateral_kernel_chroma,
    dark_channel_kernel,
    dehaze_recovery_kernel,
    sharpen_kernel,
    transmission_dark_channel_kernel,
)

logger = logging.getLogger(__name__)


def sharpen_image(img, radius, percent):
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
        # Unsharp mask: sharpen = img + percent * (img - blur)
        # The mask is inherently edge-selective because (img - blur) is
        # only large near edges and near-zero in flat/smooth areas.
        blur = cv2.GaussianBlur(img, (0, 0), radius)
        sharpened = img.copy()
        sharpen_kernel(sharpened, blur, percent)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Sharpen: Radius: {radius:.2f} | Percent: {percent:.1f}%{size_str} | Time: {elapsed:.2f}ms"
        )
        return sharpened
    except Exception as e:
        logger.error(f"High Quality Sharpen failed: {e}")
        return img


def _apply_bilateral_path(img_array, l_str, c_str):
    """Internal helper to apply Bilateral denoising path.

    Intelligently selects the optimal kernel variant:
    - luma_only: Use fast bilateral_kernel_luma
    - chroma_only: Use fast bilateral_kernel_chroma with pre-computed luma
    - both: Use bilateral_kernel_yuv (fused) to avoid redundant processing
    """
    s_scale = 1.0 / 255.0
    yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)

    # Chroma slider has 2× sensitivity
    c_str = c_str * 2.0

    sigma_color_y = max(1e-6, l_str * 0.4 * s_scale)
    sigma_space_y = 0.5 + (l_str / 100.0)
    sigma_color_uv = max(1e-6, c_str * 4.5 * s_scale)
    sigma_space_uv = 3.0 + (c_str / 8.0)

    # Smart kernel selection for optimal performance
    if l_str > 0 and c_str > 0:
        # Both enabled: use fused kernel to avoid redundant luma processing
        img_yuv_denoised = bilateral_kernel_yuv(
            yuv,
            l_str,
            sigma_color_y,
            sigma_space_y,
            sigma_color_uv,
            sigma_space_uv,
        )
    elif l_str > 0:
        # Luma only: skip chroma processing entirely
        img_yuv_denoised = bilateral_kernel_luma(
            yuv,
            l_str,
            sigma_color_y,
            sigma_space_y,
        )
    else:
        # Chroma only: compute luma once for Y-guidance, then apply chroma filter
        plane_y = np.ascontiguousarray(yuv[:, :, 0])
        img_yuv_denoised = bilateral_kernel_chroma(
            yuv,
            plane_y,
            c_str,
            sigma_color_uv,
            sigma_space_uv,
        )

    return cv2.cvtColor(img_yuv_denoised, cv2.COLOR_YUV2RGB)


def de_noise_image(
    img,
    luma_strength=0,
    chroma_strength=0,
    zoom=None,
    tier=None,
):
    """De-noising for NumPy float32 arrays using bilateral filter."""
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

    try:
        denoised = _apply_bilateral_path(img, l_str, c_str)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Denoise: Bilateral | Size: {w}x{h}{tier_str} | Time: {elapsed:.2f}ms"
        )

    except Exception as e:
        logger.error(f"Denoise failed: {e}")

    if denoised is not None:
        return np.clip(denoised, 0, 1)

    return img


def estimate_atmospheric_light(img):
    """Estimate the atmospheric light from an image using Dark Channel Prior.

    This should be called once on a low-res version of the full image,
    and the result passed to de_haze_image() as fixed_atmospheric_light
    for every tile.  This prevents per-tile colour tint differences.

    Returns a numpy array of shape (3,) with the atmospheric light per channel,
    or None if estimation fails.
    """
    if img is None:
        return None
    try:
        h_img, w_img = img.shape[:2]
        kernel_size = max(3, int(15 * (w_img / 2048.0)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        dark_channel = dark_channel_kernel(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(dark_channel, kernel)

        num_pixels = dark_channel.size
        num_brightest = max(1, num_pixels // 1000)
        indices = np.argpartition(dark_channel.flatten(), -num_brightest)[
            -num_brightest:
        ]
        brightest_pixels = img.reshape(-1, 3)[indices]
        return np.max(brightest_pixels, axis=0)
    except Exception:
        return None


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
        omega = 0.95 * strength

        # Fused kernel: computes min(I_c / A_c) per pixel in one pass,
        # avoiding a full HxWx3 normalized_img intermediate allocation.
        dark_normalized = transmission_dark_channel_kernel(img_array, atmospheric_light)

        # Morphology (Erode) - OpenCV CPU is very fast for this
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_normalized = cv2.erode(dark_normalized, kernel)
        transmission = 1.0 - omega * dark_normalized

        # Refine transmission map
        transmission = cv2.GaussianBlur(
            transmission, (kernel_size * 2 + 1, kernel_size * 2 + 1), 0
        )

        # 4. Recover radiance
        # J(x) = (I(x) - A) / max(t(x), t0) + A
        # Note: dehaze_recovery_kernel already clamps transmission to
        # max(t, 0.1) internally, so no explicit lower-bound needed here.
        result = dehaze_recovery_kernel(img_array, transmission, atmospheric_light)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Dehaze: Strength: {strength:.2f}{size_str}{zoom_str} | Time: {elapsed:.2f}ms"
        )

        return result, atmospheric_light

    except Exception as e:
        logger.error(f"Dehaze failed: {e}")
        return img, None
