#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import json
import time
import rawpy
from PIL import Image, ImageFilter
from functools import lru_cache
from numba import jit

SUPPORTED_EXTS = (".cr3", ".CR3", ".cr2", ".CR2", ".dng", ".DNG")

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False


@jit(nopython=True, fastmath=True)
def _numba_tone_map(
    img, exposure, contrast, blacks, whites, shadows, highlights, saturation
):
    """
    JIT-compiled core tone mapping function.
    Operates purely on numpy arrays for max performance.
    """
    # Ensure all parameters and constants are float32 to match image array type
    f32 = np.float32
    exposure = f32(exposure)
    contrast = f32(contrast)
    blacks = f32(blacks)
    whites = f32(whites)
    shadows = f32(shadows)
    highlights = f32(highlights)
    saturation = f32(saturation)

    # 1. Exposure
    if exposure != 0.0:
        img = img * (f32(2.0) ** exposure)

    # 1.5 Contrast
    if contrast != 1.0:
        img = (img - f32(0.5)) * contrast + f32(0.5)

    # 2. Levels
    if blacks != 0.0 or whites != 1.0:
        denom = whites - blacks
        if abs(denom) < f32(1e-6):
            denom = f32(1e-6)
        img = (img - blacks) / denom

    # 3. Tone EQ
    if shadows != 0.0 or highlights != 0.0:
        # Numba requires explicit loops for this kind of operation
        h, w, _ = img.shape
        for r in range(h):
            for c in range(w):
                lum = (
                    f32(0.2126) * img[r, c, 0]
                    + f32(0.7152) * img[r, c, 1]
                    + f32(0.0722) * img[r, c, 2]
                )
                lum = max(f32(0.0), min(f32(1.0), lum))  # Clip lum

                if shadows != 0.0:
                    s_mask = (f32(1.0) - lum) ** f32(2.0)
                    img[r, c, :] += shadows * s_mask * img[r, c, :]

                if highlights != 0.0:
                    h_mask = lum ** f32(2.0)
                    img[r, c, :] += highlights * h_mask * (f32(1.0) - img[r, c, :])

    # 4. Saturation
    if saturation != 1.0:
        h, w, _ = img.shape
        for r in range(h):
            for c in range(w):
                lum = (
                    f32(0.2126) * img[r, c, 0]
                    + f32(0.7152) * img[r, c, 1]
                    + f32(0.0722) * img[r, c, 2]
                )
                img[r, c, :] = lum + (img[r, c, :] - lum) * saturation

    return img


# ---------------- Tone Mapping ----------------
def apply_tone_map(
    img,
    exposure=0.0,
    contrast=1.0,
    blacks=0.0,
    whites=1.0,
    shadows=0.0,
    highlights=0.0,
    saturation=1.0,
):
    """
    Applies Exposure -> Levels -> Tone EQ -> Saturation -> Base Curve
    """
    img_in = img.copy()  # Ensure we don't modify the input array in-place
    total_pixels = img_in.size

    # Call the JIT-compiled function for the heavy lifting
    processed_img = _numba_tone_map(
        img_in, exposure, contrast, blacks, whites, shadows, highlights, saturation
    )

    # Clip stats for reporting (done outside the JIT function)
    clipped_shadows = np.sum(processed_img < 0.0)
    clipped_highlights = np.sum(processed_img > 1.0)

    processed_img = np.clip(processed_img, 0.0, 1.0)

    stats = {
        "pct_shadows_clipped": clipped_shadows / total_pixels * 100,
        "pct_highlights_clipped": clipped_highlights / total_pixels * 100,
        "mean": processed_img.mean(),
    }

    return processed_img, stats


def calculate_auto_exposure(img):
    """
    Analyzes image histogram to determine auto-exposure and contrast settings.
    Returns a dict with recommended {exposure, blacks, whites}.
    """
    # 1. Calculate luminance
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    # Baseline for "Standard Look"
    # Most cameras underexpose the RAW data by ~1 stop to protect highlights.
    # We apply a default +1.25 EV boost to counteract this and match the JPEG brightness.
    base_exposure = 1.25

    # We still analyze the histogram just to avoid massive clipping if the photo is ALREADY bright.
    avg_lum = np.mean(lum)

    # If image is super bright (High Key), reduce the base boost
    if avg_lum > 0.6:  # Relaxed threshold for high key
        base_exposure = 0.5
    elif avg_lum > 0.3:
        base_exposure = 1.0  # Aggressive boost even for mid-tones

    # Standard Contrast (S-Curve simulation via Levels)
    # We pull blacks down and push whites up slightly to create "Pop"
    base_blacks = 0.08  # Very aggressive black crush for high contrast
    base_whites = 0.92

    return {
        "exposure": float(base_exposure),
        "blacks": float(base_blacks),
        "whites": float(base_whites),
        "highlights": 0.0,
        "shadows": 0.0,
        "saturation": 1.10,  # 10% Saturation boost (Standard Profile)
    }


@lru_cache(maxsize=4)
def open_raw(path, half_size=False, output_bps=8):
    """
    Opens a RAW file.
    Args:
        path: File path (str or Path)
        half_size: If True, decodes at 1/2 resolution (1/4 pixels) for speed.
        output_bps: Bit depth of the output image (8 or 16).
    """
    path = str(path)  # rawpy requires str
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=half_size,
            no_auto_bright=False,
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
    path = str(path)  # rawpy requires str
    try:
        with rawpy.imread(path) as raw:
            try:
                thumb = raw.extract_thumb()
            except rawpy.LibRawNoThumbnailError:
                thumb = None

            # If we found a JPEG thumbnail
            if thumb and thumb.format == rawpy.ThumbFormat.JPEG:
                from io import BytesIO

                return Image.open(BytesIO(thumb.data))

            # Fallback: fast postprocess (half_size=True is very fast)
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # 1/4 resolution
                no_auto_bright=True,
                output_bps=8,
            )
            return Image.fromarray(rgb)

    except Exception as e:
        print(f"Error extracting thumbnail for {path}: {e}")
        return None


def sharpen_image(pil_img, radius, percent, method="High Quality"):
    """Advanced sharpening with simplified options."""
    # Ensure image is RGB (no alpha) for OpenCV algorithms
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    if method == "High Quality":
        # Convert PIL to OpenCV format for better algorithms
        try:
            import cv2

            img_array = np.array(pil_img)
            img_float = img_array.astype(np.float32) / 255.0

            # Create unsharp mask
            blur = cv2.GaussianBlur(img_float, (0, 0), radius)
            sharpened = img_float + (img_float - blur) * (percent / 100.0)

            # Edge-aware threshold to prevent halo artifacts and noise sharpening
            # Convert to uint8 for Canny
            gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges slightly to cover transition zones
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Combine: Sharpened edges, but keep original for flat areas (to avoid noise)
            # We use the original img_float for flat areas instead of the heavily blurred version
            result = np.where(edges[:, :, np.newaxis] > 0, sharpened, img_float)

            # Convert back to PIL
            result_array = np.clip(result * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(result_array)
        except Exception as e:
            print(f"High Quality Sharpen failed: {e}")

    # Fallback to original PIL implementation (Standard)
    return pil_img.filter(
        ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent))
    )


def de_noise_image(pil_img, strength, method="High Quality"):
    """Advanced de-noising with simplified options."""
    # Ensure image is RGB (no alpha) for OpenCV algorithms
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    try:
        import cv2

        img_array = np.array(pil_img)

        if strength <= 0:
            return pil_img

        if method == "High Quality":
            # Sensor-aware de-noising: Separates Luma and Chroma
            # This targets color noise aggressively while preserving luminance detail
            yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv)

            # 1. Denoise Chrominance (U, V) - Aggressive to remove color blotches
            # Color noise is visually distracting but carries less detail
            chroma_strength = float(strength) * 1.5
            # Use small d and tight sigmaSpace to prevent color bleeding
            u_denoised = cv2.bilateralFilter(u, 5, chroma_strength * 3, 1.5)
            v_denoised = cv2.bilateralFilter(v, 5, chroma_strength * 3, 1.5)

            # 2. Denoise Luminance (Y) - Very conservative to preserve fine texture
            # We use a very small sigmaSpace (0.5-0.8) to ensure only
            # the most local pixels are averaged, preserving high-frequency detail.
            luma_sigma_color = float(strength) * 1.2
            luma_sigma_space = 0.5 + (
                float(strength) / 100.0
            )  # Very tight spatial control
            y_denoised = cv2.bilateralFilter(y, 3, luma_sigma_color, luma_sigma_space)

            # Merge back
            denoised_yuv = cv2.merge([y_denoised, u_denoised, v_denoised])
            denoised = cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2RGB)

            return Image.fromarray(denoised)

        elif method == "Edge Aware":
            # Standard Bilateral filter on RGB with tighter spatial constraints
            # sigmaSpace is reduced to prevent global blurring
            denoised = cv2.bilateralFilter(img_array, 5, float(strength) * 2, 1.0)
            if denoised is not None:
                return Image.fromarray(denoised)
    except Exception as e:
        print(f"OpenCV Denoise ({method}) failed: {e}")

    # Fallback to original median filter (Standard)
    size = int(strength)
    if size < 3:
        if strength > 0:
            size = 3
        else:
            return pil_img

    # Ensure size is odd
    if size % 2 == 0:
        size += 1

    return pil_img.filter(ImageFilter.MedianFilter(size=size))


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


def get_sidecar_path(raw_path):
    """
    Returns the Path object to the sidecar JSON file for a given RAW file.
    Sidecars are stored in a hidden .pyNegative directory local to the image.
    """
    raw_path = Path(raw_path)
    return raw_path.parent / SIDECAR_DIR / f"{raw_path.name}.json"


def save_sidecar(raw_path, settings):
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


def load_sidecar(raw_path):
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
        print(f"Error loading sidecar {sidecar_path}: {e}")
        return None


def rename_sidecar(old_raw_path, new_raw_path):
    """
    Renames a sidecar file when the original RAW is moved/renamed.
    """
    old_sidecar = get_sidecar_path(old_raw_path)
    new_sidecar = get_sidecar_path(new_raw_path)

    if old_sidecar.exists():
        new_sidecar.parent.mkdir(parents=True, exist_ok=True)
        old_sidecar.rename(new_sidecar)
