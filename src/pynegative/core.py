#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import json
import time
import math
import rawpy
from PIL import Image, ImageFilter
from functools import lru_cache

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
    exposure=0.0,
    contrast=1.0,
    blacks=0.0,
    whites=1.0,
    shadows=0.0,
    highlights=0.0,
    saturation=1.0,
    temperature=0.0,
    tint=0.0,
    calculate_stats=True,
):
    """
    Applies White Balance -> Exposure -> Levels -> Tone EQ -> Saturation -> Base Curve
    Optimized for performance with in-place operations and minimal allocations.
    """
    # Create a single copy at the start to protect the input array
    img = img.copy()
    total_pixels = img.size

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

    # 2. Levels (Blacks & Whites)
    if blacks != 0.0 or whites != 1.0:
        denom = whites - blacks
        if abs(denom) < 1e-6:
            denom = 1e-6
        img -= blacks
        img /= denom

    # 3. Tone EQ (Shadows & Highlights) and 4. Saturation
    # Both need luminance. We calculate it once and reuse it.
    if shadows != 0.0 or highlights != 0.0 or saturation != 1.0:
        # Calculate luminance (Rec. 709)
        lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        np.clip(lum, 0, 1, out=lum)
        lum_3d = lum[:, :, np.newaxis]

        if shadows != 0.0:
            s_mask = (1.0 - lum_3d) ** 2
            img *= 1.0 + shadows * s_mask

        if highlights != 0.0:
            h_mask = lum_3d**2
            # img += highlights * h_mask * (1.0 - img) -> img = img * (1 - h_term) + h_term
            h_term = highlights * h_mask
            img *= 1.0 - h_term
            img += h_term

        if saturation != 1.0:
            # Reusing original luminance for saturation prevents color shifting
            # after Tone EQ adjustments.
            img -= lum_3d
            img *= saturation
            img += lum_3d

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

    return img, stats


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
    path = Path(path)
    ext = path.suffix.lower()

    if ext in STD_EXTS:
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            print(f"Error opening standard image thumbnail for {path}: {e}")
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


def sharpen_image(img, radius, percent, method="High Quality"):
    """Advanced sharpening with support for both PIL and Numpy float32."""
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

    if method == "High Quality":
        try:
            import cv2

            # Create unsharp mask
            blur = cv2.GaussianBlur(img_float, (0, 0), radius)
            sharpened = img_float + (img_float - blur) * (percent / 100.0)

            # Edge-aware threshold (Canny needs uint8)
            gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges slightly
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Combine: Sharpened edges, keep original for flat areas
            result = np.where(edges[:, :, np.newaxis] > 0, sharpened, img_float)

            if was_pil:
                result_array = np.clip(result * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(result_array)
            else:
                return np.clip(result, 0, 1.0)
        except Exception as e:
            print(f"High Quality Sharpen failed: {e}")

    # Fallback for PIL
    if was_pil:
        return img.filter(
            ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent))
        )

    # Fallback for Numpy (Basic Unsharp Mask)
    try:
        import cv2

        # Convert radius to kernel size (must be odd)
        k_size = int(2 * math.ceil(radius * 2) + 1)
        if k_size % 2 == 0:
            k_size += 1
        blur = cv2.GaussianBlur(img_float, (k_size, k_size), radius)
        result = img_float + (img_float - blur) * (percent / 100.0)
        return np.clip(result, 0, 1.0)
    except Exception:
        return img_float


def de_noise_image(img, strength, method="High Quality"):
    """Advanced de-noising with support for both PIL and Numpy float32.
    Preserves float32 precision throughout the pipeline to avoid bit-depth artifacts.
    """
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        was_pil = True
    else:
        img_array = img
        was_pil = False

    try:
        import cv2

        if strength <= 0:
            return img

        # Scaling factor for sigmaColor based on 0-1 range
        # OpenCV bilateralFilter on float32 expects sigma in the same scale as pixels
        s_scale = 1.0 / 255.0

        if method == "High Quality":
            # Sensor-aware de-noising: Separates Luma and Chroma
            # cvtColor handles float32 RGB -> YUV
            yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv)

            # Chroma: Aggressive to remove color blotches
            chroma_sigma = float(strength) * 2.0 * s_scale
            u_denoised = cv2.bilateralFilter(u, 5, chroma_sigma, 1.5)
            v_denoised = cv2.bilateralFilter(v, 5, chroma_sigma, 1.5)

            # Luma: Conservative to preserve fine texture
            luma_sigma_color = float(strength) * 0.8 * s_scale
            luma_sigma_space = 0.5 + (float(strength) / 50.0)
            y_denoised = cv2.bilateralFilter(y, 3, luma_sigma_color, luma_sigma_space)

            denoised_yuv = cv2.merge([y_denoised, u_denoised, v_denoised])
            denoised = cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2RGB)

            if was_pil:
                return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))
            else:
                return np.clip(denoised, 0, 1)

        elif method == "Edge Aware":
            # Standard Bilateral filter on RGB
            sigma = float(strength) * 1.5 * s_scale
            denoised = cv2.bilateralFilter(img_array, 5, sigma, 1.0)
            if denoised is not None:
                if was_pil:
                    return Image.fromarray(
                        (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
                    )
                else:
                    return np.clip(denoised, 0, 1)
    except Exception as e:
        print(f"OpenCV Denoise ({method}) failed: {e}")

    # Fallback to PIL or Median
    if was_pil:
        size = int(strength / 5.0)  # Scale down strength for median
        if size < 3:
            size = 3 if strength > 0 else 0
        if size == 0:
            return img
        if size % 2 == 0:
            size += 1
        # Convert back to PIL for the filter
        pil_img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
        return pil_img.filter(ImageFilter.MedianFilter(size=size))

    # Fallback for Numpy (Median Filter)
    try:
        import cv2

        size = int(strength / 5.0)
        if size < 3:
            size = 3 if strength > 0 else 0
        if size == 0:
            return img_array
        if size % 2 == 0:
            size += 1
        # medianBlur expects uint8 or float32. We can use float32.
        denoised = cv2.medianBlur(img_array, size)
        return np.clip(denoised, 0, 1)
    except Exception:
        return img_array


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


def get_exif_capture_date(raw_path):
    """
    Extracts the capture date from RAW or standard image file EXIF data.

    Returns the date as a string in YYYY-MM-DD format, or None if unavailable.
    Falls back to file modification date if EXIF date is not found.
    """
    from datetime import datetime

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

                        # Look for DateTimeOriginal (0x9003) or DateTime (0x0132)
                        import re

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
                print(f"Error extracting EXIF from {raw_path}: {e}")

        # Fallback: use file modification time
        mtime = raw_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    except Exception as e:
        print(f"Error reading file {raw_path}: {e}")
        return None
