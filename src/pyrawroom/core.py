#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import json
import time

SUPPORTED_EXTS = (".cr3", ".CR3", ".cr2", ".CR2", ".dng", ".DNG")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False


# ---------------- Tone Mapping ----------------
def apply_tone_map(img, exposure=0.0, blacks=0.0, whites=1.0, shadows=0.0, highlights=0.0, saturation=1.0):
    """
    Applies Exposure -> Levels -> Tone EQ -> Saturation -> Base Curve
    """
    img = img.copy() # Ensure we don't modify the input array in-place
    total_pixels = img.size

    # 1. Exposure (2^stops)
    if exposure != 0.0:
        img = img * (2**exposure)

    # 2. Levels (Blacks & Whites)
    if blacks != 0.0 or whites != 1.0:
        denom = whites - blacks
        if abs(denom) < 1e-6: denom = 1e-6
        img = (img - blacks) / denom

    # 3. Tone EQ (Shadows & Highlights)
    if shadows != 0.0 or highlights != 0.0:
        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        lum = np.clip(lum, 0, 1)
        lum = lum[:,:,np.newaxis]

        if shadows != 0.0:
            s_mask = (1.0 - lum) ** 2
            img += shadows * s_mask * img

        if highlights != 0.0:
            h_mask = lum ** 2
            img += highlights * h_mask * (1.0 - img)

    # 4. Saturation
    if saturation != 1.0:
        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        lum = lum[:,:,np.newaxis]
        img = lum + (img - lum) * saturation

    # 5. Base Curve (Sigmoid for "Punch")
    # Simple S-Curve implementation: f(x) = x^2 / (x^2 + (1-x)^2) ? No, that's too strong.
    # Let's use a cosmetic contrast curve: x - a*sin(2*pi*x) ?
    # Or just a simple power-sigmoid
    # Start with a subtle S-curve
    # This applies the "Film Look" that cameras do
    # Using a simple centered sigmoid: 1 / (1 + exp(-k * (x - 0.5)))
    # But mapped to 0..1
    pass # For now, let's rely on saturation + exposure boost + contrast slider (whites/blacks)
    # Actually, let's just do a tiny gamma tweak for midtone contrast if needed
    # But for "Standard Profile", the Baseline Exposure + Saturation is usually the biggest factor.

    # Clip stats for reporting
    clipped_shadows = np.sum(img < 0.0)
    clipped_highlights = np.sum(img > 1.0)

    img = np.clip(img, 0.0, 1.0)

    stats = {
        "pct_shadows_clipped": clipped_shadows / total_pixels * 100,
        "pct_highlights_clipped": clipped_highlights / total_pixels * 100,
        "mean": img.mean(),
    }

    return img, stats

def calculate_auto_exposure(img):
    """
    Analyzes image histogram to determine auto-exposure and contrast settings.
    Returns a dict with recommended {exposure, blacks, whites}.
    """
    # 1. Calculate luminance
    lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]

    # Baseline for "Standard Look"
    # Most cameras underexpose the RAW data by ~1 stop to protect highlights.
    # We apply a default +1.25 EV boost to counteract this and match the JPEG brightness.
    base_exposure = 1.25

    # We still analyze the histogram just to avoid massive clipping if the photo is ALREADY bright.
    avg_lum = np.mean(lum)

    # If image is super bright (High Key), reduce the base boost
    if avg_lum > 0.6: # Relaxed threshold for high key
        base_exposure = 0.5
    elif avg_lum > 0.3:
        base_exposure = 1.0 # Aggressive boost even for mid-tones

    # Standard Contrast (S-Curve simulation via Levels)
    # We pull blacks down and push whites up slightly to create "Pop"
    base_blacks = 0.08 # Very aggressive black crush for high contrast
    base_whites = 0.92

    return {
        "exposure": float(base_exposure),
        "blacks": float(base_blacks),
        "whites": float(base_whites),
        "highlights": 0.0,
        "shadows": 0.0,
        "saturation": 1.10 # 10% Saturation boost (Standard Profile)
    }

import rawpy
from PIL import Image, ImageFilter
from functools import lru_cache

@lru_cache(maxsize=4)
def open_raw(path, half_size=False):
    """
    Opens a RAW file.
    Args:
        path: File path (str or Path)
        half_size: If True, decodes at 1/2 resolution (1/4 pixels) for speed.
    """
    path = str(path)  # rawpy requires str
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=half_size,
            no_auto_bright=False,
            output_bps=8
        )
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
                output_bps=8
            )
            return Image.fromarray(rgb)

    except Exception as e:
        print(f"Error extracting thumbnail for {path}: {e}")
        return None

def sharpen_image(pil_img, radius, percent):
    return pil_img.filter(
        ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent))
    )

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
SIDECAR_DIR = ".pyRawRoom"

def get_sidecar_path(raw_path):
    """
    Returns the Path object to the sidecar JSON file for a given RAW file.
    Sidecars are stored in a hidden .pyRawRoom directory local to the image.
    """
    raw_path = Path(raw_path)
    return raw_path.parent / SIDECAR_DIR / f"{raw_path.name}.json"

def save_sidecar(raw_path, settings):
    """
    Saves edit settings to a JSON sidecar file.
    """
    sidecar_path = get_sidecar_path(raw_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "last_modified": time.time(),
        "raw_path": str(raw_path),
        "settings": settings
    }

    with open(sidecar_path, 'w') as f:
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
        with open(sidecar_path, 'r') as f:
            data = json.load(f)
            return data.get("settings")
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
