#!/usr/bin/env python3
import numpy as np

SUPPORTED_EXTS = (".cr3", ".CR3", ".cr2", ".CR2", ".dng", ".DNG")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False


# ---------------- Tone Mapping ----------------
def apply_tone_map(img, exposure=0.0, blacks=0.0, whites=1.0, shadows=0.0, highlights=0.0):
    """
    Applies Exposure -> Levels (Blacks/Whites) -> Tone EQ (Shadows/Highlights)
    """
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
        # Luminance
        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        lum = np.clip(lum, 0, 1)
        lum = lum[:,:,np.newaxis]

        # Shadows (Lift darks)
        if shadows != 0.0:
            s_mask = (1.0 - lum) ** 2
            img += shadows * s_mask * img

        # Highlights (Recover brights)
        if highlights != 0.0:
            h_mask = lum ** 2
            img += highlights * h_mask * (1.0 - img)

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

import rawpy
from PIL import Image, ImageFilter

def open_raw(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8
        )
    return rgb.astype(np.float32) / 255.0

def extract_thumbnail(path):
    """
    Attempts to extract an embedded thumbnail.
    Falls back to a fast, half-size RAW conversion if no thumbnail exists.
    Returns a PIL Image or None on failure.
    """
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
        ImageFilter.UnsharpMask(radius=radius, percent=percent)
    )

def save_image(pil_img, output_path, quality=95):
    fmt = output_path.split('.')[-1].lower()
    if fmt == "jpeg" or fmt == "jpg":
        pil_img.save(output_path, quality=quality)
    elif fmt == "heif" or fmt == "heic":
        if not HEIF_SUPPORTED:
            raise RuntimeError("HEIF requested but pillow-heif not installed.")
        pil_img.save(output_path, format="HEIF", quality=quality)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
