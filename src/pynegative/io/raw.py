import logging
from pathlib import Path
from functools import lru_cache
import numpy as np
import rawpy
from PIL import Image, ImageOps

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
    from io import BytesIO

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
            except (rawpy.LibRawNoThumbnailError, Exception):
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
