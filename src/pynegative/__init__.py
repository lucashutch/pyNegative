import logging
import os
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


logger = logging.getLogger(__name__)


def _setup_opencv_cache():
    """Configure OpenCV OpenCL cache directory before cv2 is imported."""
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("pynegative", ensure_exists=True)) / "opencv_cl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENCV_OPENCL_CACHE_DIR"] = str(cache_dir)
    logger.debug(f"OpenCV cache directory set to: {cache_dir}")


# MUST set OpenCV cache dir BEFORE importing cv2 anywhere
_setup_opencv_cache()

from .core import (  # noqa: E402
    apply_tone_map,
    apply_preprocess,
    open_raw,
    extract_thumbnail,
    calculate_auto_exposure,
    sharpen_image,
    save_image,
    de_noise_image,
    de_haze_image,
    apply_lens_correction,
    apply_defringe,
    save_sidecar,
    load_sidecar,
    SUPPORTED_EXTS,
    HEIF_SUPPORTED,
)

__all__ = [
    "apply_tone_map",
    "apply_preprocess",
    "open_raw",
    "extract_thumbnail",
    "calculate_auto_exposure",
    "sharpen_image",
    "save_image",
    "de_noise_image",
    "de_haze_image",
    "apply_lens_correction",
    "apply_defringe",
    "save_sidecar",
    "load_sidecar",
    "SUPPORTED_EXTS",
    "HEIF_SUPPORTED",
]

try:
    __version__ = version("pynegative")
except PackageNotFoundError:
    __version__ = "unknown"
