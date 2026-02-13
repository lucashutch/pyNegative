import logging
import exifread
import rawpy
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_lens_info(raw_path: str | Path) -> Dict[str, str]:
    """
    Extracts camera and lens information from RAW or standard image file.

    Returns a dictionary with 'camera_make', 'camera_model', and 'lens_model'.
    """
    raw_path = Path(raw_path)
    info = {"camera_make": "", "camera_model": "", "lens_model": ""}

    # 1. Try rawpy (LibRaw) - robust for RAW formats
    try:
        with rawpy.imread(str(raw_path)) as raw:
            # LibRaw lens info
            if hasattr(raw, "lens_config"):
                lens_name = raw.lens_config.lens.decode(
                    "utf-8", errors="ignore"
                ).strip()
                lens_maker = raw.lens_config.maker.decode(
                    "utf-8", errors="ignore"
                ).strip()

                if lens_name:
                    # If lens_name is short and we have a maker, combine them
                    if lens_maker and lens_maker.lower() not in lens_name.lower():
                        info["lens_model"] = f"{lens_maker} {lens_name}"
                    else:
                        info["lens_model"] = lens_name

            # LibRaw camera info
            if hasattr(raw, "idata"):
                if not info["camera_make"] and hasattr(raw.idata, "make"):
                    info["camera_make"] = raw.idata.make.decode(
                        "utf-8", errors="ignore"
                    ).strip()
                if not info["camera_model"] and hasattr(raw.idata, "model"):
                    info["camera_model"] = raw.idata.model.decode(
                        "utf-8", errors="ignore"
                    ).strip()

    except Exception as e:
        logger.debug(f"rawpy failed to read lens info from {raw_path}: {e}")

    # 2. Try exifread as fallback if info is missing
    # Only if it's a format likely supported or if we're missing critical info
    if not (info["lens_model"] and info["camera_make"] and info["camera_model"]):
        try:
            with open(raw_path, "rb") as f:
                tags = exifread.process_file(f, details=False)

                if not info["camera_make"]:
                    info["camera_make"] = str(tags.get("Image Make", "")).strip()
                if not info["camera_model"]:
                    info["camera_model"] = str(tags.get("Image Model", "")).strip()

                if not info["lens_model"]:
                    lens_tags = [
                        "EXIF LensModel",
                        "MakerNote LensModel",
                        "MakerNote LensType",
                        "Image LensModel",
                    ]
                    for tag in lens_tags:
                        val = tags.get(tag)
                        if val:
                            info["lens_model"] = str(val).strip()
                            break

                    if not info["lens_model"]:
                        for key in tags.keys():
                            if "LensModel" in key or "LensType" in key:
                                info["lens_model"] = str(tags[key]).strip()
                                break
        except Exception as e:
            if not info["lens_model"]:
                logger.debug(f"exifread failed to read lens info from {raw_path}: {e}")

    return info


def extract_embedded_correction_params(raw_path: str | Path) -> Optional[Dict]:
    """
    Extracts embedded lens correction parameters (distortion, vignetting, CA).
    Currently a placeholder for Phase 2.
    """
    # This will involve parsing MakerNotes for specific manufacturers (Sony, Olympus, etc.)
    return None
