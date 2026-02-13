import logging
import exifread
import rawpy
from pathlib import Path
from typing import Dict, Optional, Any
from . import bmff_metadata

logger = logging.getLogger(__name__)


def extract_lens_info(raw_path: str | Path) -> Dict[str, Any]:
    """
    Extracts camera and lens information from RAW or standard image file.

    Returns a dictionary with 'camera_make', 'camera_model', 'lens_model',
    'focal_length', and 'aperture'.
    """
    raw_path = Path(raw_path)
    info = {
        "camera_make": "",
        "camera_model": "",
        "lens_model": "",
        "focal_length": None,
        "aperture": None,
    }
    ext = raw_path.suffix.lower()

    # Helper to convert Ratio/fractions to float
    def to_float(val):
        if val is None:
            return None
        try:
            if hasattr(val, "num") and hasattr(val, "den"):
                return float(val.num) / float(val.den) if val.den != 0 else None
            s = str(val)
            if "/" in s:
                parts = s.split("/")
                return (
                    float(parts[0]) / float(parts[1]) if float(parts[1]) != 0 else None
                )
            return float(val)
        except:
            return None

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

    # 2. For CR3 files, use our manual BMFF extractor
    if ext == ".cr3":
        try:
            bmff_tags = bmff_metadata.extract_bmff_metadata(raw_path)
            bmff_info = bmff_metadata.get_exposure_info(bmff_tags)

            if not info["camera_make"]:
                info["camera_make"] = bmff_info.get("camera_make", "")
            if not info["camera_model"]:
                info["camera_model"] = bmff_info.get("camera_model", "")
            if not info["lens_model"]:
                info["lens_model"] = bmff_info.get("lens_model", "")

            info["focal_length"] = to_float(bmff_info.get("focal_length"))
            info["aperture"] = to_float(bmff_info.get("aperture"))
        except Exception as e:
            logger.debug(f"BMFF extraction failed for {raw_path}: {e}")

    # 3. Try exifread as fallback if info is missing
    # Only if it's a format likely supported or if we're missing critical info
    if not (
        info["lens_model"]
        and info["camera_make"]
        and info["camera_model"]
        and info["focal_length"]
    ):
        try:
            # Avoid exifread for CR3 as we know it fails and produces warnings
            if ext != ".cr3":
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

                    if not info["focal_length"]:
                        info["focal_length"] = to_float(
                            tags.get("EXIF FocalLength")
                            or tags.get("Image FocalLength")
                        )
                    if not info["aperture"]:
                        info["aperture"] = to_float(
                            tags.get("EXIF FNumber") or tags.get("Image FNumber")
                        )
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
