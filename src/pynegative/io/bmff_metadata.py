import logging
import exifread
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def extract_bmff_metadata(path: str | Path) -> Dict[str, Any]:
    """
    Extracts metadata from BMFF-based RAW files (like Canon CR3) by searching for
    embedded TIFF headers and parsing them.
    """
    path = Path(path)
    merged_tags = {}

    try:
        with open(path, "rb") as f:
            # Read first 2MB which usually contains all metadata headers
            data = f.read(2 * 1024 * 1024)

            # Find all TIFF headers (Little Endian and Big Endian)
            headers = []
            for header in [b"II\x2a\x00", b"MM\x00\x2a"]:
                pos = 0
                while True:
                    pos = data.find(header, pos)
                    if pos == -1:
                        break
                    headers.append(pos)
                    pos += 4

            # Parse each TIFF block found
            for pos in headers:
                try:
                    # Parse from this offset to the end of the buffered data
                    exif_data = data[pos:]
                    tags = exifread.process_file(BytesIO(exif_data), details=False)

                    # Merge tags, prioritizing ones with non-None values
                    for k, v in tags.items():
                        if v is not None:
                            # Use simple names if possible or keep full names
                            merged_tags[k] = v
                except Exception as e:
                    logger.debug(
                        f"Failed to parse TIFF block at offset {pos} in {path}: {e}"
                    )

    except Exception as e:
        logger.error(f"Error reading BMFF metadata from {path}: {e}")

    return merged_tags


def get_exposure_info(tags: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to extract standard exposure info from a merged tags dictionary.
    """
    info = {}

    # ISO
    iso = tags.get("EXIF ISOSpeedRatings") or tags.get("Image ISOSpeedRatings")
    if iso:
        info["iso"] = iso

    # Shutter Speed
    shutter = tags.get("EXIF ExposureTime") or tags.get("Image ExposureTime")
    if shutter:
        info["shutter_speed"] = shutter

    # Aperture
    aperture = tags.get("EXIF FNumber") or tags.get("Image FNumber")
    if aperture:
        info["aperture"] = aperture

    # Focal Length
    focal = tags.get("EXIF FocalLength") or tags.get("Image FocalLength")
    if focal:
        info["focal_length"] = focal

    # Camera
    make = tags.get("Image Make") or tags.get("EXIF Make")
    if make:
        info["camera_make"] = str(make).strip()

    model = tags.get("Image Model") or tags.get("EXIF Model")
    if model:
        info["camera_model"] = str(model).strip()

    # Lens
    lens = tags.get("EXIF LensModel") or tags.get("Image LensModel")
    if lens:
        info["lens_model"] = str(lens).strip()

    return info
