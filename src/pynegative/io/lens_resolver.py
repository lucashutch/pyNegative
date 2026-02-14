import logging
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from . import lens_metadata
from . import lens_db_xml

logger = logging.getLogger(__name__)


class ProfileSource(Enum):
    EMBEDDED = auto()
    LENSFUN_DB = auto()
    MANUAL = auto()
    NONE = auto()


def format_lens_name(maker: str, model: str) -> str:
    maker = maker.strip()
    model = model.strip()
    if model.lower().startswith(maker.lower()):
        return model
    return f"{maker} {model}"


def resolve_lens_profile(
    raw_path: str | Path,
) -> Tuple[ProfileSource, Optional[Dict[str, Any]]]:
    """
    Resolves the lens profile using the 3-tier priority logic:
    1. Embedded RAW metadata
    2. Lensfun database auto-match
    3. Manual selection (returns None, UI handles manual mode)
    """
    raw_path = Path(raw_path)

    # Extract basic info from EXIF
    exif_info = lens_metadata.extract_lens_info(raw_path)
    camera_make = exif_info.get("camera_make", "")
    camera_model = exif_info.get("camera_model", "")
    lens_model = exif_info.get("lens_model", "")
    focal_length = exif_info.get("focal_length")
    aperture = exif_info.get("aperture")

    # Tier 1: Embedded Correction Params
    embedded_params = lens_metadata.extract_embedded_correction_params(raw_path)
    if embedded_params:
        logger.info(f"Found embedded lens correction profile for {lens_model}")
        return ProfileSource.EMBEDDED, {
            "name": lens_model,
            "params": embedded_params,
            "exif": exif_info,
        }

    # Tier 2: Lensfun Database Match
    db = lens_db_xml.get_instance()
    if db.loaded:
        matched_lens = db.find_lens(
            camera_make,
            camera_model,
            lens_model,
            focal_length=focal_length,
            aperture=aperture,
        )
        if matched_lens:
            name = format_lens_name(matched_lens["maker"], matched_lens["model"])
            logger.info(f"Matched lensfun profile: {name}")
            return ProfileSource.LENSFUN_DB, {
                "name": name,
                "lens_data": matched_lens,
                "exif": exif_info,
            }

    # Tier 3: Manual (or no match found)
    if lens_model:
        return ProfileSource.MANUAL, {"name": lens_model, "exif": exif_info}

    return ProfileSource.NONE, {"name": None, "exif": exif_info}
