import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any

from . import lens_db_xml, lens_metadata

logger = logging.getLogger(__name__)


class ProfileSource(Enum):
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
) -> tuple[ProfileSource, dict[str, Any] | None]:
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

    # Tier 1: Lensfun Database Match
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
            logger.debug(f"Matched lensfun profile: {name}")

            # Get distortion params if available
            distortion = None
            vignetting = None
            tca = None
            if focal_length is not None:
                distortion = db.get_distortion_params(matched_lens, focal_length)
                tca = db.get_tca_params(matched_lens, focal_length)
                if aperture is not None:
                    vignetting = db.get_vignette_params(
                        matched_lens, focal_length, aperture
                    )

            return ProfileSource.LENSFUN_DB, {
                "name": name,
                "lens_data": matched_lens,
                "distortion": distortion,
                "vignetting": vignetting,
                "tca": tca,
                "exif": exif_info,
            }

    # Tier 3: Manual (or no match found)
    if lens_model:
        return ProfileSource.MANUAL, {"name": lens_model, "exif": exif_info}

    return ProfileSource.NONE, {"name": None, "exif": exif_info}
