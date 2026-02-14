import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LensDatabase:
    def __init__(self):
        self.cameras = []  # List of dicts
        self.lenses = []  # List of dicts
        self.mounts = {}  # mount_id -> mount_name
        self.loaded = False
        self.db_path = None

    def load(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        if not self.db_path.exists():
            logger.warning(f"Lens database path not found: {self.db_path}")
            return False

        self.cameras = []
        self.lenses = []

        xml_files = list(self.db_path.glob("*.xml"))
        for xml_file in xml_files:
            try:
                self._parse_file(xml_file)
            except Exception as e:
                logger.error(f"Error parsing {xml_file}: {e}")

        self.loaded = True
        logger.info(
            f"Loaded {len(self.cameras)} cameras and {len(self.lenses)} lenses from {self.db_path}"
        )
        return True

    def _parse_file(self, file_path: Path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Parse mounts
        for mount in root.findall("mount"):
            name = mount.findtext("name")
            if name:
                # We use the name as ID for simplicity if no ID is present
                self.mounts[name.lower()] = name

        # Parse cameras
        for cam in root.findall("camera"):
            maker = cam.findtext("maker")
            model = cam.findtext("model")
            mount = cam.findtext("mount")
            if maker and model:
                self.cameras.append(
                    {
                        "maker": maker,
                        "model": model,
                        "mount": mount,
                        "maker_lower": maker.lower(),
                        "model_lower": model.lower(),
                    }
                )

        # Parse lenses
        for lens in root.findall("lens"):
            maker = lens.findtext("maker")
            model = lens.findtext("model")
            mounts = [m.text for m in lens.findall("mount") if m.text]
            crop_factor = lens.findtext("crop_factor")

            # Calibration data for matching
            focal_min = lens.findtext("focal_min")
            focal_max = lens.findtext("focal_max")
            aperture = lens.findtext("aperture")

            if maker and model:
                self.lenses.append(
                    {
                        "maker": maker,
                        "model": model,
                        "mounts": mounts,
                        "crop_factor": float(crop_factor) if crop_factor else 1.0,
                        "maker_lower": maker.lower(),
                        "model_lower": model.lower(),
                        "focal_min": float(focal_min) if focal_min else None,
                        "focal_max": float(focal_max) if focal_max else None,
                        "aperture": float(aperture) if aperture else None,
                        "raw_element": lens,  # Keep reference for Phase 2+ calibration data
                    }
                )

    def find_lens(
        self,
        camera_maker: str,
        camera_model: str,
        lens_model: str,
        focal_length: Optional[float] = None,
        aperture: Optional[float] = None,
    ) -> Optional[Dict]:
        if not self.loaded:
            return None

        # 1. Normalize inputs
        lens_model_lower = lens_model.strip().lower()
        camera_maker_lower = camera_maker.strip().lower()

        # If lens model is empty or too short, we can't reliably match
        if len(lens_model_lower) < 2:
            return None

        best_match = None
        best_score = 0

        for lens in self.lenses:
            score = 0

            # Tier 1: Hardware Specs Check (Focal length and Aperture)
            # If we have EXIF specs, they MUST be within the lens's physical range
            if (
                focal_length is not None
                and lens["focal_min"] is not None
                and lens["focal_max"] is not None
            ):
                # Allow a small margin for focal length (e.g. 17.9mm matched to 18mm lens)
                if not (
                    lens["focal_min"] - 1.0 <= focal_length <= lens["focal_max"] + 1.0
                ):
                    continue  # Not a physical match

                # If exact focal length match for prime lens or zoom boundary, boost score
                if (
                    abs(lens["focal_min"] - focal_length) < 0.1
                    or abs(lens["focal_max"] - focal_length) < 0.1
                ):
                    score += 15

            if aperture is not None and lens["aperture"] is not None:
                # Aperture match (allow small margin)
                # If EXIF aperture is faster (lower f-stop) than lens max aperture, it's probably not the lens
                if aperture < lens["aperture"] - 0.1:
                    continue  # Too fast for this lens

                if abs(lens["aperture"] - aperture) < 0.1:
                    score += 10

            # Tier 2: String matching
            # Exact match is highest priority
            if lens_model_lower == lens["model_lower"]:
                score += 100

            # Check if lens_model is a substring of DB model or vice versa
            if (
                lens_model_lower in lens["model_lower"]
                or lens["model_lower"] in lens_model_lower
            ):
                score += 20

                # Check maker match
                if (
                    lens["maker_lower"] in camera_maker_lower
                    or camera_maker_lower in lens["maker_lower"]
                ):
                    score += 5

                # If lens name contains the DB maker name, it's a good sign
                if lens["maker_lower"] in lens_model_lower:
                    score += 15

                # Length similarity (prefer closer match lengths)
                len_diff = abs(len(lens_model_lower) - len(lens["model_lower"]))
                score += max(0, 10 - len_diff // 2)

            if score > best_score:
                best_score = score
                best_match = lens

        # Minimum score threshold to avoid garbage matches
        # If we have a decent string match (20+) or specs + string match
        if best_score >= 20:
            return best_match

        return None

    def search_lenses(self, query: str) -> List[Dict]:
        if not self.loaded:
            return []

        query_lower = query.lower()
        results = []
        for lens in self.lenses:
            if query_lower in lens["model_lower"] or query_lower in lens["maker_lower"]:
                results.append(lens)

        # Sort by relevance (exact match first, then starts with, then contains)
        return sorted(
            results,
            key=lambda x: (
                not x["model_lower"].startswith(query_lower),
                len(x["model"]),
            ),
        )[:50]

    def get_all_lens_names(self) -> List[str]:
        if not self.loaded:
            return []

        lens_names = set()
        for lens in self.lenses:
            maker = lens["maker"].strip()
            model = lens["model"].strip()
            if model.lower().startswith(maker.lower()):
                lens_names.add(model)
            else:
                lens_names.add(f"{maker} {model}")

        return sorted(list(lens_names))


# Global instance
_instance = LensDatabase()


def get_instance():
    return _instance


def is_database_available():
    return _instance.loaded


def load_database(db_path: str | Path):
    return _instance.load(db_path)
