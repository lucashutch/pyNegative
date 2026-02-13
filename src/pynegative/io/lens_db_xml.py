import os
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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

            if maker and model:
                self.lenses.append(
                    {
                        "maker": maker,
                        "model": model,
                        "mounts": mounts,
                        "crop_factor": float(crop_factor) if crop_factor else 1.0,
                        "maker_lower": maker.lower(),
                        "model_lower": model.lower(),
                        "raw_element": lens,  # Keep reference for Phase 2+ calibration data
                    }
                )

    def find_lens(
        self, camera_maker: str, camera_model: str, lens_model: str
    ) -> Optional[Dict]:
        if not self.loaded:
            return None

        # 1. Try exact match on lens model
        lens_model_lower = lens_model.lower()
        camera_maker_lower = camera_maker.lower()

        # Check for common third-party manufacturers in the lens name
        third_party = [
            "sigma",
            "tamron",
            "tokina",
            "samyang",
            "rokinon",
            "zeiss",
            "voigtlander",
            "laowa",
        ]
        is_third_party = any(tp in lens_model_lower for tp in third_party)

        best_match = None
        best_score = 0

        for lens in self.lenses:
            score = 0

            # Exact match is highest priority
            if lens_model_lower == lens["model_lower"]:
                return lens

            # Check if lens_model is a substring of DB model or vice versa
            if (
                lens_model_lower in lens["model_lower"]
                or lens["model_lower"] in lens_model_lower
            ):
                score += 10

                # Check maker match
                if (
                    lens["maker_lower"] in camera_maker_lower
                    or camera_maker_lower in lens["maker_lower"]
                ):
                    score += 5

                # If lens name contains the DB maker name, it's a good sign
                if lens["maker_lower"] in lens_model_lower:
                    score += 8

                # Length similarity (prefer closer match lengths)
                len_diff = abs(len(lens_model_lower) - len(lens["model_lower"]))
                score += max(0, 5 - len_diff // 5)

            if score > best_score:
                best_score = score
                best_match = lens

        # Minimum score threshold to avoid garbage matches
        if best_score >= 10:
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
        return sorted(list(set(f"{l['maker']} {l['model']}" for l in self.lenses)))


# Global instance
_instance = LensDatabase()


def get_instance():
    return _instance


def is_database_available():
    return _instance.loaded


def load_database(db_path: str | Path):
    return _instance.load(db_path)
