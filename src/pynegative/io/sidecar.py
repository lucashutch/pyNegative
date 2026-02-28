import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SIDECAR_DIR = ".pyNegative"
THUMBNAIL_DIR = "thumbnails"


def get_thumbnail_cache_dir(raw_path: str | Path) -> Path:
    """Returns the path to the thumbnail cache directory."""
    return Path(raw_path).parent / SIDECAR_DIR / THUMBNAIL_DIR


def save_cached_thumbnail(
    raw_path: str | Path, pil_img: Image.Image, metadata: dict, size: int
) -> None:
    """Saves thumbnail and metadata to disk."""
    cache_dir = get_thumbnail_cache_dir(raw_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mtime = int(Path(raw_path).stat().st_mtime)
    base_name = f"{Path(raw_path).name}.{mtime}.{size}"

    # Save Image (WebP for efficiency)
    img_path = cache_dir / f"{base_name}.webp"
    pil_img.save(img_path, "WEBP", quality=85)

    # Save Metadata
    meta_path = cache_dir / f"{base_name}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)


def load_cached_thumbnail(
    raw_path: str | Path, size: int
) -> tuple[Image.Image | None, dict]:
    """Loads thumbnail and metadata from disk if they exist and are valid."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return None, {}

    mtime = int(raw_path.stat().st_mtime)
    base_name = f"{raw_path.name}.{mtime}.{size}"
    cache_dir = get_thumbnail_cache_dir(raw_path)

    img_path = cache_dir / f"{base_name}.webp"
    meta_path = cache_dir / f"{base_name}.json"

    if img_path.exists() and meta_path.exists():
        try:
            with Image.open(img_path) as img:
                # Load fully into memory so we can close the file handle
                pil_img = img.copy()
            with open(meta_path) as f:
                metadata = json.load(f)
            return pil_img, metadata
        except Exception as e:
            logger.error(f"Error loading cached thumbnail {img_path}: {e}")

    return None, {}


def get_sidecar_path(raw_path: str | Path) -> Path:
    """
    Returns the Path object to the sidecar JSON file for a given RAW file.
    Sidecars are stored in a hidden .pyNegative directory local to the image.
    """
    raw_path = Path(raw_path)
    return raw_path.parent / SIDECAR_DIR / f"{raw_path.name}.json"


def save_sidecar(raw_path: str | Path, settings: dict) -> None:
    """
    Saves edit settings to a JSON sidecar file.
    """
    sidecar_path = get_sidecar_path(raw_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure rating is present
    save_settings = settings.copy()
    if "rating" not in save_settings:
        save_settings["rating"] = 0

    # Preserve existing envelope fields (especially snapshots)
    existing = _load_sidecar_raw(raw_path)
    data = existing if existing else {}
    data["last_modified"] = time.time()
    data["raw_path"] = str(raw_path)
    data["settings"] = save_settings

    # Keep schema version compatible with snapshots when present
    if "snapshots" in data:
        data["version"] = "1.1"
    else:
        data["version"] = data.get("version", "1.0")

    _save_sidecar_raw(raw_path, data)


def load_sidecar(raw_path: str | Path) -> dict | None:
    """
    Loads edit settings from a JSON sidecar file if it exists.
    Returns the settings dict or None.
    """
    sidecar_path = get_sidecar_path(raw_path)
    if not sidecar_path.exists():
        return None

    try:
        with open(sidecar_path) as f:
            data = json.load(f)
            settings = data.get("settings")
            if settings:
                if "rating" not in settings:
                    settings["rating"] = 0
            return settings
    except Exception as e:
        logger.error(f"Error loading sidecar {sidecar_path}: {e}")
        return None


def rename_sidecar(old_raw_path: str | Path, new_raw_path: str | Path) -> None:
    """
    Renames a sidecar file and associated thumbnails when the original RAW is moved/renamed.
    """
    old_raw_path = Path(old_raw_path)
    new_raw_path = Path(new_raw_path)

    old_sidecar = get_sidecar_path(old_raw_path)
    new_sidecar = get_sidecar_path(new_raw_path)

    if old_sidecar.exists():
        new_sidecar.parent.mkdir(parents=True, exist_ok=True)
        old_sidecar.rename(new_sidecar)

    # Thumbnails
    old_thumb_dir = get_thumbnail_cache_dir(old_raw_path)
    new_thumb_dir = get_thumbnail_cache_dir(new_raw_path)

    if old_thumb_dir.exists():
        new_thumb_dir.mkdir(parents=True, exist_ok=True)
        old_name = old_raw_path.name
        new_name = new_raw_path.name

        for thumb_file in old_thumb_dir.glob(f"{old_name}.*"):
            new_thumb_filename = thumb_file.name.replace(old_name, new_name, 1)
            thumb_file.rename(new_thumb_dir / new_thumb_filename)


def get_sidecar_mtime(raw_path: str | Path) -> float | None:
    """
    Returns the last modified time of the sidecar file if it exists.
    """
    sidecar_path = get_sidecar_path(raw_path)
    if sidecar_path.exists():
        return sidecar_path.stat().st_mtime
    return None


# ---------------------------------------------------------------------------
# Snapshot / Version persistence
# ---------------------------------------------------------------------------

MAX_AUTO_SNAPSHOTS = 50


def _load_sidecar_raw(raw_path: str | Path) -> dict:
    """Load the full sidecar envelope (not just settings)."""
    sidecar_path = get_sidecar_path(raw_path)
    if not sidecar_path.exists():
        return {}
    try:
        with open(sidecar_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading sidecar envelope {sidecar_path}: {e}")
        return {}


def _save_sidecar_raw(raw_path: str | Path, data: dict) -> None:
    """Write the full sidecar envelope to disk."""
    sidecar_path = get_sidecar_path(raw_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(data, f, indent=4)


def save_snapshot(
    raw_path: str | Path,
    settings: dict,
    *,
    label: str | None = None,
    is_auto: bool = True,
    is_tagged: bool = False,
) -> dict:
    """Persist a snapshot entry to the sidecar file.

    Returns the created snapshot dict (including its ``id``).
    """
    data = _load_sidecar_raw(raw_path)

    snapshots: list = data.get("snapshots", [])

    snapshot = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "label": label,
        "is_auto": is_auto,
        "is_tagged": is_tagged,
        "settings": settings.copy(),
    }

    snapshots.append(snapshot)

    # Prune oldest auto-save entries beyond the cap
    snapshots = prune_snapshots_list(snapshots, MAX_AUTO_SNAPSHOTS)

    data["snapshots"] = snapshots
    if "version" in data:
        data["version"] = "1.1"

    _save_sidecar_raw(raw_path, data)
    return snapshot


def load_snapshots(raw_path: str | Path) -> list[dict]:
    """Return all snapshot entries for an image, newest first."""
    data = _load_sidecar_raw(raw_path)
    snapshots = data.get("snapshots", [])
    return sorted(snapshots, key=lambda s: s.get("timestamp", 0), reverse=True)


def delete_snapshot(raw_path: str | Path, snapshot_id: str) -> bool:
    """Delete a snapshot by id. Returns True if found and removed."""
    data = _load_sidecar_raw(raw_path)
    snapshots: list = data.get("snapshots", [])
    before = len(snapshots)
    snapshots = [s for s in snapshots if s.get("id") != snapshot_id]
    if len(snapshots) == before:
        return False
    data["snapshots"] = snapshots
    _save_sidecar_raw(raw_path, data)
    return True


def update_snapshot(
    raw_path: str | Path,
    snapshot_id: str,
    *,
    label: str | None = ...,
    is_tagged: bool | None = None,
) -> bool:
    """Update mutable fields on an existing snapshot. Returns True on success."""
    data = _load_sidecar_raw(raw_path)
    snapshots: list = data.get("snapshots", [])
    for snap in snapshots:
        if snap.get("id") == snapshot_id:
            if label is not ...:
                snap["label"] = label
            if is_tagged is not None:
                snap["is_tagged"] = is_tagged
            data["snapshots"] = snapshots
            _save_sidecar_raw(raw_path, data)
            return True
    return False


def prune_snapshots_list(
    snapshots: list[dict], max_auto: int = MAX_AUTO_SNAPSHOTS
) -> list[dict]:
    """Remove oldest auto (non-tagged) snapshots exceeding *max_auto*."""
    auto = [s for s in snapshots if s.get("is_auto") and not s.get("is_tagged")]
    keep = [s for s in snapshots if not s.get("is_auto") or s.get("is_tagged")]

    if len(auto) > max_auto:
        auto.sort(key=lambda s: s.get("timestamp", 0))
        auto = auto[-max_auto:]

    return keep + auto


def format_snapshot_timestamp(timestamp: float) -> str:
    """Format a snapshot timestamp for display."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
