import logging
from datetime import datetime
from pathlib import Path

from PIL import Image

# Import STD_EXTS from raw.py to avoid duplication if possible,
# but for now I'll just redefine it or import it.
from .raw import STD_EXTS

logger = logging.getLogger(__name__)


def get_exif_capture_date(raw_path: str | Path) -> str | None:
    """
    Extracts the capture date from RAW or standard image file EXIF data.

    Returns the date as a string in YYYY-MM-DD format, or None if unavailable.
    Falls back to file modification date if EXIF date is not found.
    """

    raw_path = Path(raw_path)
    ext = raw_path.suffix.lower()

    try:
        if ext in STD_EXTS:
            with Image.open(raw_path) as img:
                exif = img.getexif()
                if exif:
                    # 306 = DateTime, 36867 = DateTimeOriginal
                    for tag in (36867, 306):
                        date_str = exif.get(tag)
                        if date_str and isinstance(date_str, str):
                            # Format: "YYYY:MM:DD HH:MM:SS"
                            try:
                                parts = date_str.split(" ")[0].split(":")
                                if len(parts) == 3:
                                    return f"{parts[0]}-{parts[1]}-{parts[2]}"
                            except Exception:
                                pass

        # Fallback: use file modification time
        mtime = raw_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    except Exception as e:
        logger.error(f"Error reading file {raw_path}: {e}")
        return None


def format_date(timestamp: float) -> str:
    """Formats a timestamp as YYYY-MM-DD."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
