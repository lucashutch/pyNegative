from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from PySide6 import QtCore
import rawpy


class RenameSettingsManager(QtCore.QObject):
    """Manages batch renaming settings and preview generation for exports."""

    # Naming pattern presets
    PATTERNS = {
        "Prefix + Date + Sequence": "{prefix}_{date}_{seq:03d}",
        "Date + Prefix + Sequence": "{date}_{prefix}_{seq:03d}",
        "Prefix + Sequence": "{prefix}_{seq:03d}",
        "Date + Sequence": "{date}_{seq:03d}",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_settings = self._get_default_settings()

    def _get_default_settings(self):
        """Get default rename settings."""
        return {
            "enabled": False,
            "pattern": "Prefix + Sequence",
            "prefix": "",
            "start_seq": 1,
        }

    def get_current_settings(self):
        """Get current rename settings as a dictionary."""
        return self._current_settings.copy()

    def update_setting(self, key, value):
        """Update a single rename setting."""
        if key in self._current_settings:
            self._current_settings[key] = value

    def set_enabled(self, enabled: bool):
        """Enable or disable renaming."""
        self._current_settings["enabled"] = enabled

    def is_enabled(self) -> bool:
        """Check if renaming is enabled."""
        return self._current_settings.get("enabled", False)

    def get_pattern_names(self) -> List[str]:
        """Get list of available pattern names."""
        return list(self.PATTERNS.keys())

    def get_exif_date(self, raw_path: str) -> Optional[str]:
        """Extract capture date from RAW file EXIF data.

        Returns date as YYYY-MM-DD string or None if unavailable.
        Falls back to file modification date if EXIF date unavailable.
        """
        try:
            with rawpy.imread(raw_path) as raw:
                # Try to get DateTimeOriginal from EXIF
                if hasattr(raw, "extract_exif") or hasattr(raw, "raw_image"):
                    # rawpy doesn't directly expose EXIF, but we can access it
                    # through the raw structure if available
                    exif_data = None
                    if hasattr(raw, "extract_exif"):
                        try:
                            exif_data = raw.extract_exif()
                        except Exception:
                            pass

                    # If we have EXIF data, parse it
                    if exif_data:
                        # Parse DateTimeOriginal from EXIF
                        # This is a simplified parser - in production you might
                        # want to use a more robust EXIF library
                        return self._parse_exif_date(exif_data)

            # Fallback to file modification time
            mtime = Path(raw_path).stat().st_mtime
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

        except Exception as e:
            print(f"Error reading EXIF date from {raw_path}: {e}")
            # Fallback to file modification time
            try:
                mtime = Path(raw_path).stat().st_mtime
                return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            except (OSError, IOError):
                return None

    def _parse_exif_date(self, exif_data: bytes) -> Optional[str]:
        """Parse DateTimeOriginal from EXIF data.

        This is a basic parser. For more robust parsing, consider using
        a dedicated EXIF library.
        """
        try:
            # Look for DateTimeOriginal tag (0x9003) in EXIF
            # Format is typically: "2024:01:15 14:30:00"
            exif_str = exif_data.decode("utf-8", errors="ignore")

            # Search for common date patterns
            import re

            # Match YYYY:MM:DD or YYYY-MM-DD patterns
            date_match = re.search(r"(\d{4})[:/-](\d{2})[:/-](\d{2})", exif_str)
            if date_match:
                year, month, day = date_match.groups()
                return f"{year}-{month}-{day}"

        except Exception as e:
            print(f"Error parsing EXIF date: {e}")

        return None

    def generate_preview(
        self,
        files: List[Path],
        pattern_name: str,
        prefix: str,
        start_seq: int,
        destination: Path,
        format_ext: str,
    ) -> List[Tuple[str, str, Optional[str]]]:
        """Generate preview of renamed files.

        Args:
            files: List of source RAW file paths
            pattern_name: Name of the pattern preset to use
            prefix: Custom prefix text
            start_seq: Starting sequence number
            destination: Export destination folder
            format_ext: Export format extension (e.g., "jpg" or "heic")

        Returns:
            List of tuples: (original_name, new_name, warning_message)
            warning_message is None if no issues, otherwise describes the conflict
        """
        if pattern_name not in self.PATTERNS:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        pattern = self.PATTERNS[pattern_name]
        preview = []
        used_names = set()

        for i, file_path in enumerate(files):
            seq = start_seq + i
            original_name = file_path.name

            # Get date from EXIF
            date_str = self.get_exif_date(str(file_path))
            if date_str is None:
                date_str = "unknown-date"

            # Generate new filename
            try:
                new_name = pattern.format(
                    prefix=prefix,
                    date=date_str,
                    seq=seq,
                )
            except Exception as e:
                preview.append((original_name, "", f"Pattern error: {e}"))
                continue

            # Add extension
            new_filename = f"{new_name}.{format_ext}"

            # Check for conflicts
            warning = None
            target_path = destination / new_filename

            if target_path.exists():
                warning = "File exists - will skip"
            elif new_filename in used_names:
                warning = "Duplicate name generated"

            used_names.add(new_filename)
            preview.append((original_name, new_filename, warning))

        return preview

    def validate_settings(self) -> List[str]:
        """Validate current rename settings.

        Returns list of error messages, empty if valid.
        """
        errors = []
        settings = self._current_settings

        if not settings.get("enabled"):
            return errors

        pattern = settings.get("pattern", "")
        if pattern not in self.PATTERNS:
            errors.append(f"Invalid pattern: {pattern}")

        prefix = settings.get("prefix", "")
        if not prefix:
            errors.append("Prefix cannot be empty when renaming is enabled")

        # Check for invalid filename characters in prefix
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            if char in prefix:
                errors.append(f"Prefix contains invalid character: '{char}'")
                break

        start_seq = settings.get("start_seq", 1)
        if not isinstance(start_seq, int) or start_seq < 1:
            errors.append("Start number must be a positive integer")

        return errors

    def create_rename_mapping(
        self,
        files: List[Path],
        preview: List[Tuple[str, str, Optional[str]]],
    ) -> dict:
        """Create a mapping from source file to target filename.

        Only includes files without warnings (conflicts).

        Returns:
            Dictionary mapping source Path to target filename string
        """
        mapping = {}
        for i, file_path in enumerate(files):
            if i < len(preview):
                _, new_name, warning = preview[i]
                if not warning and new_name:  # Only map non-conflicting files
                    mapping[file_path] = new_name
        return mapping
