from pathlib import Path
from PySide6 import QtCore


class ExportSettingsManager(QtCore.QObject):
    """Manages export settings, presets, and destination configuration."""

    # Signals
    settingsChanged = QtCore.Signal(dict)
    presetApplied = QtCore.Signal(str)
    presetSaved = QtCore.Signal(str)
    destinationChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QtCore.QSettings("pyNegative", "Export")
        self._current_settings = self._get_default_settings()
        self._custom_destination = ""

    def _get_default_settings(self):
        """Get default export settings."""
        return {
            "format": "HEIF",
            "jpeg_quality": 95,
            "heif_quality": 95,
            "heif_bit_depth": "8-bit",
            "max_width": "",
            "max_height": "",
            "rename_enabled": False,
            "rename_pattern": "Prefix + Date + Sequence",
            "rename_prefix": "",
            "rename_start_seq": 1,
        }

    def get_current_settings(self):
        """Get current export settings as a dictionary."""
        return self._current_settings.copy()

    def update_setting(self, key, value):
        """Update a single setting value."""
        if key in self._current_settings:
            self._current_settings[key] = value
            self.settingsChanged.emit(self._current_settings)

    def update_settings(self, settings_dict):
        """Update multiple settings at once."""
        self._current_settings.update(settings_dict)
        self.settingsChanged.emit(self._current_settings)

    def load_presets(self):
        """Load all available preset names."""
        presets = ["Custom", "Web", "Photo Print", "Archival", "Large Format Print"]

        self.settings.beginGroup("presets")
        custom_presets = list(self.settings.childKeys())
        self.settings.endGroup()

        presets.extend(custom_presets)
        return presets

    def get_preset(self, preset_name):
        """Get settings for a specific preset."""
        if preset_name == "Web":
            return {
                "format": "JPEG",
                "jpeg_quality": 80,
                "max_width": "1920",
                "max_height": "1080",
                "heif_quality": 90,
                "heif_bit_depth": "8-bit",
            }
        elif preset_name == "Photo Print":
            return {
                "format": "JPEG",
                "jpeg_quality": 95,
                "max_width": "3600",
                "max_height": "2400",
                "heif_quality": 90,
                "heif_bit_depth": "8-bit",
            }
        elif preset_name == "Archival":
            return {
                "format": "HEIF",
                "heif_quality": 95,
                "max_width": "",
                "max_height": "",
                "jpeg_quality": 95,
                "heif_bit_depth": "8-bit",
            }
        elif preset_name == "Large Format Print":
            return {
                "format": "JPEG",
                "jpeg_quality": 100,
                "max_width": "10800",
                "max_height": "7200",
                "heif_quality": 95,
                "heif_bit_depth": "8-bit",
            }
        elif preset_name == "Custom":
            return self._get_default_settings()
        else:
            # Try to load custom preset
            self.settings.beginGroup("presets")
            preset = self.settings.value(preset_name)
            self.settings.endGroup()
            return preset if preset else self._get_default_settings()

    def apply_preset(self, preset_name):
        """Apply a preset to current settings."""
        settings = self.get_preset(preset_name)
        self._current_settings.update(settings)
        self.presetApplied.emit(preset_name)
        self.settingsChanged.emit(self._current_settings)
        return settings

    def save_preset(self, preset_name, settings=None):
        """Save current or provided settings as a preset."""
        if settings is None:
            settings = self._current_settings.copy()

        self.settings.beginGroup("presets")
        self.settings.setValue(preset_name, settings)
        self.settings.endGroup()

        self.presetSaved.emit(preset_name)

    def delete_preset(self, preset_name):
        """Delete a custom preset."""
        self.settings.beginGroup("presets")
        self.settings.remove(preset_name)
        self.settings.endGroup()

    def reset_to_defaults(self):
        """Reset settings to defaults."""
        self._current_settings = self._get_default_settings()
        self.settingsChanged.emit(self._current_settings)

    def get_destination(self, use_default=True, gallery_folder=None):
        """Get export destination folder."""
        if use_default and gallery_folder:
            default_dest = Path(gallery_folder) / "exported"
            default_dest.mkdir(exist_ok=True)
            return str(default_dest)

        if self._custom_destination:
            return self._custom_destination

        return None

    def set_custom_destination(self, folder_path):
        """Set custom destination folder."""
        self._custom_destination = folder_path
        self.settings.setValue("custom_destination", folder_path)
        self.destinationChanged.emit(folder_path)

    def get_custom_destination(self):
        """Get custom destination folder path."""
        return self._custom_destination or self.settings.value(
            "custom_destination", "", type=str
        )

    def load_destination_settings(self):
        """Load destination-related settings from persistent storage."""
        self._custom_destination = self.settings.value(
            "custom_destination", "", type=str
        )
        use_default = self.settings.value("use_default_destination", True, type=bool)
        return {
            "use_default_destination": use_default,
            "custom_destination": self._custom_destination,
        }

    def save_destination_settings(self, use_default):
        """Save destination mode preference."""
        self.settings.setValue("use_default_destination", use_default)

    def get_supported_formats(self):
        """Get list of supported export formats."""
        return ["JPEG", "HEIF"]

    def validate_settings(self):
        """Validate current export settings."""
        errors = []

        format = self._current_settings.get("format")
        if format not in self.get_supported_formats():
            errors.append(f"Unsupported format: {format}")

        # Validate dimensions if provided
        max_width = self._current_settings.get("max_width", "")
        max_height = self._current_settings.get("max_height", "")

        if max_width and not max_width.isdigit():
            errors.append("Max width must be a positive integer")
        if max_height and not max_height.isdigit():
            errors.append("Max height must be a positive integer")

        return errors

    def get_rename_settings(self):
        """Get current rename settings as a dictionary."""
        return {
            "rename_enabled": self._current_settings.get("rename_enabled", False),
            "rename_pattern": self._current_settings.get(
                "rename_pattern", "Prefix + Date + Sequence"
            ),
            "rename_prefix": self._current_settings.get("rename_prefix", ""),
            "rename_start_seq": self._current_settings.get("rename_start_seq", 1),
        }

    def set_rename_enabled(self, enabled):
        """Enable or disable renaming."""
        self._current_settings["rename_enabled"] = enabled
        self.settingsChanged.emit(self._current_settings)

    def update_rename_settings(self, settings_dict):
        """Update rename settings."""
        for key in ["rename_pattern", "rename_prefix", "rename_start_seq"]:
            if key in settings_dict:
                self._current_settings[key] = settings_dict[key]
        self.settingsChanged.emit(self._current_settings)
