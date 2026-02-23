from PySide6 import QtCore


class BaseSettingsManager(QtCore.QObject):
    """Base class for settings managers."""

    settingsChanged = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_settings = self._get_default_settings()

    def _get_default_settings(self) -> dict:
        """Get default settings. Should be overridden by subclasses."""
        return {}

    def get_current_settings(self) -> dict:
        """Get current settings as a dictionary."""
        return self._current_settings.copy()

    def update_setting(self, key, value):
        """Update a single setting value."""
        if key in self._current_settings:
            # We don't automatically emit if value hasn't changed to avoid loops
            if self._current_settings[key] != value:
                self._current_settings[key] = value
                self.settingsChanged.emit(self._current_settings)

    def update_settings(self, settings_dict: dict):
        """Update multiple settings at once."""
        changed = False
        for k, v in settings_dict.items():
            if k in self._current_settings and self._current_settings[k] != v:
                self._current_settings[k] = v
                changed = True
        if changed:
            self.settingsChanged.emit(self._current_settings)

    def reset_to_defaults(self):
        """Reset settings to defaults."""
        self._current_settings = self._get_default_settings()
        self.settingsChanged.emit(self._current_settings)

    def validate_settings(self) -> list[str]:
        """Validate current settings. Returns list of error messages, empty if valid."""
        return []
