from pathlib import Path

from PySide6 import QtCore

from .. import core as pynegative
from .undomanager import UndoManager


class SettingsManager(QtCore.QObject):
    # Signals
    settingsCopied = QtCore.Signal(str, dict)  # source_path, settings
    settingsPasted = QtCore.Signal(list, dict)  # target_paths, settings
    undoStateChanged = QtCore.Signal()
    showToast = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Settings clipboard - Settings clipboard
        self.settings_clipboard = None
        self.clipboard_source_path = None

        # Undo/Redo system
        self.undo_manager = UndoManager()
        self.undo_timer = QtCore.QTimer()
        self.undo_timer.setSingleShot(True)
        self.undo_timer.timeout.connect(self._push_undo_state)
        self._undo_state_description = ""
        self._undo_timer_active = False

        # Current state
        self.current_rating = 0
        self.current_path = None

    def set_current_path(self, path):
        """Set the current image path."""
        self.current_path = Path(path) if path else None

    def set_current_settings(self, settings, rating):
        """Set current settings and rating for undo/redo."""
        self.current_rating = rating

    def copy_settings_from_current(self, current_settings):
        """Copy settings from currently loaded photo."""
        if not self.current_path:
            return

        settings = current_settings.copy()
        self.settings_clipboard = settings
        self.clipboard_source_path = self.current_path
        self.showToast.emit(f"Settings copied from {self.current_path.name}")
        self.settingsCopied.emit(str(self.current_path), settings)

    def copy_settings_from_path(self, path):
        """Copy settings from a specific photo by path."""
        settings = pynegative.load_sidecar(path)
        if not settings:
            return

        # Remove rating from settings (we don't want to sync rating)
        settings_copy = settings.copy()
        settings_copy.pop("rating", None)

        self.settings_clipboard = settings_copy
        self.clipboard_source_path = Path(path)
        self.showToast.emit(f"Settings copied from {Path(path).name}")
        self.settingsCopied.emit(str(path), settings_copy)

    def paste_settings_to_current(self, current_settings_callback):
        """Paste settings to currently loaded photo."""
        if not self.settings_clipboard or not self.current_path:
            return

        self._apply_settings_to_photo(self.current_path, self.settings_clipboard)
        self.showToast.emit("Settings applied to current photo")

        # Apply to UI through callback
        if current_settings_callback:
            current_settings_callback(self.settings_clipboard)

        self.settingsPasted.emit([str(self.current_path)], self.settings_clipboard)

    def paste_settings_to_selected(
        self, selected_paths, current_settings_callback=None
    ):
        """Paste settings to all selected paths."""
        if not self.settings_clipboard:
            return

        if not selected_paths:
            return

        # Apply to each selected photo
        for path_str in selected_paths:
            path = Path(path_str)
            self._apply_settings_to_photo(path, self.settings_clipboard)

        # If current photo is among selected, apply immediately
        if self.current_path and str(self.current_path) in selected_paths:
            if current_settings_callback:
                current_settings_callback(self.settings_clipboard)

        # Push undo state for the batch operation
        self._push_undo_state_immediate(
            f"Paste settings to {len(selected_paths)} photos"
        )

        self.showToast.emit(f"Settings applied to {len(selected_paths)} photos")
        self.settingsPasted.emit(selected_paths, self.settings_clipboard)

    def has_clipboard_content(self):
        """Check if clipboard has settings."""
        return self.settings_clipboard is not None

    def clear_clipboard(self):
        """Clear settings clipboard."""
        self.settings_clipboard = None
        self.clipboard_source_path = None

    def schedule_undo_state(self, description, current_settings):
        """Schedule undo state push with batching."""
        self._undo_state_description = description
        self._current_settings_for_undo = current_settings
        self.undo_timer.start(1000)  # Batch within 1 second
        self._undo_timer_active = True

    def push_immediate_undo_state(self, description, current_settings):
        """Push undo state immediately."""
        settings = current_settings.copy()
        self.undo_manager.push_state(description, settings, self.current_rating)
        self.undoStateChanged.emit()

    def undo(self):
        """Handle undo action."""
        state = self.undo_manager.undo()
        if state:
            self.showToast.emit(f"Undone: {state['description']}")
            self.undoStateChanged.emit()
            return state
        return None

    def redo(self):
        """Handle redo action."""
        state = self.undo_manager.redo()
        if state:
            self.showToast.emit(f"Redone: {state['description']}")
            self.undoStateChanged.emit()
            return state
        return None

    def auto_save_sidecar(self, path, settings, rating):
        """Auto-save settings to sidecar file."""
        if not path:
            return

        save_settings = settings.copy()
        save_settings["rating"] = rating
        pynegative.save_sidecar(path, save_settings)

    def _apply_settings_to_photo(self, path, settings):
        """Apply settings to a photo by saving to its sidecar."""
        # Load existing sidecar to preserve rating
        existing_settings = pynegative.load_sidecar(path) or {}
        rating = existing_settings.get("rating", 0)

        # Apply new settings but preserve rating
        combined_settings = settings.copy()
        combined_settings["rating"] = rating

        pynegative.save_sidecar(path, combined_settings)

    def _push_undo_state(self):
        """Push undo state after delay."""
        if not self._undo_state_description:
            return

        if hasattr(self, "_current_settings_for_undo"):
            settings = self._current_settings_for_undo.copy()
            self.undo_manager.push_state(
                self._undo_state_description, settings, self.current_rating
            )

        self._undo_state_description = ""
        self._undo_timer_active = False
        self.undoStateChanged.emit()

    def _push_undo_state_immediate(self, description):
        """Push undo state immediately with current settings."""
        if hasattr(self, "_current_settings_for_undo"):
            settings = self._current_settings_for_undo.copy()
            self.undo_manager.push_state(description, settings, self.current_rating)
            self.undoStateChanged.emit()
