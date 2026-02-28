import logging

from PySide6 import QtCore

from ... import core as pynegative

logger = logging.getLogger(__name__)

AUTOSAVE_INTERVAL_MS = 60_000  # 60 seconds


class VersionManager(QtCore.QObject):
    """Manages edit history snapshots (autosave + manual) for images."""

    snapshotSaved = QtCore.Signal(dict)  # the snapshot entry
    snapshotRestored = QtCore.Signal(dict)  # the snapshot entry
    snapshotsChanged = QtCore.Signal()  # generic refresh signal

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self._last_snapshot_settings: dict | None = None

        # 60-second autosave timer
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self._on_autosave_tick)

    # ------------------------------------------------------------------
    # Timer lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the autosave timer (call on image load)."""
        self._last_snapshot_settings = None
        self._autosave_timer.start()

    def stop(self):
        """Stop the autosave timer (call on image unload / switch)."""
        self._autosave_timer.stop()

    # ------------------------------------------------------------------
    # Autosave
    # ------------------------------------------------------------------

    def _on_autosave_tick(self):
        raw_path = self.editor.raw_path
        if raw_path is None:
            return

        current = self._collect_settings()
        if current == self._last_snapshot_settings:
            return  # Nothing changed since last snapshot

        snapshot = pynegative.save_snapshot(
            raw_path,
            current,
            label=None,
            is_auto=True,
            is_tagged=False,
        )
        self._last_snapshot_settings = current
        logger.debug("Autosaved snapshot %s", snapshot["id"])
        self.snapshotSaved.emit(snapshot)
        self.snapshotsChanged.emit()

    # ------------------------------------------------------------------
    # Manual save (Ctrl+S)
    # ------------------------------------------------------------------

    def save_manual_snapshot(self):
        """Create a manual (non-auto) snapshot of the current state."""
        raw_path = self.editor.raw_path
        if raw_path is None:
            return

        current = self._collect_settings()
        snapshot = pynegative.save_snapshot(
            raw_path,
            current,
            label=None,
            is_auto=False,
            is_tagged=False,
        )
        self._last_snapshot_settings = current
        logger.info("Manual snapshot saved: %s", snapshot["id"])
        self.snapshotSaved.emit(snapshot)
        self.snapshotsChanged.emit()
        self.editor.show_toast("Snapshot saved")

    # ------------------------------------------------------------------
    # Tagged version (Ctrl+Shift+S)
    # ------------------------------------------------------------------

    def save_tagged_version(self, label: str | None = None):
        """Create a tagged version of the current state."""
        raw_path = self.editor.raw_path
        if raw_path is None:
            return

        if not label:
            label = pynegative.format_snapshot_timestamp(__import__("time").time())

        current = self._collect_settings()
        snapshot = pynegative.save_snapshot(
            raw_path,
            current,
            label=label,
            is_auto=False,
            is_tagged=True,
        )
        self._last_snapshot_settings = current
        logger.info("Tagged version saved: %s (%s)", snapshot["id"], label)
        self.snapshotSaved.emit(snapshot)
        self.snapshotsChanged.emit()
        self.editor.show_toast(f"Version saved: {label}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_settings(self) -> dict:
        """Gather the full edit state from the editor."""
        settings = self.editor.editing_controls.get_all_settings()
        settings["crop"] = self.editor.image_processor.get_current_settings().get(
            "crop"
        )
        settings["rating"] = self.editor.editing_controls.star_rating_widget.rating()
        return settings

    def load_snapshots(self) -> list[dict]:
        """Load all snapshots for the current image."""
        raw_path = self.editor.raw_path
        if raw_path is None:
            return []
        return pynegative.load_snapshots(raw_path)
