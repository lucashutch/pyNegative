"""Tests for the VersionManager editor manager."""

from unittest.mock import MagicMock, patch

import pytest

from pynegative.ui.editor_managers.version_manager import VersionManager


@pytest.fixture()
def mock_editor():
    """Create a minimal mock editor for VersionManager."""
    editor = MagicMock()
    editor.raw_path = "/tmp/test.CR3"
    editor.editing_controls.get_all_settings.return_value = {"exposure": 0.5}
    editor.editing_controls.star_rating_widget.rating.return_value = 3
    editor.image_processor.get_current_settings.return_value = {"crop": None}
    return editor


@pytest.fixture()
def version_manager(mock_editor, qapp):
    vm = VersionManager(mock_editor)
    return vm


class TestVersionManagerLifecycle:
    def test_start_starts_timer(self, version_manager):
        version_manager.start()
        assert version_manager._autosave_timer.isActive()

    def test_stop_stops_timer(self, version_manager):
        version_manager.start()
        version_manager.stop()
        assert not version_manager._autosave_timer.isActive()


class TestManualSnapshot:
    @patch("pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot")
    def test_save_manual_snapshot(self, mock_save, version_manager):
        mock_save.return_value = {"id": "abc", "is_auto": False}
        version_manager.save_manual_snapshot()

        mock_save.assert_called_once()
        call_kwargs = mock_save.call_args
        assert call_kwargs[1]["is_auto"] is False
        assert call_kwargs[1]["is_tagged"] is False
        version_manager.editor.show_toast.assert_called_once()

    def test_no_save_when_no_raw_path(self, version_manager):
        version_manager.editor.raw_path = None
        with patch(
            "pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot"
        ) as mock_save:
            version_manager.save_manual_snapshot()
            mock_save.assert_not_called()


class TestTaggedVersion:
    @patch("pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot")
    def test_save_tagged_version(self, mock_save, version_manager):
        mock_save.return_value = {"id": "def", "is_tagged": True}
        version_manager.save_tagged_version("My Edit")

        mock_save.assert_called_once()
        call_kwargs = mock_save.call_args
        assert call_kwargs[1]["is_tagged"] is True
        assert call_kwargs[1]["label"] == "My Edit"

    @patch("pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot")
    @patch(
        "pynegative.ui.editor_managers.version_manager.pynegative.format_snapshot_timestamp"
    )
    def test_default_label_when_none(self, mock_fmt, mock_save, version_manager):
        mock_fmt.return_value = "2026-02-28 14:30"
        mock_save.return_value = {"id": "ghi", "is_tagged": True}
        version_manager.save_tagged_version(None)

        call_kwargs = mock_save.call_args
        assert call_kwargs[1]["label"] == "2026-02-28 14:30"


class TestAutoSaveTick:
    @patch("pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot")
    def test_autosave_creates_snapshot(self, mock_save, version_manager):
        mock_save.return_value = {"id": "auto1", "is_auto": True}
        version_manager._on_autosave_tick()

        mock_save.assert_called_once()
        call_kwargs = mock_save.call_args
        assert call_kwargs[1]["is_auto"] is True

    @patch("pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot")
    def test_autosave_skips_if_unchanged(self, mock_save, version_manager):
        mock_save.return_value = {"id": "auto1", "is_auto": True}
        version_manager._on_autosave_tick()
        mock_save.reset_mock()

        # Same settings, should skip
        version_manager._on_autosave_tick()
        mock_save.assert_not_called()

    @patch("pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot")
    def test_autosave_fires_on_settings_change(self, mock_save, version_manager):
        mock_save.return_value = {"id": "auto1", "is_auto": True}
        version_manager._on_autosave_tick()
        mock_save.reset_mock()

        # Change settings
        version_manager.editor.editing_controls.get_all_settings.return_value = {
            "exposure": 0.8
        }
        mock_save.return_value = {"id": "auto2", "is_auto": True}
        version_manager._on_autosave_tick()
        mock_save.assert_called_once()

    def test_autosave_noop_without_raw_path(self, version_manager):
        version_manager.editor.raw_path = None
        with patch(
            "pynegative.ui.editor_managers.version_manager.pynegative.save_snapshot"
        ) as mock_save:
            version_manager._on_autosave_tick()
            mock_save.assert_not_called()


class TestCollectSettings:
    def test_includes_crop_and_rating(self, version_manager):
        version_manager.editor.image_processor.get_current_settings.return_value = {
            "crop": (0.1, 0.2, 0.9, 0.8)
        }
        result = version_manager._collect_settings()
        assert result["crop"] == (0.1, 0.2, 0.9, 0.8)
        assert result["rating"] == 3


class TestLoadSnapshots:
    @patch("pynegative.ui.editor_managers.version_manager.pynegative.load_snapshots")
    def test_load_snapshots_delegates(self, mock_load, version_manager):
        mock_load.return_value = [{"id": "1"}]
        result = version_manager.load_snapshots()
        assert result == [{"id": "1"}]

    def test_empty_when_no_raw_path(self, version_manager):
        version_manager.editor.raw_path = None
        assert version_manager.load_snapshots() == []
