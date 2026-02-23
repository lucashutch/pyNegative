import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pynegative.ui.settingsmanager import SettingsManager


@pytest.fixture
def settings_manager(qtbot):
    return SettingsManager()


def test_settings_clipboard(settings_manager):
    settings_manager.set_current_path("test.ARW")
    current_settings = {"exposure": 1.5}

    settings_manager.copy_settings_from_current(current_settings)
    assert settings_manager.has_clipboard_content()
    assert settings_manager.settings_clipboard["exposure"] == 1.5

    # Test paste to current
    callback = MagicMock()
    settings_manager.paste_settings_to_current(callback)
    callback.assert_called_with(current_settings)


def test_copy_settings_from_path(settings_manager):
    with patch("pynegative.core.load_sidecar") as mock_load:
        mock_load.return_value = {"exposure": 1.0, "rating": 5}
        settings_manager.copy_settings_from_path("other.ARW")

        assert settings_manager.settings_clipboard["exposure"] == 1.0
        assert "rating" not in settings_manager.settings_clipboard


def test_undo_redo_scheduling(settings_manager, qtbot):
    settings = {"v": 1}
    settings_manager.schedule_undo_state("test", settings)
    assert settings_manager._undo_timer_active

    # Wait for timer
    qtbot.wait(1100)
    assert not settings_manager._undo_timer_active
    assert settings_manager.undo_manager.can_undo() is False  # First push is index 0


def test_auto_save(settings_manager):
    with patch("pynegative.core.save_sidecar") as mock_save:
        settings_manager.auto_save_sidecar(Path("test.ARW"), {"x": 1}, 4)
        mock_save.assert_called()
        args = mock_save.call_args[0]
        assert args[1]["rating"] == 4


def test_paste_settings_to_selected(settings_manager):
    settings_manager.settings_clipboard = {"val": 10}
    paths = ["a.ARW", "b.ARW"]

    with (
        patch("pynegative.core.save_sidecar") as mock_save,
        patch("pynegative.core.load_sidecar", return_value={"rating": 3}),
    ):
        settings_manager.paste_settings_to_selected(paths)
        assert mock_save.call_count == 2
        # Check rating preserved
        args = mock_save.call_args[0]
        assert args[1]["rating"] == 3
