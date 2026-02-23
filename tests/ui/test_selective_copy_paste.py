import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pynegative.ui.settingsmanager import SettingsManager
from pynegative.ui.widgets.selective_paste_dialog import SelectivePasteDialog


@pytest.fixture
def settings_manager():
    return SettingsManager()


def test_settings_manager_selective_copy(settings_manager):
    settings_manager.set_current_path("test.ARW")
    current_settings = {
        "exposure": 1.5,
        "raw_val_exposure": 150,
        "temperature": 5500,
        "raw_val_temperature": 5500,
        "contrast": 20,
        "version": 2,
    }

    # Copy only exposure
    settings_manager.copy_settings_selective(current_settings, ["exposure"])

    assert settings_manager.has_clipboard_content()
    assert settings_manager.settings_clipboard["exposure"] == 1.5
    assert settings_manager.settings_clipboard["raw_val_exposure"] == 150
    assert "temperature" not in settings_manager.settings_clipboard
    assert "contrast" not in settings_manager.settings_clipboard
    assert settings_manager.settings_clipboard["version"] == 2


def test_settings_manager_selective_paste_current(settings_manager):
    target = Path("test.ARW")
    settings_manager.set_current_path(target)

    # Clipboard has exposure and temperature
    settings_manager.settings_clipboard = {
        "exposure": 1.5,
        "raw_val_exposure": 150,
        "temperature": 6000,
        "version": 2,
    }

    # Mock existing settings (e.g. contrast is already set)
    existing = {"exposure": 0.0, "temperature": 5000, "contrast": 10, "rating": 4}

    callback = MagicMock()
    with (
        patch("pynegative.core.load_sidecar", return_value=existing),
        patch("pynegative.core.save_sidecar") as mock_save,
    ):
        # Paste only exposure
        settings_manager.paste_settings_to_current(
            callback, keys_to_include=["exposure"]
        )

        # Verify saved settings
        mock_save.assert_called()
        saved_settings = mock_save.call_args[0][1]
        assert saved_settings["exposure"] == 1.5
        assert saved_settings["temperature"] == 5000  # Preserved
        assert saved_settings["contrast"] == 10  # Preserved
        assert saved_settings["rating"] == 4  # Preserved

        # Verify UI callback received merged settings
        callback.assert_called()
        merged_settings = callback.call_args[0][0]
        assert merged_settings["exposure"] == 1.5
        assert merged_settings["temperature"] == 5000


def test_selective_paste_dialog_logic(qtbot):
    dialog = SelectivePasteDialog()

    # Initially all should be selected (based on my implementation)
    keys = dialog.get_selected_keys()
    assert "exposure" in keys
    assert "temperature" in keys

    # Deselect all
    dialog._select_all(False)
    assert len(dialog.get_selected_keys()) == 0

    # Select a group header checkbox
    tone_group = dialog._group_checkboxes["Tone"]
    tone_group.setChecked(True)

    keys = dialog.get_selected_keys()
    assert "exposure" in keys
    assert "contrast" in keys
    assert "temperature" not in keys  # Different group

    # Toggle individual
    dialog._checkboxes["exposure"].setChecked(False)
    assert "exposure" not in dialog.get_selected_keys()
    assert "contrast" in dialog.get_selected_keys()


def test_settings_manager_selective_paste_raw_rotation(settings_manager):
    settings_manager.set_current_path("test.ARW")
    settings_manager.settings_clipboard = {
        "rotation": 45.0,
        "raw_rotation": 4500,
        "version": 2,
    }

    existing = {"rotation": 0.0, "rating": 0}

    with (
        patch("pynegative.core.load_sidecar", return_value=existing),
        patch("pynegative.core.save_sidecar") as mock_save,
    ):
        settings_manager.paste_settings_to_current(None, keys_to_include=["rotation"])

        saved_settings = mock_save.call_args[0][1]
        assert saved_settings["rotation"] == 45.0
        assert saved_settings["raw_rotation"] == 4500
