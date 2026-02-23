import pytest
from unittest.mock import MagicMock
from pynegative.ui.exportsettingsmanager import ExportSettingsManager


@pytest.fixture
def manager():
    return ExportSettingsManager()


def test_export_settings_defaults(manager):
    settings = manager.get_current_settings()
    assert settings["format"] == "HEIF"
    assert settings["jpeg_quality"] == 95


def test_update_setting(manager):
    mock_slot = MagicMock()
    manager.settingsChanged.connect(mock_slot)
    manager.update_setting("format", "JPEG")
    assert manager.get_current_settings()["format"] == "JPEG"
    mock_slot.assert_called()


def test_presets(manager):
    presets = manager.load_presets()
    assert "Web" in presets

    web_settings = manager.apply_preset("Web")
    assert web_settings["format"] == "JPEG"
    assert web_settings["max_width"] == "1920"

    archival = manager.apply_preset("Archival")
    assert archival["format"] == "HEIF"


def test_custom_presets(manager):
    custom_settings = {"format": "JPEG", "jpeg_quality": 50}
    manager.save_preset("MyTestPreset", custom_settings)

    presets = manager.load_presets()
    assert "MyTestPreset" in presets

    loaded = manager.get_preset("MyTestPreset")
    assert loaded["jpeg_quality"] == 50

    manager.delete_preset("MyTestPreset")
    presets_after = manager.load_presets()
    assert "MyTestPreset" not in presets_after


def test_destination(manager, tmp_path):
    dest = manager.get_destination(use_default=True, gallery_folder=str(tmp_path))
    assert "exported" in dest
    assert (tmp_path / "exported").exists()

    manager.set_custom_destination("/tmp/custom")
    assert manager.get_destination(use_default=False) == "/tmp/custom"


def test_validate_settings(manager):
    manager.update_setting("max_width", "abc")
    errors = manager.validate_settings()
    assert "Max width must be a positive integer" in errors[0]

    manager.update_setting("max_width", "1024")
    assert not manager.validate_settings()


def test_rename_settings(manager):
    manager.set_rename_enabled(True)
    assert manager.get_rename_settings()["rename_enabled"] is True

    manager.update_rename_settings({"rename_prefix": "Test"})
    assert manager.get_rename_settings()["rename_prefix"] == "Test"
