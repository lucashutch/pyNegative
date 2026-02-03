"""Test open folder on complete checkbox functionality."""

from pynegative.ui.exportsettingsmanager import ExportSettingsManager


def test_open_folder_default_setting():
    """Test that open_folder_on_complete defaults to False when no saved settings exist."""
    settings_manager = ExportSettingsManager()

    # Clear any persisted setting first
    settings_manager.settings.remove("open_folder_on_complete")

    # Create a fresh instance to test default value
    settings_manager2 = ExportSettingsManager()
    current_settings = settings_manager2.get_current_settings()
    assert current_settings["open_folder_on_complete"] is False


def test_open_folder_update_setting():
    """Test updating the open_folder_on_complete setting."""
    settings_manager = ExportSettingsManager()

    # Update to True
    settings_manager.update_setting("open_folder_on_complete", True)
    current_settings = settings_manager.get_current_settings()
    assert current_settings["open_folder_on_complete"] is True

    # Update to False
    settings_manager.update_setting("open_folder_on_complete", False)
    current_settings = settings_manager.get_current_settings()
    assert current_settings["open_folder_on_complete"] is False


def test_open_folder_persists():
    """Test that open_folder_on_complete setting persists to QSettings."""
    # Create first instance and set to True
    settings_manager1 = ExportSettingsManager()
    settings_manager1.update_setting("open_folder_on_complete", True)

    # Create second instance and verify it loads True
    settings_manager2 = ExportSettingsManager()
    current_settings = settings_manager2.get_current_settings()
    assert current_settings["open_folder_on_complete"] is True

    # Clean up - reset to False
    settings_manager2.update_setting("open_folder_on_complete", False)
