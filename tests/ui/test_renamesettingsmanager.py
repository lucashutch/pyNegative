import pytest
from pathlib import Path
from unittest.mock import patch
from pynegative.ui.renamesettingsmanager import RenameSettingsManager


@pytest.fixture
def manager():
    return RenameSettingsManager()


def test_rename_settings_defaults(manager):
    settings = manager.get_current_settings()
    assert settings["enabled"] is False
    assert settings["prefix"] == ""
    assert manager.is_enabled() is False


def test_pattern_names(manager):
    names = manager.get_pattern_names()
    assert "Prefix + Sequence" in names
    assert len(names) >= 4


def test_get_exif_date_fallback(manager, tmp_path):
    path = tmp_path / "test.ARW"
    path.touch()

    with patch("rawpy.imread") as mock_imread:
        # Mocking rawpy error to trigger fallback
        mock_imread.side_effect = Exception("failed")
        date = manager.get_exif_date(str(path))
        assert date is not None
        assert len(date) == 10  # YYYY-MM-DD


def test_generate_preview(manager, tmp_path):
    files = [tmp_path / "img1.ARW", tmp_path / "img2.ARW"]
    for f in files:
        f.touch()

    with patch.object(manager, "get_exif_date", return_value="2024-01-01"):
        preview = manager.generate_preview(
            files,
            pattern_name="Prefix + Sequence",
            prefix="Vacation",
            start_seq=1,
            destination=tmp_path / "export",
            format_ext="jpg",
        )

        assert len(preview) == 2
        assert preview[0][1] == "Vacation_001.jpg"
        assert preview[1][1] == "Vacation_002.jpg"


def test_validate_settings(manager):
    manager.set_enabled(True)
    manager.update_setting("prefix", "")
    errors = manager.validate_settings()
    assert "Prefix cannot be empty" in errors[0]

    manager.update_setting("prefix", "Valid")
    assert not manager.validate_settings()

    manager.update_setting("prefix", "invalid/char")
    assert "invalid character" in manager.validate_settings()[0]


def test_create_rename_mapping(manager):
    files = [Path("a.ARW"), Path("b.ARW")]
    preview = [("a.ARW", "new_a.jpg", None), ("b.ARW", "new_b.jpg", "Conflict")]

    mapping = manager.create_rename_mapping(files, preview)
    assert len(mapping) == 1
    assert mapping[Path("a.ARW")] == "new_a.jpg"
