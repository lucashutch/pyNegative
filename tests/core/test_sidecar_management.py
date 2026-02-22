#!/usr/bin/env python3
"""Unit tests for sidecar file management functions in pynegative.core"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from pynegative import core


@pytest.fixture
def temp_raw_path():
    """Provides a temporary RAW file path and ensures cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = Path(tmpdir) / "test_image.cr2"
        raw_file.touch()  # Create a dummy raw file
        yield raw_file
        # Cleanup is handled by TemporaryDirectory context manager


@pytest.fixture
def temp_sidecar_dir(temp_raw_path):
    """Provides the path to the sidecar directory for a given raw path."""
    return temp_raw_path.parent / core.SIDECAR_DIR


class TestSidecars:
    """Tests for sidecar file logic"""

    def test_get_sidecar_path(self, temp_raw_path):
        """Test the get_sidecar_path function."""
        expected_path = (
            temp_raw_path.parent / core.SIDECAR_DIR / f"{temp_raw_path.name}.json"
        )
        actual_path = core.get_sidecar_path(temp_raw_path)
        assert actual_path == expected_path

    def test_save_load_sidecar_basic_settings(self, temp_raw_path):
        """Test saving and loading a full set of settings."""
        settings = {
            "rating": 3,
            "exposure": 0.5,
            "contrast": 1.2,
            "blacks": 0.1,
            "whites": 0.9,
            "shadows": 0.2,
            "highlights": -0.3,
            "saturation": 1.1,
            "sharpen_enabled": True,
            "sharpen_radius": 2.5,
            "sharpen_percent": 150,
            "de_noise": 2,
        }
        core.save_sidecar(temp_raw_path, settings)
        loaded_settings = core.load_sidecar(temp_raw_path)

        assert loaded_settings == settings

    def test_load_non_existent_sidecar(self, temp_raw_path):
        """Test loading a sidecar that does not exist."""
        loaded_settings = core.load_sidecar(temp_raw_path)
        assert loaded_settings is None

    def test_load_corrupted_sidecar(self, temp_raw_path, temp_sidecar_dir):
        """Test loading a sidecar with invalid JSON content."""
        sidecar_path = core.get_sidecar_path(temp_raw_path)
        temp_sidecar_dir.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w") as f:
            f.write("this is not valid json {")  # Corrupted JSON

        loaded_settings = core.load_sidecar(temp_raw_path)
        assert loaded_settings is None

    def test_load_sidecar_with_missing_keys(self, temp_raw_path, temp_sidecar_dir):
        """Test loading a sidecar where some keys are missing, expecting defaults."""
        # Simulate an older sidecar without 'rating', 'de_noise', 'sharpen_enabled'
        # and with only a subset of tone settings
        old_settings_subset = {
            "exposure": 0.1,
            "contrast": 1.1,
            "blacks": 0.05,
            "whites": 0.95,
            "saturation": 1.05,
        }

        # Manually create the sidecar data to control its content
        sidecar_data = {
            "version": "1.0",
            "last_modified": time.time(),
            "raw_path": str(temp_raw_path),
            "settings": old_settings_subset,
        }
        sidecar_path = core.get_sidecar_path(temp_raw_path)
        temp_sidecar_dir.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w") as f:
            json.dump(sidecar_data, f, indent=4)

        loaded_settings = core.load_sidecar(temp_raw_path)

        # Assert that 'rating' is added with default 0 by load_sidecar
        assert loaded_settings is not None
        assert loaded_settings["rating"] == 0
        assert loaded_settings["exposure"] == 0.1
        assert loaded_settings["contrast"] == 1.1
        assert loaded_settings["blacks"] == 0.05
        assert loaded_settings["whites"] == 0.95
        assert loaded_settings["saturation"] == 1.05

        # Other keys that were not in the original subset should not be present
        # as load_sidecar only adds 'rating' if missing.
        assert "shadows" not in loaded_settings
        assert "highlights" not in loaded_settings
        assert "sharpen_enabled" not in loaded_settings
        assert "sharpen_radius" not in loaded_settings
        assert "sharpen_percent" not in loaded_settings
        assert "de_noise" not in loaded_settings

    def test_save_load_empty_settings(self, temp_raw_path):
        """Test saving and loading an empty settings dictionary."""
        settings = {}
        core.save_sidecar(temp_raw_path, settings)
        loaded_settings = core.load_sidecar(temp_raw_path)

        # 'rating' should be added by save_sidecar and load_sidecar if missing
        assert loaded_settings == {"rating": 0}

    def test_rename_sidecar(self, temp_raw_path):
        """Test renaming a sidecar file."""
        old_raw = temp_raw_path
        new_raw = temp_raw_path.parent / "new_image.cr2"
        new_raw.touch()  # Create dummy new raw file
        settings = {"exposure": 1.5, "rating": 2}

        core.save_sidecar(old_raw, settings)
        old_sidecar = core.get_sidecar_path(old_raw)
        assert old_sidecar.exists()

        core.rename_sidecar(old_raw, new_raw)

        new_sidecar = core.get_sidecar_path(new_raw)
        assert new_sidecar.exists()
        assert not old_sidecar.exists()

        # Verify data survived
        loaded = core.load_sidecar(new_raw)
        assert loaded == settings
