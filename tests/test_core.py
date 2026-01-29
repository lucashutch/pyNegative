#!/usr/bin/env python3
"""Unit tests for pynegative.py"""
import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from unittest.mock import patch
import json
import time

import pynegative
from pynegative import core # Import core for sidecar functions

class TestApplyToneMap:
    """Tests for the apply_tone_map function"""

    def test_no_adjustments(self):
        """Test with default parameters (no adjustments)"""
        # Create a simple 2x2 RGB image
        img = np.array([
            [[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]],
            [[0.3, 0.3, 0.3], [0.9, 0.9, 0.9]]
        ], dtype=np.float32)

        result, stats = pynegative.apply_tone_map(img)

        # Result should be equal to input (within float precision)
        np.testing.assert_array_almost_equal(result, img)
        assert stats["mean"] == pytest.approx(0.6)

    def test_exposure_adjustment(self):
        """Test exposure adjustment (+1 stop)"""
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        # +1 stop should double values
        result, _ = pynegative.apply_tone_map(img, exposure=1.0)
        np.testing.assert_array_almost_equal(result, np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32))

    def test_blacks_whites_adjustment(self):
        """Test blacks and whites level adjustments"""
        img = np.array([
            [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [0.7, 0.7, 0.7]]
        ], dtype=np.float32)

        # Pull blacks to 0.1, whites to 0.9
        result, _ = pynegative.apply_tone_map(img, blacks=0.1, whites=0.9)

        # 0.5 -> (0.5-0.1)/0.8 = 0.4/0.8 = 0.5 (Mid stays mid)
        # 0.0 -> (0.0-0.1)/0.8 = -0.125 -> clipped to 0
        # 1.0 -> (1.0-0.1)/0.8 = 0.9/0.8 = 1.125 -> clipped to 1
        assert result[0,0,0] == pytest.approx(0.5)
        assert result[0,1,0] == 0.0
        assert result[1,0,0] == 1.0

    def test_shadows_highlights(self):
        """Test shadows and highlights tone adjustments"""
        img = np.array([[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]], dtype=np.float32)
        # Increase shadows, decrease highlights
        result, _ = pynegative.apply_tone_map(img, shadows=0.2, highlights=-0.2)

        # Shadows (0.2) should be boosted
        assert result[0,0,0] > 0.2
        # Highlights (0.8) should be reduced
        assert result[0,1,0] < 0.8

    def test_clipping_statistics(self):
        """Test that clipping statistics are calculated correctly"""
        img = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        _, stats = pynegative.apply_tone_map(img)
        assert stats["pct_highlights_clipped"] > 0.0
        assert stats["pct_shadows_clipped"] > 0.0

    def test_clipping(self):
        """Test that values are clipped to [0, 1]"""
        img = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        result, stats = pynegative.apply_tone_map(img)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert stats["pct_highlights_clipped"] > 0
        assert stats["pct_shadows_clipped"] > 0

    def test_saturation_adjustment(self):
        """Test saturation adjustment"""
        # Partially saturated color [0.5, 0.2, 0.2]
        img = np.array([[[0.5, 0.2, 0.2]]], dtype=np.float32)

        # Desaturate to 0 (should become grayscale)
        result, _ = pynegative.apply_tone_map(img, saturation=0.0)

        # Luminance for [0.5, 0.2, 0.2] is 0.5*0.2126 + 0.2*0.7152 + 0.2*0.0722 = 0.26378
        expected_gray = 0.26378
        np.testing.assert_array_almost_equal(result, np.array([[[expected_gray, expected_gray, expected_gray]]], dtype=np.float32))

        # Oversaturate
        result, _ = pynegative.apply_tone_map(img, saturation=2.0)
        # Manual check: lum + (img-lum)*2 = 0.26378 + (img-0.26378)*2
        # R: 0.26378 + (0.5-0.26378)*2 = 0.26378 + 0.23622*2 = 0.73622
        expected_r = 0.73622
        assert result[0,0,0] == pytest.approx(expected_r)

    def test_edge_case_zero_division_protection(self):
        """Test that the function handles edge cases like blacks == whites"""
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        # This should not raise a division by zero error
        result, _ = pynegative.apply_tone_map(img, blacks=0.5, whites=0.5)
        assert np.all(np.isfinite(result))


class TestCalculateAutoExposure:
    """Tests for auto-exposure calculation"""

    def test_basic_calculation(self):
        """Test that it returns expected keys"""
        img = np.random.rand(10, 10, 3).astype(np.float32)
        settings = pynegative.calculate_auto_exposure(img)
        assert "exposure" in settings
        assert "blacks" in settings
        assert "whites" in settings
        assert "saturation" in settings

    def test_normal_image(self):
        # Middle gray image
        img = np.full((10, 10, 3), 0.18, dtype=np.float32)
        settings = pynegative.calculate_auto_exposure(img)

        assert "exposure" in settings
        assert "blacks" in settings
        assert "whites" in settings
        assert settings["exposure"] > 0 # Should boost underexposed RAW

    def test_bright_image(self):
        # Very bright image
        img = np.full((10, 10, 3), 0.8, dtype=np.float32)
        settings = pynegative.calculate_auto_exposure(img)

        # Should have less boost than normal
        normal_img = np.full((10, 10, 3), 0.18, dtype=np.float32)
        normal_settings = pynegative.calculate_auto_exposure(normal_img)
        assert settings["exposure"] < normal_settings["exposure"]


class TestSharpening:
    """Tests for image sharpening"""

    def test_basic_sharpening(self):
        """Test basic sharpening with typical parameters"""
        # Create a simple PIL image
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))

        result = pynegative.sharpen_image(pil_img, radius=2.0, percent=100)

        # Should return a PIL Image
        assert isinstance(result, Image.Image)
        assert result.size == pil_img.size
        assert result.mode == pil_img.mode

    def test_sharpening_with_floats(self):
        """Regression test for TypeError when floats are passed to sharpen_image"""
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))

        # UI passes these as floats from division
        try:
            pynegative.sharpen_image(pil_img, radius=2.5, percent=150.0)
        except TypeError as e:
            pytest.fail(f"sharpen_image failed with floats: {e}")


class TestSaveImage:
    """Tests for the save_image function"""

    def test_save_jpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pil_img = Image.new('RGB', (10, 10), color=(255, 0, 0))
            output_path = tmpdir / "test.jpg"

            pynegative.save_image(pil_img, output_path)
            assert output_path.exists()

    def test_save_heif_not_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pil_img = Image.new('RGB', (10, 10), color=(255, 0, 0))
            output_path = tmpdir / "test.heic"

            # Mock HEIF_SUPPORTED to False
            with patch.object(pynegative.core, 'HEIF_SUPPORTED', False):
                with pytest.raises(RuntimeError, match="HEIF requested but pillow-heif not installed"):
                    pynegative.save_image(pil_img, output_path)

# New and updated tests for sidecar logic
@pytest.fixture
def temp_raw_path():
    """Provides a temporary RAW file path and ensures cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = Path(tmpdir) / "test_image.cr2"
        raw_file.touch() # Create a dummy raw file
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
        expected_path = temp_raw_path.parent / core.SIDECAR_DIR / f"{temp_raw_path.name}.json"
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
            "de_noise": 2
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
        with open(sidecar_path, 'w') as f:
            f.write("this is not valid json {") # Corrupted JSON

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
            "saturation": 1.05
        }
        
        # Manually create the sidecar data to control its content
        sidecar_data = {
            "version": "1.0",
            "last_modified": time.time(),
            "raw_path": str(temp_raw_path),
            "settings": old_settings_subset
        }
        sidecar_path = core.get_sidecar_path(temp_raw_path)
        temp_sidecar_dir.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, 'w') as f:
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
        new_raw.touch() # Create dummy new raw file
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
