#!/usr/bin/env python3
"""Unit tests for pynegative.py"""
import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from unittest.mock import patch

import pynegative

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

class TestSidecars:
    """Tests for sidecar file logic"""

    def test_sidecar_path(self):
        raw_path = Path("/tmp/test.dng")
        expected = Path("/tmp/.pyNegative/test.dng.json")
        assert pynegative.core.get_sidecar_path(raw_path) == expected

    def test_save_load_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "test.dng"
            settings = {"exposure": 1.5, "blacks": 0.05}

            pynegative.save_sidecar(raw_path, settings)

            # Check file exists
            sidecar_path = pynegative.core.get_sidecar_path(raw_path)
            assert sidecar_path.exists()

            # Load and verify
            loaded = pynegative.load_sidecar(raw_path)
            assert loaded == settings

    def test_rename_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            old_raw = tmpdir / "old.dng"
            new_raw = tmpdir / "new.dng"
            settings = {"exposure": 1.5}

            pynegative.save_sidecar(old_raw, settings)
            old_sidecar = pynegative.core.get_sidecar_path(old_raw)
            assert old_sidecar.exists()

            pynegative.core.rename_sidecar(old_raw, new_raw)

            new_sidecar = pynegative.core.get_sidecar_path(new_raw)
            assert new_sidecar.exists()
            assert not old_sidecar.exists()

            # Verify data survived
            loaded = pynegative.load_sidecar(new_raw)
            assert loaded == settings
