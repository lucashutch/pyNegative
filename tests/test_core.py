#!/usr/bin/env python3
"""Unit tests for pyrawroom.py"""
import pytest
import numpy as np
from PIL import Image, ImageFilter
import tempfile
import os
from unittest.mock import patch, MagicMock

import pyrawroom


class TestApplyToneMap:
    """Tests for the apply_tone_map function"""

    def test_no_adjustments(self):
        """Test with default parameters (no adjustments)"""
        # Create a simple 2x2 RGB image
        img = np.array([
            [[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]],
            [[0.3, 0.3, 0.3], [0.9, 0.9, 0.9]]
        ], dtype=np.float32)

        result, stats = pyrawroom.apply_tone_map(img)

        # With no adjustments, output should be the same (clipped to [0,1])
        np.testing.assert_array_almost_equal(result, img)
        assert stats["pct_shadows_clipped"] == 0.0
        assert stats["pct_highlights_clipped"] == 0.0
        assert "mean" in stats

    def test_exposure_adjustment(self):
        """Test exposure adjustment"""
        img = np.array([
            [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        ], dtype=np.float32)

        # Increase exposure by +1 stop (should double values)
        result, stats = pyrawroom.apply_tone_map(img, exposure=1.0)

        expected = img * 2.0
        expected = np.clip(expected, 0.0, 1.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_blacks_whites_adjustment(self):
        """Test blacks and whites level adjustments"""
        img = np.array([
            [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [0.25, 0.25, 0.25]]
        ], dtype=np.float32)

        # Apply blacks and whites
        result, stats = pyrawroom.apply_tone_map(img, blacks=0.2, whites=0.8)

        # Values should be remapped: (img - blacks) / (whites - blacks)
        expected = (img - 0.2) / (0.8 - 0.2)
        expected = np.clip(expected, 0.0, 1.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_shadows_highlights(self):
        """Test shadows and highlights tone adjustments"""
        img = np.array([
            [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],
        ], dtype=np.float32)

        # Apply shadows and highlights adjustments
        result, stats = pyrawroom.apply_tone_map(img, shadows=0.1, highlights=0.1)

        # Result should be modified (shadows lift darks, highlights recover brights)
        # Just verify the shape is correct and values are in range
        assert result.shape == img.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_clipping_statistics(self):
        """Test that clipping statistics are calculated correctly"""
        img = np.array([
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        ], dtype=np.float32)

        # Apply extreme exposure to cause clipping
        result, stats = pyrawroom.apply_tone_map(img, exposure=2.0)

        # Should have some highlights clipping
        assert stats["pct_highlights_clipped"] > 0.0
        assert isinstance(stats["mean"], (float, np.floating))

    def test_edge_case_zero_division_protection(self):
        """Test that the function handles edge cases like blacks == whites"""
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

        # This should not raise a division by zero error
        result, stats = pyrawroom.apply_tone_map(img, blacks=0.5, whites=0.5)

        assert result.shape == img.shape
        assert np.all(np.isfinite(result))


class TestSharpenImage:
    """Tests for the sharpen_image function"""

    def test_basic_sharpening(self):
        """Test basic sharpening with typical parameters"""
        # Create a simple PIL image
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))

        result = pyrawroom.sharpen_image(pil_img, radius=2.0, percent=100)

        # Should return a PIL Image
        assert isinstance(result, Image.Image)
        assert result.size == pil_img.size
        assert result.mode == pil_img.mode


class TestSaveImage:
    """Tests for the save_image function"""

    def test_jpeg_format(self):
        """Test JPEG format saving"""
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.jpg")
            pyrawroom.save_image(pil_img, output_path, quality=95)

            # Verify file was created
            assert os.path.exists(output_path)

            # Verify it can be opened as an image
            saved_img = Image.open(output_path)
            assert saved_img.size == pil_img.size

    def test_jpeg_format_uppercase(self):
        """Test JPEG format with uppercase extension"""
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.JPEG")
            pyrawroom.save_image(pil_img, output_path, quality=90)

            assert os.path.exists(output_path)

    def test_unsupported_format(self):
        """Test error handling for unsupported formats"""
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.unsupported")

            with pytest.raises(ValueError, match="Unsupported format"):
                pyrawroom.save_image(pil_img, output_path)

    def test_heif_format_without_support(self):
        """Test HEIF format error when pillow-heif not available"""
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.heif")

            # Mock HEIF_SUPPORTED as False in the core module
            with patch.object(pyrawroom.core, 'HEIF_SUPPORTED', False):
                with pytest.raises(RuntimeError, match="HEIF requested but pillow-heif not installed"):
                    pyrawroom.save_image(pil_img, output_path)
