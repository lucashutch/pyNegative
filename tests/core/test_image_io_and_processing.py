#!/usr/bin/env python3
"""Unit tests for image I/O and processing functions in pynegative.core"""

import pytest
from PIL import Image
import tempfile
from pathlib import Path
from unittest.mock import patch

import pynegative


class TestSharpening:
    """Tests for image sharpening"""

    def test_basic_sharpening(self):
        """Test basic sharpening with typical parameters"""
        # Create a simple PIL image
        pil_img = Image.new("RGB", (10, 10), color=(128, 128, 128))

        result = pynegative.sharpen_image(pil_img, radius=2.0, percent=100)

        # Should return a PIL Image
        assert isinstance(result, Image.Image)
        assert result.size == pil_img.size
        assert result.mode == pil_img.mode

    def test_sharpening_with_floats(self):
        """Regression test for TypeError when floats are passed to sharpen_image"""
        pil_img = Image.new("RGB", (10, 10), color=(128, 128, 128))

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
            pil_img = Image.new("RGB", (10, 10), color=(255, 0, 0))
            output_path = tmpdir / "test.jpg"

            pynegative.save_image(pil_img, output_path)
            assert output_path.exists()

    def test_save_heif_not_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pil_img = Image.new("RGB", (10, 10), color=(255, 0, 0))
            output_path = tmpdir / "test.heic"

            # Mock HEIF_SUPPORTED to False
            with patch.object(pynegative.core, "HEIF_SUPPORTED", False):
                with pytest.raises(
                    RuntimeError, match="HEIF requested but pillow-heif not installed"
                ):
                    pynegative.save_image(pil_img, output_path)
