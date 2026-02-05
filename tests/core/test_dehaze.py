#!/usr/bin/env python3
import pytest
import numpy as np
from PIL import Image
import pynegative


class TestDehaze:
    def test_no_dehaze(self):
        """Test that strength=0 returns original image"""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        result, _ = pynegative.de_haze_image(img, 0)
        np.testing.assert_array_equal(result, img)

    def test_dehaze_output_range(self):
        """Test that dehaze output is clipped to [0, 1]"""
        # Create a hazy-like image (low contrast, high offset)
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        img += np.random.normal(0, 0.1, (100, 100, 3))
        img = np.clip(img, 0, 1)

        result, _ = pynegative.de_haze_image(img, 10.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert result.shape == img.shape

        # Test with PIL input
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        result_pil, _ = pynegative.de_haze_image(pil_img, 10.0)
        assert isinstance(result_pil, Image.Image)
        assert result_pil.size == pil_img.size
