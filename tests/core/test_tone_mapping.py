#!/usr/bin/env python3
"""Unit tests for tone mapping and auto-exposure functions in pynegative.core"""

import numpy as np
import pytest

import pynegative
from pynegative.processing.constants import LUMA_B, LUMA_G, LUMA_R


class TestApplyToneMap:
    """Tests for the apply_tone_map function"""

    def test_no_adjustments(self):
        """Test with default parameters (no adjustments)"""
        img = np.array(
            [[[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]], [[0.3, 0.3, 0.3], [0.9, 0.9, 0.9]]],
            dtype=np.float32,
        )

        result, stats = pynegative.apply_tone_map(img, apply_gamma=False)

        np.testing.assert_array_almost_equal(result, img)
        assert stats["mean"] == pytest.approx(0.6)

    def test_exposure_adjustment(self):
        """Test exposure adjustment (+1 stop)"""
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        preprocessed = pynegative.apply_preprocess(img, exposure=1.0)
        result, _ = pynegative.apply_tone_map(preprocessed, apply_gamma=False)
        np.testing.assert_array_almost_equal(
            result, np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        )

    def test_preprocess_vignette(self):
        """Test vignette application in preprocess"""
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        result = pynegative.apply_preprocess(
            img,
            vignette_k1=0.5,
            vignette_cx=5.0,
            vignette_cy=5.0,
            full_width=10.0,
            full_height=10.0,
        )
        assert result[5, 5, 0] < result[0, 0, 0]

    def test_preprocess_white_balance(self):
        """Test white balance in preprocess"""
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = pynegative.apply_preprocess(img, temperature=0.5)
        assert result[0, 0, 0] != result[0, 0, 2]

    def test_blacks_whites_adjustment(self):
        """Test blacks and whites level adjustments"""
        img = np.array(
            [[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [0.7, 0.7, 0.7]]],
            dtype=np.float32,
        )

        # New semantics: blacks=-0.5 -> b_mapped = -(-0.5)*0.2 = 0.1
        #                whites=-0.2 -> w_mapped = 1.0 + (-0.2)*0.5 = 0.9
        # So effective range is [0.1, 0.9], denominator = 0.8
        result, _ = pynegative.apply_tone_map(
            img, blacks=-0.5, whites=-0.2, apply_gamma=False
        )

        # 0.5 -> (0.5-0.1)/0.8 = 0.4/0.8 = 0.5 (Mid stays mid)
        # 0.0 -> (0.0-0.1)/0.8 = -0.125 -> clipped to 0
        # 1.0 -> (1.0-0.1)/0.8 = 0.9/0.8 = 1.125 -> clipped to 1
        assert result[0, 0, 0] == pytest.approx(0.5)
        assert result[0, 1, 0] == 0.0
        assert result[1, 0, 0] == 1.0

    def test_shadows_highlights(self):
        """Test shadows and highlights tone adjustments"""
        img = np.array([[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]], dtype=np.float32)
        # Increase shadows, decrease highlights
        result, _ = pynegative.apply_tone_map(
            img, shadows=0.2, highlights=-0.2, apply_gamma=False
        )

        # Shadows (0.2) should be boosted
        assert result[0, 0, 0] > 0.2
        # Highlights (0.8) should be reduced
        assert result[0, 1, 0] < 0.8

    def test_clipping_statistics(self):
        """Test that clipping statistics are calculated correctly"""
        img = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        _, stats = pynegative.apply_tone_map(img, apply_gamma=False)
        assert stats["pct_highlights_clipped"] > 0.0
        assert stats["pct_shadows_clipped"] > 0.0

    def test_clipping(self):
        """Test that values are clipped to [0, 1]"""
        img = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        result, stats = pynegative.apply_tone_map(img, apply_gamma=False)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert stats["pct_highlights_clipped"] > 0
        assert stats["pct_shadows_clipped"] > 0

    def test_saturation_adjustment(self):
        """Test saturation adjustment"""
        # Partially saturated color [0.5, 0.2, 0.2]
        img = np.array([[[0.5, 0.2, 0.2]]], dtype=np.float32)

        # Desaturate to -1.0 (s_mult = 1.0 + (-1.0) = 0.0, should become grayscale)
        result, _ = pynegative.apply_tone_map(img, saturation=-1.0, apply_gamma=False)

        # Luminance for [0.5, 0.2, 0.2] is 0.5*LUMA_R + 0.2*LUMA_G + 0.2*LUMA_B
        expected_gray = 0.5 * LUMA_R + 0.2 * LUMA_G + 0.2 * LUMA_B
        np.testing.assert_array_almost_equal(
            result,
            np.array(
                [[[expected_gray, expected_gray, expected_gray]]], dtype=np.float32
            ),
        )

        # Oversaturate: saturation=1.0 (s_mult = 1.0 + 1.0 = 2.0)
        result, _ = pynegative.apply_tone_map(img, saturation=1.0, apply_gamma=False)
        # Manual check: lum + (img-lum)*2
        expected_r = expected_gray + (0.5 - expected_gray) * 2.0
        assert result[0, 0, 0] == pytest.approx(expected_r)

    def test_edge_case_zero_division_protection(self):
        """Test that the function handles edge cases without division by zero"""
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        # Extreme values that make w_mapped very close to b_mapped
        # This should not raise a division by zero error
        result, _ = pynegative.apply_tone_map(img, blacks=0.0, whites=-1.0)
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
        assert settings["exposure"] > 0  # Should boost underexposed RAW

    def test_bright_image(self):
        # Very bright image
        img = np.full((10, 10, 3), 0.8, dtype=np.float32)
        settings = pynegative.calculate_auto_exposure(img)

        # Should have less boost than normal
        normal_img = np.full((10, 10, 3), 0.18, dtype=np.float32)
        normal_settings = pynegative.calculate_auto_exposure(normal_img)
        assert settings["exposure"] < normal_settings["exposure"]
