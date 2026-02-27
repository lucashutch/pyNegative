import pynegative  # noqa: F401
import numpy as np
from pynegative.processing.effects import (
    sharpen_image,
    de_noise_image,
    estimate_atmospheric_light,
    de_haze_image,
)


def test_sharpen_image_none():
    assert sharpen_image(None, 1, 1) is None


def test_sharpen_image_zero():
    img = np.ones((10, 10, 3), dtype=np.float32)
    out = sharpen_image(img, radius=0, percent=0)
    assert np.array_equal(img, out)


def test_sharpen_image_valid():
    img = np.ones((10, 10, 3), dtype=np.float32)
    out = sharpen_image(img, radius=1.0, percent=0.5)
    assert out.shape == (10, 10, 3)


def test_de_noise_image_none():
    assert de_noise_image(None) is None


def test_de_noise_image_zero():
    img = np.ones((10, 10, 3), dtype=np.float32)
    out = de_noise_image(img, 0, 0)
    assert np.array_equal(img, out)


def test_de_noise_image_bilateral():
    img = np.random.rand(10, 10, 3).astype(np.float32)
    out = de_noise_image(img, 10, 10)
    assert out.shape == (10, 10, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_de_noise_image_luma_only():
    """Test bilateral denoise with luma-only processing."""
    img = np.random.rand(10, 10, 3).astype(np.float32)
    out = de_noise_image(img, luma_strength=10, chroma_strength=0)
    assert out.shape == (10, 10, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_de_noise_image_chroma_only():
    """Test bilateral denoise with chroma-only processing."""
    img = np.random.rand(10, 10, 3).astype(np.float32)
    out = de_noise_image(img, luma_strength=0, chroma_strength=10)
    assert out.shape == (10, 10, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_estimate_atmospheric_light_none():
    assert estimate_atmospheric_light(None) is None


def test_estimate_atmospheric_light_valid():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[5, 5] = [1.0, 1.0, 1.0]  # bright spot
    val = estimate_atmospheric_light(img)
    assert val is not None
    assert val.shape == (3,)


def test_de_haze_image_none():
    out, al = de_haze_image(None, 0.5)
    assert out is None
    assert al is None


def test_de_haze_image_zero():
    img = np.ones((10, 10, 3), dtype=np.float32)
    out, al = de_haze_image(img, 0.0)
    assert np.array_equal(img, out)


def test_de_haze_image_legacy_scale():
    img = np.random.rand(20, 20, 3).astype(np.float32)
    # uses 50.0 value which is > 1.0 so it will divide by 50
    out, al = de_haze_image(img, 50.0)
    assert out.shape == (20, 20, 3)
    assert al is not None


def test_de_haze_image_fixed_light():
    img = np.random.rand(20, 20, 3).astype(np.float32)
    fixed = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    out, al = de_haze_image(img, 0.5, fixed_atmospheric_light=fixed)
    assert out.shape == (20, 20, 3)
    assert np.array_equal(al, fixed)
