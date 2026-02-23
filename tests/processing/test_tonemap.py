import pynegative  # noqa: F401
import numpy as np
from pynegative.processing.tonemap import (
    apply_preprocess,
    apply_tone_map,
    calculate_auto_wb,
)


def test_apply_preprocess_none():
    assert apply_preprocess(None) is None


def test_apply_preprocess_invalid_args():
    img = np.ones((10, 10, 3), dtype=np.float32)
    out = apply_preprocess(img, temperature="invalid", tint="none")
    assert out.shape == (10, 10, 3)


def test_apply_tone_map_none():
    out, stats = apply_tone_map(None)
    assert out is None
    assert stats is None


def test_apply_tone_map_invalid_args():
    img = np.ones((10, 10, 3), dtype=np.float32)
    # The current code returns img, {} on invalid exception which we might not hit perfectly but we test safe_float wrapper
    out, stats = apply_tone_map(img, contrast="bad", calculate_stats=False)
    assert out.shape == (10, 10, 3)


def test_calculate_auto_wb():
    img = np.ones((10, 10, 3), dtype=np.float32)
    img[:, :, 0] = 0.5  # Red
    img[:, :, 1] = 0.8  # Green
    img[:, :, 2] = 0.2  # Blue
    res = calculate_auto_wb(img)
    assert "temperature" in res
    assert "tint" in res


def test_calculate_auto_wb_zeros():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    res = calculate_auto_wb(img)
    assert "temperature" in res
