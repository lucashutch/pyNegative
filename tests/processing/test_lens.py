# OpenCV must be loaded via pynegative before numpy to avoid import errors
import pynegative  # noqa: F401
import numpy as np

from pynegative.processing.lens import (
    LensMapCache,
    generate_ptlens_map,
    generate_poly3_map,
    remap_tca_distortion_kernel,
    get_distortion_maps,
    calculate_autocrop_scale,
    vignette_kernel,
    generate_tca_maps,
    get_tca_distortion_maps,
    get_lens_distortion_maps,
    apply_lens_correction,
)
from pynegative.utils.numba_lens import (
    _bilinear_sample,
    _get_distortion_rescale,
    _get_tca_rescale,
)


def test_lens_map_cache():
    cache = LensMapCache(max_entries=2)
    w, h, fw, fh = 10, 10, 10, 10
    cx, cy = 5.0, 5.0
    map_x, map_y = cache.get_maps(w, h, "poly3", {"k1": 0.1}, cx, cy, fw, fh, 1.0)
    assert map_x is not None
    assert map_y is not None
    assert map_x.shape == (h, w)

    map_x2, map_y2 = cache.get_maps(
        w, h, "ptlens", {"a": 0.01, "b": 0.02, "c": 0.03}, cx, cy, fw, fh, 1.0
    )
    assert map_x2 is not None


def test_generate_ptlens_map():
    map_x, map_y = generate_ptlens_map(10, 10, 0.01, 0.01, 0.01, 5.0, 5.0, 10, 10, 1.0)
    assert map_x.shape == (10, 10)
    assert map_y.shape == (10, 10)


def test_generate_poly3_map():
    map_x, map_y = generate_poly3_map(10, 10, 0.1, 5.0, 5.0, 10, 10, 1.0)
    assert map_x.shape == (10, 10)
    assert map_y.shape == (10, 10)


def test_bilinear_sample():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[5, 5] = [1.0, 2.0, 3.0]
    val = _bilinear_sample(img, 5.0, 5.0, 0, 10, 10)
    assert val == 1.0


def test_tca_rescale_and_distortion_rescale():
    assert _get_distortion_rescale(0.5, 1, np.array([0.1], dtype=np.float32)) > 1.0
    assert _get_tca_rescale(0.5, np.array([1.0, 0.1, 0.0], dtype=np.float32)) > 1.0


def test_remap_tca_distortion_kernel():
    img = np.ones((10, 10, 3), dtype=np.float32)
    out = np.zeros_like(img)
    dist_p = np.array([0.1], dtype=np.float32)
    t_red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t_blue = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    remap_tca_distortion_kernel(
        img, out, 1, dist_p, t_red, t_blue, 5.0, 5.0, 10, 10, 1.0
    )
    assert out.shape == (10, 10, 3)


def test_get_distortion_maps():
    mx, my = get_distortion_maps(10, 10, -0.1)
    assert mx.shape == (10, 10)


def test_calculate_autocrop_scale():
    scale = calculate_autocrop_scale("poly3", {"k1": 0.1}, 100, 100)
    assert scale >= 1.0
    scale_pt = calculate_autocrop_scale(
        "ptlens", {"a": -0.01, "b": -0.01, "c": -0.01}, 100, 100
    )
    assert scale_pt >= 1.0


def test_vignette_kernel():
    img = np.ones((10, 10, 3), dtype=np.float32)
    vignette_kernel(img, -0.1, -0.1, -0.1, 5.0, 5.0, 10, 10)
    assert img.shape == (10, 10, 3)
    assert img[5, 5, 0] <= 1.0


def test_generate_tca_maps():
    maps = generate_tca_maps(
        10,
        10,
        1,
        np.array([0.1], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        5.0,
        5.0,
        10,
        10,
        1.0,
    )
    assert len(maps) == 6


def test_get_tca_distortion_maps():
    settings = {"lens_distortion": 0.2, "lens_ca": 1.0, "lens_autocrop": True}
    lens_info = {
        "tca": {"vr0": 1.001, "vb0": 0.999},
        "distortion": {"model": "poly3", "k1": -0.1},
    }
    maps = get_tca_distortion_maps(10, 10, settings, lens_info, full_size=(10, 10))
    assert len(maps) == 7
    assert maps[-1] > 1.0


def test_get_lens_distortion_maps():
    settings = {"lens_distortion": 0.2, "lens_autocrop": True}
    lens_info = {"distortion": {"model": "ptlens", "c": -0.1}}
    mx, my, zoom = get_lens_distortion_maps(
        10, 10, settings, lens_info, roi_offset=(0, 0), full_size=(10, 10)
    )
    assert mx is not None


def test_apply_lens_correction():
    img = np.ones((10, 10, 3), dtype=np.float32)
    settings = {
        "lens_distortion": -0.1,
        "lens_vignette": -0.1,
        "lens_ca": 1.0,
        "lens_autocrop": True,
    }
    lens_info = {
        "distortion": {"model": "poly3", "k1": -0.1},
        "vignetting": {"model": "pa", "k1": -0.1},
        "tca": {"vr0": 1.001, "vb0": 0.999},
    }
    out = apply_lens_correction(img, settings, lens_info)
    assert out.shape == (10, 10, 3)

    lens_info_no_tca = {
        "distortion": {"model": "poly3", "k1": -0.1},
    }
    out2 = apply_lens_correction(img, settings, lens_info_no_tca)
    assert out2.shape == (10, 10, 3)

    out3 = apply_lens_correction(img, settings, lens_info, skip_vignette=True)
    assert out3.shape == (10, 10, 3)
