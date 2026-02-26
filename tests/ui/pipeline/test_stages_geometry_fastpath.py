from unittest.mock import patch

import numpy as np

from pynegative.ui.pipeline import stages
from pynegative.ui.pipeline.stages import get_fused_geometry


def test_get_fused_geometry_reuses_maps_for_identity_affine():
    stages._FUSED_GEOMETRY_CACHE.clear()

    settings = {
        "lens_enabled": True,
        "lens_ca": 0.0,
        "lens_autocrop": False,
        "lens_distortion": 0.0,
    }

    mx = np.zeros((8, 10), dtype=np.float32)
    my = np.zeros((8, 10), dtype=np.float32)

    with (
        patch(
            "pynegative.processing.lens.get_lens_distortion_maps",
            return_value=(mx, my, 1.0),
        ) as mock_maps,
        patch(
            "pynegative.ui.pipeline.stages.GeometryResolver.get_fused_maps"
        ) as mock_fused,
    ):
        fused_maps, out_w, out_h, zoom_factor = get_fused_geometry(
            settings=settings,
            lens_info=None,
            w_src=10,
            h_src=8,
            rotate_val=0.0,
            crop_val=None,
            flip_h=False,
            flip_v=False,
            ts_roi=1.0,
            roi_offset=(0, 0),
            full_size=(10, 8),
        )

    mock_maps.assert_called_once()
    mock_fused.assert_not_called()
    assert len(fused_maps) == 1
    assert fused_maps[0][0] is mx
    assert fused_maps[0][1] is my
    assert out_w == 10
    assert out_h == 8
    assert zoom_factor == 1.0


def test_get_fused_geometry_uses_cache_for_rotated_case():
    stages._FUSED_GEOMETRY_CACHE.clear()

    settings = {
        "lens_enabled": True,
        "lens_ca": 0.0,
        "lens_autocrop": False,
        "lens_distortion": 0.0,
    }

    mx = np.zeros((8, 10), dtype=np.float32)
    my = np.zeros((8, 10), dtype=np.float32)

    with (
        patch(
            "pynegative.processing.lens.get_lens_distortion_maps",
            return_value=(mx, my, 1.0),
        ) as mock_maps,
        patch(
            "pynegative.ui.pipeline.stages.GeometryResolver.get_fused_maps",
            return_value=(mx, my),
        ) as mock_fused,
    ):
        get_fused_geometry(
            settings=settings,
            lens_info=None,
            w_src=10,
            h_src=8,
            rotate_val=5.0,
            crop_val=None,
            flip_h=False,
            flip_v=False,
            ts_roi=1.0,
            roi_offset=(0, 0),
            full_size=(10, 8),
        )
        get_fused_geometry(
            settings=settings,
            lens_info=None,
            w_src=10,
            h_src=8,
            rotate_val=5.0,
            crop_val=None,
            flip_h=False,
            flip_v=False,
            ts_roi=1.0,
            roi_offset=(0, 0),
            full_size=(10, 8),
        )

    assert mock_maps.call_count == 1
    assert mock_fused.call_count == 1
