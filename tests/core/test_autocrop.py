import pytest
import numpy as np
from pynegative.processing.lens import calculate_autocrop_scale, apply_lens_correction


def test_autocrop_scale_calculation():
    # Pincushion-like distortion (stretches edges out, causes black borders)
    # k1 > 0 in poly3 model
    fw, fh = 6000, 4000
    params = {"k1": 0.1}

    scale = calculate_autocrop_scale("poly3", params, fw, fh)

    # Scale should be > 1.0 to zoom in and hide black borders
    assert scale > 1.0
    # Specifically for k1=0.1 at corners (rn=1.0), rescale = 1.1
    assert pytest.approx(scale, rel=1e-3) == 1.1


def test_autocrop_no_distortion():
    fw, fh = 6000, 4000
    params = {"k1": 0.0}
    scale = calculate_autocrop_scale("poly3", params, fw, fh)
    assert scale == 1.0


def test_autocrop_barrel_distortion():
    # Barrel distortion (squeezes edges in, no black borders)
    # k1 < 0 in poly3 model
    fw, fh = 6000, 4000
    params = {"k1": -0.1}
    scale = calculate_autocrop_scale("poly3", params, fw, fh)

    # Correction expands it back out, sampling from center.
    # No black borders, so scale should be 1.0.
    assert scale == 1.0


def test_apply_correction_with_autocrop():
    # Create small test image
    size = 100
    img = np.ones((size, size, 3), dtype=np.float32)
    # Put a distinct border at the very edge
    img[0, :, :] = 0.5
    img[-1, :, :] = 0.5
    img[:, 0, :] = 0.5
    img[:, -1, :] = 0.5

    settings = {
        "lens_distortion": 0.2,  # Strong pincushion
        "lens_autocrop": True,
    }

    corrected = apply_lens_correction(img, settings)

    # Without autocrop, the corners of 'corrected' would be black (0.0)
    # With autocrop, the corners should be filled with content from the inner part of the image.
    assert corrected[0, 0, 0] > 0.0
    assert corrected[size - 1, size - 1, 0] > 0.0

    # Check that it's different from the non-autocropped version
    settings_no_crop = settings.copy()
    settings_no_crop["lens_autocrop"] = False
    corrected_no_crop = apply_lens_correction(img, settings_no_crop)

    # Corners should be black in no-crop version
    assert corrected_no_crop[0, 0, 0] == 0.0
