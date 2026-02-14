import numpy as np
import cv2
from pynegative.processing.geometry import GeometryResolver
from pynegative.processing.lens import generate_poly3_map


def test_geometry_fused_maps():
    """
    Verify that fused maps (distortion + affine) produce the same result
    as sequential application (distortion then affine).
    """
    w, h = 200, 150
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Add some pattern to the image
    cv2.rectangle(img, (20, 20), (180, 130), (1.0, 0.5, 0.2), 2)
    cv2.line(img, (0, 0), (w, h), (0, 1, 0), 1)

    # 1. Generate a lens distortion map (pincushion)
    k1 = 0.1
    cx, cy = w / 2.0, h / 2.0
    map_x_dist, map_y_dist = generate_poly3_map(w, h, k1, cx, cy, w, h)

    # 2. Define an affine transform (rotate + crop)
    angle = 15.0
    crop = (0.1, 0.1, 0.9, 0.9)
    resolver = GeometryResolver(w, h)
    resolver.resolve(rotate=angle, crop=crop, expand=True)

    # 3. Method A: Sequential application
    # a. Apply distortion
    img_distorted = cv2.remap(img, map_x_dist, map_y_dist, cv2.INTER_LINEAR)
    # b. Apply affine geometry
    M = resolver.get_matrix_2x3()
    out_w, out_h = resolver.get_output_size()
    expected_out = cv2.warpAffine(
        img_distorted, M, (out_w, out_h), flags=cv2.INTER_LINEAR
    )

    # 4. Method B: Fused application
    fused_x, fused_y = resolver.get_fused_maps(map_x_dist, map_y_dist)
    actual_out = cv2.remap(img, fused_x, fused_y, cv2.INTER_LINEAR)

    # 5. Compare
    # Note: Minor differences are expected due to interpolation order and precision
    # INTER_LINEAR in remap vs warpAffine might differ slightly in implementation details
    # but the overall transformation should be the same.
    # We use a reasonably loose tolerance for float image comparison.

    # Calculate difference
    diff = np.abs(expected_out - actual_out)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    print(f"Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")

    # Verification: most pixels should be very close.
    # We allow some error due to double interpolation in Method A vs single in Method B.
    assert mean_diff < 0.05
    assert max_diff < 0.8


def test_geometry_fused_identity():
    """Identity affine transform should preserve the lens map."""
    w, h = 100, 100
    map_x = np.random.rand(h, w).astype(np.float32) * w
    map_y = np.random.rand(h, w).astype(np.float32) * h

    resolver = GeometryResolver(w, h)
    resolver.resolve(expand=False)  # Identity

    fused_x, fused_y = resolver.get_fused_maps(map_x, map_y)

    np.testing.assert_array_almost_equal(map_x, fused_x)
    np.testing.assert_array_almost_equal(map_y, fused_y)
