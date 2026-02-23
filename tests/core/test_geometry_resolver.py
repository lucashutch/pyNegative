import pynegative

import cv2
import numpy as np

from pynegative.processing.geometry import GeometryResolver, apply_geometry


def test_geometry_resolver_identity():
    resolver = GeometryResolver(100, 100)
    matrix = resolver.resolve(expand=False)
    expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(matrix, expected)


def test_geometry_resolver_flip():
    resolver = GeometryResolver(100, 100)

    # Flip H
    matrix = resolver.resolve(flip_h=True, expand=False)
    # Expected: x -> -x + 99 (for 100px width)
    expected = np.array([[-1, 0, 99], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(matrix, expected)

    # Flip V
    matrix = resolver.resolve(flip_v=True, expand=False)
    # Expected: y -> -y + 99
    expected = np.array([[1, 0, 0], [0, -1, 99]], dtype=np.float32)
    np.testing.assert_array_almost_equal(matrix, expected)


def test_geometry_resolver_vs_apply_geometry():
    """Verify that GeometryResolver matches apply_geometry (since apply_geometry now uses it)."""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[20:50, 30:60, 0] = 1.0  # Red square

    params = {
        "rotate": 15.0,
        "flip_h": True,
        "flip_v": False,
        "crop": (0.1, 0.1, 0.9, 0.9),
    }

    # New unified approach
    out_1 = apply_geometry(img.copy(), **params)

    # Manual approach using resolver
    resolver = GeometryResolver(100, 100)
    resolver.resolve(**params, expand=True)
    M = resolver.get_matrix_2x3()
    w, h = resolver.get_output_size()
    out_2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    np.testing.assert_array_almost_equal(out_1, out_2)


def test_geometry_resolver_crop():
    resolver = GeometryResolver(100, 100)
    # Crop (0.1, 0.2, 0.9, 0.8) on a 100x100 image
    # Means tx = 10, ty = 20
    matrix = resolver.resolve(crop=(0.1, 0.2, 0.9, 0.8), expand=False)
    # Expected translation: x -> x - 10, y -> y - 20
    expected = np.array([[1, 0, -10], [0, 1, -20]], dtype=np.float32)
    np.testing.assert_array_almost_equal(matrix, expected)

    w, h = resolver.get_output_size()
    assert w == 80  # (0.9 - 0.1) * 100
    assert h == 60  # (0.8 - 0.2) * 100


def test_geometry_resolver_inverse():
    resolver = GeometryResolver(100, 100)
    resolver.resolve(rotate=30, expand=True)

    M = resolver.get_matrix_2x3()
    M_inv = resolver.get_inverse_matrix()

    # Point at (10, 20) in source
    pt = np.array([10, 20, 1], dtype=np.float32)

    # Transform to destination
    dst = M @ pt
    dst_h = np.append(dst, 1.0)

    # Transform back to source
    src_back = M_inv @ dst_h

    np.testing.assert_array_almost_equal(pt[:2], src_back[:2], decimal=5)
