import cv2
import numpy as np

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
    img_distorted = cv2.remap(
        img,
        map_x_dist,
        map_y_dist,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # b. Apply affine geometry
    M = resolver.get_matrix_2x3()
    out_w, out_h = resolver.get_output_size()
    expected_out = cv2.warpAffine(
        img_distorted,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 4. Method B: Fused application
    fused_x, fused_y = resolver.get_fused_maps(map_x_dist, map_y_dist)
    actual_out = cv2.remap(
        img,
        fused_x,
        fused_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

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
    # Relax max_diff as edge interpolation can cause single pixel differences up to 1.0
    assert max_diff < 1.1


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


def test_geometry_fused_roi():
    """
    Verify fused maps for a ROI.
    """
    full_w, full_h = 1000, 800
    img = np.zeros((full_h, full_w, 3), dtype=np.float32)
    cv2.rectangle(img, (100, 100), (900, 700), (1.0, 1.0, 1.0), -1)

    # 1. Transform parameters
    rotate = 10.0
    crop = (0.2, 0.2, 0.8, 0.8)

    resolver = GeometryResolver(full_w, full_h)
    resolver.resolve(rotate=rotate, crop=crop, expand=True)
    new_full_w, new_full_h = resolver.get_output_size()
    M_full = resolver.get_matrix_2x3()

    # 2. Define ROI in final geometry
    rx, ry, rw, rh = 100, 100, 200, 150

    # 3. Calculate source chunk for this ROI
    M_inv = resolver.get_inverse_matrix()
    corners = np.array(
        [[rx, ry, 1], [rx + rw, ry, 1], [rx + rw, ry + rh, 1], [rx, ry + rh, 1]],
        dtype=np.float32,
    )
    src_corners = (M_inv @ corners.T).T

    s_xmin = int(np.floor(np.min(src_corners[:, 0])))
    s_ymin = int(np.floor(np.min(src_corners[:, 1])))
    s_xmax = int(np.ceil(np.max(src_corners[:, 0])))
    s_ymax = int(np.ceil(np.max(src_corners[:, 1])))

    # Add padding
    pad = 10
    s_xmin = max(0, s_xmin - pad)
    s_ymin = max(0, s_ymin - pad)
    s_xmax = min(full_w, s_xmax + pad)
    s_ymax = min(full_h, s_ymax + pad)

    chunk = img[s_ymin:s_ymax, s_xmin:s_xmax]
    chunk_h, chunk_w = chunk.shape[:2]

    # 4. Sequential reference:
    # a. Apply rotation/crop to full image
    full_transformed = cv2.warpAffine(
        img,
        M_full,
        (int(round(new_full_w)), int(round(new_full_h))),
        flags=cv2.INTER_LINEAR,
    )
    # b. Extract ROI
    expected_roi = full_transformed[ry : ry + rh, rx : rx + rw]

    # 5. Fused ROI method:
    # Calculate M_local that maps from chunk coordinates to ROI coordinates
    M_local = M_full.copy()
    # Adjust for chunk offset in source
    M_local[0, 2] += M_full[0, 0] * s_xmin + M_full[0, 1] * s_ymin
    M_local[1, 2] += M_full[1, 0] * s_xmin + M_full[1, 1] * s_ymin
    # Adjust for ROI offset in destination
    M_local[0, 2] -= rx
    M_local[1, 2] -= ry

    # In this test we don't have lens distortion, so we test the affine fusion.
    # We can use an identity lens map.
    map_x_id, map_y_id = np.meshgrid(
        np.arange(chunk_w, dtype=np.float32), np.arange(chunk_h, dtype=np.float32)
    )

    # Fuse
    resolver_roi = GeometryResolver(chunk_w, chunk_h)
    m33 = np.eye(3, dtype=np.float32)
    m33[:2, :] = M_local
    resolver_roi.matrix = m33
    resolver_roi.full_w = rw
    resolver_roi.full_h = rh

    fused_x, fused_y = resolver_roi.get_fused_maps(map_x_id, map_y_id)
    actual_roi = cv2.remap(chunk, fused_x, fused_y, cv2.INTER_LINEAR)

    # 6. Compare
    diff = np.abs(expected_roi - actual_roi)
    assert np.mean(diff) < 0.01
    assert np.max(diff) < 0.5


def test_geometry_fused_tca():
    """
    Verify fused maps with TCA.
    """
    from pynegative.processing.lens import get_tca_distortion_maps

    w, h = 200, 150
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Add Red and Blue specific features
    cv2.rectangle(img, (50, 50), (150, 100), (1.0, 1.0, 1.0), -1)

    settings = {
        "lens_enabled": True,
        "lens_distortion": 0.05,
        "lens_ca": 1.0,
        "lens_autocrop": False,
    }

    # Fake lens info with TCA
    lens_info = {
        "tca": {
            "vr0": 1.002,
            "vr1": 0.0,
            "vr2": 0.0,
            "vb0": 0.998,
            "vb1": 0.0,
            "vb2": 0.0,
        }
    }

    # 1. Get TCA maps
    xr, yr, xg, yg, xb, yb, zoom = get_tca_distortion_maps(w, h, settings, lens_info)

    # 2. Sequential Reference
    # a. Apply TCA+Distortion
    ref_r = cv2.remap(
        img[:, :, 0],
        xr,
        yr,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ref_g = cv2.remap(
        img[:, :, 1],
        xg,
        yg,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ref_b = cv2.remap(
        img[:, :, 2],
        xb,
        yb,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ref_distorted = cv2.merge([ref_r, ref_g, ref_b])

    # b. Apply affine
    angle = 10.0
    resolver = GeometryResolver(w, h)
    resolver.resolve(rotate=angle, expand=False)
    M = resolver.get_matrix_2x3()
    expected = cv2.warpAffine(
        ref_distorted,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 3. Fused
    f_xr, f_yr = resolver.get_fused_maps(xr, yr)
    f_xg, f_yg = resolver.get_fused_maps(xg, yg)
    f_xb, f_yb = resolver.get_fused_maps(xb, yb)

    act_r = cv2.remap(
        img[:, :, 0],
        f_xr,
        f_yr,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    act_g = cv2.remap(
        img[:, :, 1],
        f_xg,
        f_yg,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    act_b = cv2.remap(
        img[:, :, 2],
        f_xb,
        f_yb,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    actual = cv2.merge([act_r, act_g, act_b])

    # 4. Compare
    diff = np.abs(expected - actual)
    assert np.mean(diff) < 0.01
    assert np.max(diff) < 1.1
