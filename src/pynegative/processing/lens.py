import cv2
import numpy as np
import logging
from numba import njit, prange

logger = logging.getLogger(__name__)


@njit(parallel=True)
def generate_ptlens_map(w, h, a, b, c, cx, cy, full_w, full_h, zoom=1.0):
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    max_r = np.sqrt((full_w / 2.0) ** 2 + (full_h / 2.0) ** 2)
    inv_max_r = 1.0 / max_r
    d = 1.0 - a - b - c

    inv_zoom = 1.0 / zoom

    for y in prange(h):
        for x in range(w):
            # Coordinates relative to full image center
            dx = (x - cx) * inv_zoom
            dy = (y - cy) * inv_zoom

            r = np.sqrt(dx * dx + dy * dy)
            rn = r * inv_max_r  # Normalized radius

            if rn == 0:
                map_x[y, x] = cx
                map_y[y, x] = cy
                continue

            # PTLens model: r_src = r_dist * (a*rn^3 + b*rn^2 + c*rn + d)
            rescale = a * rn**3 + b * rn**2 + c * rn + d

            map_x[y, x] = cx + dx * rescale
            map_y[y, x] = cy + dy * rescale

    return map_x, map_y


@njit(parallel=True)
def generate_poly3_map(w, h, k1, cx, cy, full_w, full_h, zoom=1.0):
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    max_r2 = (full_w / 2.0) ** 2 + (full_h / 2.0) ** 2
    inv_max_r2 = 1.0 / max_r2

    inv_zoom = 1.0 / zoom

    for y in prange(h):
        for x in range(w):
            dx = (x - cx) * inv_zoom
            dy = (y - cy) * inv_zoom
            r2 = dx * dx + dy * dy
            rn2 = r2 * inv_max_r2

            # Poly3 model: r_src = r_dist * (1 + k1 * rn^2)
            rescale = 1.0 + k1 * rn2

            map_x[y, x] = cx + dx * rescale
            map_y[y, x] = cy + dy * rescale

    return map_x, map_y


def get_distortion_maps(w, h, k1, center_x=None, center_y=None):
    """Generate OpenCV remapping maps for simple radial distortion (OpenCV model)."""
    if center_x is None:
        center_x = w / 2.0
    if center_y is None:
        center_y = h / 2.0

    f = max(w, h)
    K = np.array([[f, 0, center_x], [0, f, center_y], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 0)
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1
    )
    return map_x, map_y


def calculate_autocrop_scale(model, params, fw, fh):
    """
    Calculate the magnification scale (>1.0) required to remove black borders.
    """
    max_r_orig = np.sqrt((fw / 2.0) ** 2 + (fh / 2.0) ** 2)

    def get_rescale(rn):
        """Returns r_src / r_dist for a given r_dist / max_r_orig."""
        if model == "ptlens":
            a = params.get("a", 0.0)
            b = params.get("b", 0.0)
            c = params.get("c", 0.0)
            d = 1.0 - a - b - c
            return a * rn**3 + b * rn**2 + c * rn + d
        elif model == "poly3" or model == "manual":
            k1 = params.get("k1", 0.0)
            return 1.0 + k1 * rn**2
        return 1.0

    # Points to check: corners and mid-edges
    rn_h = (fw / 2.0) / max_r_orig
    rn_v = (fh / 2.0) / max_r_orig

    # We check corners (rn=1.0), mid-horizontal, and mid-vertical
    test_radii = [1.0, rn_h, rn_v]

    max_rescale = 1.0
    for rn in test_radii:
        rescale = get_rescale(rn)
        if rescale > max_rescale:
            max_rescale = rescale

    # To keep the sample within original bounds, we need to zoom in the sampling.
    # If r_src = r_dist * rescale, we want r_src <= r_orig_bound.
    # Zooming in (magnifying) means we divide the sampling coordinates by zoom.
    # r_src_zoomed = (r_dist / zoom) * get_rescale(r_dist / (zoom * max_r_orig))
    # This is complex. But for small rescale, zoom = max_rescale is a very good approx.
    return max_rescale


def apply_lens_correction(
    img, settings, lens_info=None, scale=1.0, roi_offset=None, full_size=None
):
    """
    Main entry point for lens correction in the pipeline.
    """
    if img is None:
        return None

    h, w = img.shape[:2]

    # 1. Determine Full Tier Size (for normalization)
    if full_size:
        fw, fh = full_size
    else:
        fw, fh = w, h

    if roi_offset:
        rx, ry = roi_offset
    else:
        rx, ry = 0, 0

    # Relative center in the current image chunk
    cx = fw / 2.0 - rx
    cy = fh / 2.0 - ry

    # 2. Get Correction Model and Params
    model = "manual"
    params = {}

    if lens_info:
        if "distortion" in lens_info and lens_info["distortion"]:
            dist = lens_info["distortion"]
            model = dist.get("model", "poly3")
            params = dist
        elif "params" in lens_info and lens_info["params"]:
            params = lens_info["params"]
            model = params.get("model", "poly3")

    # 3. Add manual slider override
    manual_k1 = settings.get("lens_distortion", 0.0)

    # 4. Handle Auto Crop
    zoom = 1.0
    if settings.get("lens_autocrop", True):
        # We need a copy of params with manual k1 added for scale calculation
        calc_params = params.copy()
        if model == "ptlens":
            calc_params["c"] = params.get("c", 0.0) + manual_k1
        else:
            calc_params["k1"] = params.get("k1", 0.0) + manual_k1

        zoom = calculate_autocrop_scale(model, calc_params, fw, fh)

    # 5. Apply based on model
    map_x, map_y = None, None

    if model == "ptlens":
        a = params.get("a", 0.0)
        b = params.get("b", 0.0)
        c = params.get("c", 0.0)
        # Apply manual k1 as an offset to c
        c += manual_k1
        if abs(a) > 1e-6 or abs(b) > 1e-6 or abs(c) > 1e-6 or abs(zoom - 1.0) > 1e-6:
            map_x, map_y = generate_ptlens_map(w, h, a, b, c, cx, cy, fw, fh, zoom)

    elif model == "poly3" or model == "manual":
        k1 = params.get("k1", 0.0) + manual_k1
        if abs(k1) > 1e-6 or abs(zoom - 1.0) > 1e-6:
            map_x, map_y = generate_poly3_map(w, h, k1, cx, cy, fw, fh, zoom)

    if map_x is not None:
        return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)

    return img
