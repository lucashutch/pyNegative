"""Numba JIT kernels for lens correction: distortion, vignette, TCA."""

import numpy as np

from ._numba_base import njit, prange


@njit(fastmath=True, cache=True, parallel=True)
def generate_ptlens_map(w, h, a, b, c, cx, cy, full_w, full_h, zoom=1.0):
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    hw = full_w / 2.0
    hh = full_h / 2.0
    max_r = np.sqrt(hw * hw + hh * hh)
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
            rn2 = rn * rn
            rescale = a * rn2 * rn + b * rn2 + c * rn + d

            map_x[y, x] = cx + dx * rescale
            map_y[y, x] = cy + dy * rescale

    return map_x, map_y


@njit(fastmath=True, cache=True, parallel=True)
def generate_poly3_map(w, h, k1, cx, cy, full_w, full_h, zoom=1.0):
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    hw = full_w / 2.0
    hh = full_h / 2.0
    max_r2 = hw * hw + hh * hh
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


@njit(fastmath=True, cache=True, inline="always")
def _bilinear_sample(img, x, y, c, h, w):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return 0.0

    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0, c]
    v10 = img[y0, x1, c]
    v01 = img[y1, x0, c]
    v11 = img[y1, x1, c]

    return (
        v00 * (1 - dx) * (1 - dy)
        + v10 * dx * (1 - dy)
        + v01 * (1 - dx) * dy
        + v11 * dx * dy
    )


@njit(fastmath=True, cache=True, inline="always")
def _get_distortion_rescale(rn, model_type, p):
    if model_type == 0:  # ptlens
        # p = [a, b, c, d]
        rn2 = rn * rn
        return p[0] * rn2 * rn + p[1] * rn2 + p[2] * rn + p[3]
    elif model_type == 1:  # poly3
        # p = [k1]
        return 1.0 + p[0] * rn * rn
    return 1.0


@njit(fastmath=True, cache=True, inline="always")
def _get_tca_rescale(rn, p):
    # Lensfun poly3 TCA: r_src = r_dist * (v0 + v1*r^2 + v2*r^4)
    rn2 = rn * rn
    return p[0] + p[1] * rn2 + p[2] * rn2 * rn2


@njit(fastmath=True, cache=True, parallel=True)
def remap_tca_distortion_kernel(
    img,
    out,
    model_type,
    dist_params,
    tca_red,
    tca_blue,
    cx,
    cy,
    full_w,
    full_h,
    zoom=1.0,
):
    h, w, channels = img.shape
    hw = full_w / 2.0
    hh = full_h / 2.0
    max_r = np.sqrt(hw * hw + hh * hh)
    inv_max_r = 1.0 / max_r
    inv_zoom = 1.0 / zoom

    for y in prange(h):
        for x in range(w):
            # Coordinates relative to center
            dx = (x - cx) * inv_zoom
            dy = (y - cy) * inv_zoom

            r = np.sqrt(dx * dx + dy * dy)
            rn = r * inv_max_r

            # 1. Base Distortion
            dist_rescale = _get_distortion_rescale(rn, model_type, dist_params)

            # 2. TCA Rescale (Relative to distorted Green)
            tca_r_rescale = _get_tca_rescale(rn, tca_red)
            tca_b_rescale = _get_tca_rescale(rn, tca_blue)

            # Fetch coordinates
            # Green (Reference)
            gx = cx + dx * dist_rescale
            gy = cy + dy * dist_rescale

            # Red
            rx = cx + dx * dist_rescale * tca_r_rescale
            ry = cy + dy * dist_rescale * tca_r_rescale

            # Blue
            bx = cx + dx * dist_rescale * tca_b_rescale
            by = cy + dy * dist_rescale * tca_b_rescale

            # Sample
            if channels >= 3:
                out[y, x, 0] = _bilinear_sample(img, rx, ry, 0, h, w)
                out[y, x, 1] = _bilinear_sample(img, gx, gy, 1, h, w)
                out[y, x, 2] = _bilinear_sample(img, bx, by, 2, h, w)
            else:
                out[y, x, 0] = _bilinear_sample(img, gx, gy, 0, h, w)


@njit(fastmath=True, cache=True, parallel=True)
def vignette_kernel(img, k1, k2, k3, cx, cy, full_w, full_h):
    h, w, c = img.shape
    max_r2 = (full_w / 2.0) ** 2 + (full_h / 2.0) ** 2
    inv_max_r2 = 1.0 / max_r2

    for y in prange(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy
            rn2 = r2 * inv_max_r2
            rn4 = rn2 * rn2
            rn6 = rn4 * rn2

            # I_corr = I_dist * (1 + k1*r^2 + k2*r^4 + k3*r^6)
            gain = 1.0 + k1 * rn2 + k2 * rn4 + k3 * rn6

            # Explicit channel unrolling for better ILP
            img[y, x, 0] *= gain
            img[y, x, 1] *= gain
            img[y, x, 2] *= gain


@njit(fastmath=True, cache=True, parallel=True)
def generate_tca_maps(
    w, h, model_type, dist_params, tca_red, tca_blue, cx, cy, full_w, full_h, zoom=1.0
):
    """
    Generates 3 sets of maps (Red, Green, Blue) for TCA correction.
    """
    map_xr = np.zeros((h, w), dtype=np.float32)
    map_yr = np.zeros((h, w), dtype=np.float32)
    map_xg = np.zeros((h, w), dtype=np.float32)
    map_yg = np.zeros((h, w), dtype=np.float32)
    map_xb = np.zeros((h, w), dtype=np.float32)
    map_yb = np.zeros((h, w), dtype=np.float32)

    hw = full_w / 2.0
    hh = full_h / 2.0
    max_r = np.sqrt(hw * hw + hh * hh)
    inv_max_r = 1.0 / max_r
    inv_zoom = 1.0 / zoom

    for y in prange(h):
        for x in range(w):
            dx = (x - cx) * inv_zoom
            dy = (y - cy) * inv_zoom

            r = np.sqrt(dx * dx + dy * dy)
            rn = r * inv_max_r

            dist_rescale = _get_distortion_rescale(rn, model_type, dist_params)
            tca_r_rescale = _get_tca_rescale(rn, tca_red)
            tca_b_rescale = _get_tca_rescale(rn, tca_blue)

            # Green (Reference)
            map_xg[y, x] = cx + dx * dist_rescale
            map_yg[y, x] = cy + dy * dist_rescale

            # Red
            map_xr[y, x] = cx + dx * dist_rescale * tca_r_rescale
            map_yr[y, x] = cy + dy * dist_rescale * tca_r_rescale

            # Blue
            map_xb[y, x] = cx + dx * dist_rescale * tca_b_rescale
            map_yb[y, x] = cy + dy * dist_rescale * tca_b_rescale

    return map_xr, map_yr, map_xg, map_yg, map_xb, map_yb
