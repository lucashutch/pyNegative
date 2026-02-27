import logging
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

from ..utils.numba_lens import (
    generate_poly3_map,
    generate_ptlens_map,
    generate_tca_maps,
    remap_tca_distortion_kernel,
    vignette_kernel,
)

logger = logging.getLogger(__name__)


class LensMapCache:
    """
    Cache for lens distortion maps to avoid expensive re-generation.
    Uses LRU (Least Recently Used) eviction policy.
    """

    def __init__(self, max_entries: int = 8):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def get_maps(
        self,
        w: int,
        h: int,
        model: str,
        params: dict,
        cx: float,
        cy: float,
        fw: int,
        fh: int,
        zoom: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        params_tuple = tuple(sorted(params.items()))
        key = (w, h, model, params_tuple, cx, cy, fw, fh, zoom)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        map_x, map_y = None, None
        if model == "ptlens":
            a = params.get("a", 0.0)
            b = params.get("b", 0.0)
            c = params.get("c", 0.0)
            map_x, map_y = generate_ptlens_map(w, h, a, b, c, cx, cy, fw, fh, zoom)
        elif model == "poly3" or model == "manual":
            k1 = params.get("k1", 0.0)
            map_x, map_y = generate_poly3_map(w, h, k1, cx, cy, fw, fh, zoom)

        if map_x is not None:
            with self._lock:
                if len(self._cache) >= self._max_entries:
                    self._cache.popitem(last=False)
                self._cache[key] = (map_x, map_y)

        return map_x, map_y

    def clear(self):
        with self._lock:
            self._cache.clear()


# Global instance for the pipeline
_LENS_MAP_CACHE = LensMapCache()


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


def get_tca_distortion_maps(
    w: int,
    h: int,
    settings: dict,
    lens_info: dict | None = None,
    roi_offset: tuple[int, int] | None = None,
    full_size: tuple[int, int] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """
    Returns 3 sets of maps for TCA + Distortion.
    """
    if full_size:
        fw, fh = full_size
    else:
        fw, fh = w, h

    if roi_offset:
        rx, ry = roi_offset
    else:
        rx, ry = 0, 0

    cx = fw / 2.0 - rx
    cy = fh / 2.0 - ry

    # 1. Resolve model and params
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

    manual_k1 = settings.get("lens_distortion", 0.0)
    ca_intensity = settings.get("lens_ca", 1.0)

    # 2. TCA Params
    tca_red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tca_blue = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if lens_info and "tca" in lens_info and lens_info["tca"]:
        tca = lens_info["tca"]
        tca_red[0] = 1.0 + (tca.get("vr0", 1.0) - 1.0) * ca_intensity
        tca_red[1] = tca.get("vr1", 0.0) * ca_intensity
        tca_red[2] = tca.get("vr2", 0.0) * ca_intensity

        tca_blue[0] = 1.0 + (tca.get("vb0", 1.0) - 1.0) * ca_intensity
        tca_blue[1] = tca.get("vb1", 0.0) * ca_intensity
        tca_blue[2] = tca.get("vb2", 0.0) * ca_intensity

    # 3. Autocrop
    zoom = 1.0
    if settings.get("lens_autocrop", True):
        calc_params = params.copy()
        if model == "ptlens":
            calc_params["c"] = params.get("c", 0.0) + manual_k1
        else:
            calc_params["k1"] = params.get("k1", 0.0) + manual_k1
        zoom = calculate_autocrop_scale(model, calc_params, fw, fh)

    # 4. Prepare params for kernel
    dist_p = np.zeros(4, dtype=np.float32)
    m_type = 0  # ptlens
    if model == "ptlens":
        dist_p[0] = params.get("a", 0.0)
        dist_p[1] = params.get("b", 0.0)
        dist_p[2] = params.get("c", 0.0) + manual_k1
        dist_p[3] = 1.0 - dist_p[0] - dist_p[1] - dist_p[2]
        m_type = 0
    else:
        dist_p[0] = params.get("k1", 0.0) + manual_k1
        m_type = 1  # poly3

    # 5. Generate
    maps = generate_tca_maps(
        w, h, m_type, dist_p, tca_red, tca_blue, cx, cy, fw, fh, zoom
    )
    return (*maps, zoom)


def get_lens_distortion_maps(
    w: int,
    h: int,
    settings: dict,
    lens_info: dict | None = None,
    roi_offset: tuple[int, int] | None = None,
    full_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Resolves lens parameters and retrieves (cached) distortion maps.
    Returns (map_x, map_y, zoom).
    """
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
        calc_params = params.copy()
        if model == "ptlens":
            calc_params["c"] = params.get("c", 0.0) + manual_k1
        else:
            calc_params["k1"] = params.get("k1", 0.0) + manual_k1

        zoom = calculate_autocrop_scale(model, calc_params, fw, fh)

    # 5. Retrieve Maps
    map_x, map_y = None, None
    if model == "ptlens":
        a = params.get("a", 0.0)
        b = params.get("b", 0.0)
        c = params.get("c", 0.0) + manual_k1
        if abs(a) > 1e-6 or abs(b) > 1e-6 or abs(c) > 1e-6 or abs(zoom - 1.0) > 1e-6:
            map_x, map_y = _LENS_MAP_CACHE.get_maps(
                w, h, "ptlens", {"a": a, "b": b, "c": c}, cx, cy, fw, fh, zoom
            )
    elif model == "poly3" or model == "manual":
        k1 = params.get("k1", 0.0) + manual_k1
        if abs(k1) > 1e-6 or abs(zoom - 1.0) > 1e-6:
            map_x, map_y = _LENS_MAP_CACHE.get_maps(
                w, h, "poly3", {"k1": k1}, cx, cy, fw, fh, zoom
            )

    return map_x, map_y, zoom


def apply_lens_correction(
    img,
    settings,
    lens_info=None,
    scale=1.0,
    roi_offset=None,
    full_size=None,
    skip_vignette=False,
):
    """
    Main entry point for lens correction in the pipeline.
    If skip_vignette is True, vignette is not applied (caller handles it).
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

    # 2. Add manual slider override
    manual_vig = settings.get("lens_vignette", 0.0)
    ca_intensity = settings.get("lens_ca", 1.0)  # Default to 1.0 (full DB effect)

    # 3. TCA Params
    tca_red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tca_blue = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    has_tca_data = False

    if lens_info and "tca" in lens_info and lens_info["tca"]:
        tca = lens_info["tca"]
        # Lensfun poly3: v0 + v1*r^2 + v2*r^4
        # We apply intensity to the deviation from 1.0
        tca_red[0] = 1.0 + (tca.get("vr0", 1.0) - 1.0) * ca_intensity
        tca_red[1] = tca.get("vr1", 0.0) * ca_intensity
        tca_red[2] = tca.get("vr2", 0.0) * ca_intensity

        tca_blue[0] = 1.0 + (tca.get("vb0", 1.0) - 1.0) * ca_intensity
        tca_blue[1] = tca.get("vb1", 0.0) * ca_intensity
        tca_blue[2] = tca.get("vb2", 0.0) * ca_intensity
        has_tca_data = True
        logger.debug(
            f"TCA data found for lens. Intensity: {ca_intensity:.2f}, R coeffs: {tca_red}, B coeffs: {tca_blue}"
        )

    # 4. Apply Correction
    do_tca = has_tca_data and abs(ca_intensity) > 1e-3
    map_x, map_y = None, None
    zoom = 1.0
    manual_k1 = settings.get("lens_distortion", 0.0)
    model = "manual"

    if do_tca:
        # We still need to resolve params for the kernel
        params = {}
        if lens_info:
            if "distortion" in lens_info and lens_info["distortion"]:
                dist = lens_info["distortion"]
                model = dist.get("model", "poly3")
                params = dist
            elif "params" in lens_info and lens_info["params"]:
                params = lens_info["params"]
                model = params.get("model", "poly3")

        if settings.get("lens_autocrop", True):
            calc_params = params.copy()
            if model == "ptlens":
                calc_params["c"] = params.get("c", 0.0) + manual_k1
            else:
                calc_params["k1"] = params.get("k1", 0.0) + manual_k1
            zoom = calculate_autocrop_scale(model, calc_params, fw, fh)

        # Prepare distortion params for kernel
        dist_p = np.zeros(4, dtype=np.float32)
        m_type = 0  # ptlens
        if model == "ptlens":
            dist_p[0] = params.get("a", 0.0)
            dist_p[1] = params.get("b", 0.0)
            dist_p[2] = params.get("c", 0.0) + manual_k1
            dist_p[3] = 1.0 - dist_p[0] - dist_p[1] - dist_p[2]
            m_type = 0
        else:
            dist_p[0] = params.get("k1", 0.0) + manual_k1
            m_type = 1  # poly3

        out = np.zeros_like(img)
        t0 = time.perf_counter()
        remap_tca_distortion_kernel(
            img, out, m_type, dist_p, tca_red, tca_blue, cx, cy, fw, fh, zoom
        )
        img = out
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"Combined TCA+Distortion kernel: {elapsed:.2f}ms")
    else:
        # Distortion only via cached maps
        map_x, map_y, zoom = get_lens_distortion_maps(
            w, h, settings, lens_info, roi_offset, full_size
        )

        # Recover model for logging (optional, but let's keep it accurate)
        if lens_info:
            if "distortion" in lens_info and lens_info["distortion"]:
                model = lens_info["distortion"].get("model", "poly3")
            elif "params" in lens_info and lens_info["params"]:
                model = lens_info["params"].get("model", "poly3")

        if map_x is not None:
            img = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)

    # 5. Apply Vignette Correction
    vig_k1 = 0.0
    vig_k2 = 0.0
    vig_k3 = 0.0

    if not skip_vignette:
        if lens_info and "vignetting" in lens_info and lens_info["vignetting"]:
            vig = lens_info["vignetting"]
            if vig.get("model") == "pa":
                vig_k1 = vig.get("k1", 0.0)
                vig_k2 = vig.get("k2", 0.0)
                vig_k3 = vig.get("k3", 0.0)

        manual_vig = settings.get("lens_vignette", 0.0)
        vig_k1 += manual_vig

    if abs(vig_k1) > 1e-6 or abs(vig_k2) > 1e-6 or abs(vig_k3) > 1e-6:
        # If we didn't remap, we should work on a copy to avoid side effects
        if map_x is None and not do_tca:
            img = img.copy()

        t0 = time.perf_counter()
        vignette_kernel(img, vig_k1, vig_k2, vig_k3, cx, cy, fw, fh)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.debug(
            f"Vignette correction: k1={vig_k1:.4f}, k2={vig_k2:.4f}, k3={vig_k3:.4f} ({elapsed:.2f}ms)"
        )

    if (
        do_tca
        or map_x is not None
        or abs(vig_k1) > 1e-6
        or abs(vig_k2) > 1e-6
        or abs(vig_k3) > 1e-6
    ):
        logger.debug(
            f"Lens Correction applied: model={model}, zoom={zoom:.3f}, manual_dist={manual_k1:.4f}, vignette={vig_k1:.4f}, tca={do_tca}"
        )

    return img
