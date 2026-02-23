import logging
import time

import numpy as np

from ..utils.numba_defringe import defringe_kernel

logger = logging.getLogger(__name__)


def apply_defringe(img, settings):
    """
    Applies manual defringing to remove purple and green fringes from high-contrast edges.
    Operates on post-tone mapped (but still float32) image data.
    """
    purple_thresh = settings.get("defringe_purple", 0.0)
    green_thresh = settings.get("defringe_green", 0.0)
    edge_thresh = settings.get("defringe_edge", 0.05)
    radius = settings.get("defringe_radius", 1.0)

    if purple_thresh <= 0 and green_thresh <= 0:
        return img

    start_time = time.perf_counter()

    out = np.empty_like(img)

    # Thresholds from sliders are typically 0-100 or 0-1
    # We expect 0.0 to 1.0 here for the kernel.
    p_t = float(purple_thresh) / 100.0 if purple_thresh > 1.0 else float(purple_thresh)
    g_t = float(green_thresh) / 100.0 if green_thresh > 1.0 else float(green_thresh)
    e_t = float(edge_thresh)
    r = float(radius)

    defringe_kernel(img, out, p_t, g_t, e_t, r)

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Defringe applied: purple={p_t:.2f}, green={g_t:.2f}, edge={e_t:.2f}, radius={r:.1f} ({elapsed:.2f}ms)"
    )

    return out
