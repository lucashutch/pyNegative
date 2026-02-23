import cv2
import numpy as np
from ... import core as pynegative
from ...processing.geometry import GeometryResolver


def process_denoise_stage(img, res_key, heavy_params, zoom_scale, preprocess_key=None):
    """Processes and caches the denoising stage."""
    l_str = float(heavy_params.get("denoise_luma", 0))
    c_str = float(heavy_params.get("denoise_chroma", 0))

    if l_str <= 0 and c_str <= 0:
        return img

    requested_method = heavy_params.get("denoise_method", "High Quality")
    effective_method = requested_method

    if "NLMeans" in requested_method:
        is_roi = isinstance(res_key, tuple)
        if is_roi:
            if zoom_scale < 0.95:
                if "High Quality" in requested_method or "Hybrid" in requested_method:
                    effective_method = "NLMeans (Numba Hybrid YUV)"
            else:
                effective_method = requested_method
        elif res_key in [
            "tier_0.0625",
            "tier_0.125",
            "tier_0.25",
            "preview",
            "quarter",
        ]:
            effective_method = "NLMeans (Numba Ultra Fast YUV)"
        elif res_key in ["tier_0.5", "half"]:
            effective_method = "NLMeans (Numba Fast+ YUV)"
        elif res_key in ["tier_1.0", "full"]:
            if "High Quality" in requested_method or "Hybrid" in requested_method:
                effective_method = "NLMeans (Numba Hybrid YUV)"
            else:
                effective_method = requested_method

    processed = pynegative.de_noise_image(
        img,
        luma_strength=l_str,
        chroma_strength=c_str,
        method=effective_method,
        zoom=zoom_scale,
        tier=res_key,
    )

    return processed


def process_heavy_stage(
    img,
    res_key,
    heavy_params,
    zoom_scale,
    last_heavy_adjusted="de_haze",
    dehaze_atmospheric_light=None,
    preprocess_key=None,
):
    """Processes and caches the heavy effects stage."""
    dehaze_p = {"de_haze": heavy_params["de_haze"]}
    sharpen_p = {
        "sharpen_value": heavy_params["sharpen_value"],
        "sharpen_radius": heavy_params["sharpen_radius"],
        "sharpen_percent": heavy_params["sharpen_percent"],
    }

    def apply_dehaze(image):
        if dehaze_p["de_haze"] <= 0:
            return image
        processed, atmos = pynegative.de_haze_image(
            image,
            dehaze_p["de_haze"],
            zoom=zoom_scale,
            fixed_atmospheric_light=dehaze_atmospheric_light,
        )
        return processed

    def apply_sharpen(image):
        if sharpen_p["sharpen_value"] <= 0:
            return image
        return pynegative.sharpen_image(
            image,
            sharpen_p["sharpen_radius"],
            sharpen_p["sharpen_percent"],
            "High Quality",
        )

    active = last_heavy_adjusted
    if active == "de_haze":
        pipeline = [
            ("sharpen", sharpen_p, apply_sharpen),
            ("dehaze", dehaze_p, apply_dehaze),
        ]
    else:
        pipeline = [
            ("dehaze", dehaze_p, apply_dehaze),
            ("sharpen", sharpen_p, apply_sharpen),
        ]

    processed = img
    accumulated_params = {}
    if preprocess_key is not None:
        accumulated_params["_preprocess_key"] = preprocess_key

    # Scale sharpen radius by the tier resolution, NOT by tile/full ratio.
    try:
        tier_scale = float(res_key.split("_")[1])
    except (IndexError, ValueError):
        tier_scale = 1.0

    adj_sharpen_p = sharpen_p.copy()
    adj_sharpen_p["sharpen_radius"] = max(0.3, sharpen_p["sharpen_radius"] * tier_scale)

    def apply_sharpen_scaled(image):
        if adj_sharpen_p["sharpen_value"] <= 0:
            return image
        return pynegative.sharpen_image(
            image,
            adj_sharpen_p["sharpen_radius"],
            adj_sharpen_p["sharpen_percent"],
            "High Quality",
        )

    pipeline_scaled = []
    for name, p, func in pipeline:
        if name == "sharpen":
            pipeline_scaled.append((name, adj_sharpen_p, apply_sharpen_scaled))
        else:
            pipeline_scaled.append((name, p, func))

    for _i, (_name, _params, func) in enumerate(pipeline_scaled):
        processed = func(processed)

    return processed


def get_fused_geometry(
    settings,
    lens_info,
    w_src,
    h_src,
    rotate_val,
    crop_val,
    flip_h,
    flip_v,
    ts_roi=1.0,
    roi_offset=None,
    full_size=None,
    M_override=None,
    out_size_override=None,
):
    """
    Calculates fused maps for lens correction and affine transforms.
    Returns (list_of_maps, out_w, out_h, zoom_factor).
    Each element in list_of_maps is (map_x, map_y).
    If no maps, list_of_maps contains only [M].
    """
    # 1. Setup Resolver
    resolver = GeometryResolver(w_src, h_src)
    if M_override is not None:
        if M_override.shape == (2, 3):
            m33 = np.eye(3, dtype=np.float32)
            m33[:2, :] = M_override
            resolver.matrix = m33
        else:
            resolver.matrix = M_override.copy()

        if out_size_override:
            resolver.full_w, resolver.full_h = out_size_override
    else:
        resolver.resolve(
            rotate=rotate_val,
            crop=crop_val,
            flip_h=flip_h,
            flip_v=flip_v,
            expand=True,
        )

    M = resolver.get_matrix_2x3()
    out_w, out_h = resolver.get_output_size()
    out_w, out_h = int(round(out_w)), int(round(out_h))

    # 2. Get Lens Maps (TCA aware)
    zoom_factor = 1.0
    fused_maps = []

    if settings.get("lens_enabled", True):
        ca_intensity = settings.get("lens_ca", 1.0)
        has_tca = lens_info and "tca" in lens_info and abs(ca_intensity) > 1e-3

        from ...processing.lens import (
            get_lens_distortion_maps,
            get_tca_distortion_maps,
        )

        if has_tca:
            xr, yr, xg, yg, xb, yb, zoom_factor = get_tca_distortion_maps(
                w_src, h_src, settings, lens_info, roi_offset, full_size
            )
            # Fuse each channel's map
            fused_maps.append(resolver.get_fused_maps(xr, yr))
            fused_maps.append(resolver.get_fused_maps(xg, yg))
            fused_maps.append(resolver.get_fused_maps(xb, yb))
        else:
            mx, my, zoom_factor = get_lens_distortion_maps(
                w_src, h_src, settings, lens_info, roi_offset, full_size
            )
            if mx is not None:
                fused_maps.append(resolver.get_fused_maps(mx, my))

    if not fused_maps:
        # Fallback to affine only
        fused_maps = [M]

    return fused_maps, out_w, out_h, zoom_factor


def resolve_vignette_params(settings, lens_info, roi_offset=None, full_size=None):
    """Resolve vignette parameters from settings and lens_info."""
    if not settings.get("lens_enabled", True):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0

    vignette = settings.get("lens_vignette", 0.0)
    vig_k1 = vignette
    vig_k2 = 0.0
    vig_k3 = 0.0

    if lens_info and "vignetting" in lens_info and lens_info["vignetting"]:
        vig = lens_info["vignetting"]
        if vig.get("model") == "pa":
            vig_k1 += vig.get("k1", 0.0)
            vig_k2 = vig.get("k2", 0.0)
            vig_k3 = vig.get("k3", 0.0)

    if abs(vig_k1) < 1e-6 and abs(vig_k2) < 1e-6 and abs(vig_k3) < 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0

    if full_size:
        fw, fh = full_size
    else:
        fh, fw = 0, 0

    if roi_offset:
        rx, ry = roi_offset
    else:
        rx, ry = 0, 0

    cx = fw / 2.0 - rx
    cy = fh / 2.0 - ry

    return vig_k1, vig_k2, vig_k3, cx, cy, float(fw), float(fh)


def process_lens_vignette(
    img, scale, settings, lens_info, roi_offset=None, full_size=None
):
    if not settings.get("lens_enabled", True):
        return img

    vignette = settings.get("lens_vignette", 0.0)
    has_auto_vig = False
    if lens_info:
        has_auto_vig = "vignetting" in lens_info and lens_info["vignetting"] is not None

    if abs(vignette) < 1e-5 and not has_auto_vig:
        return img

    from ...processing.lens import vignette_kernel

    # Resolve center
    if full_size:
        fw, fh = full_size
    else:
        fh, fw = img.shape[:2]

    if roi_offset:
        rx, ry = roi_offset
    else:
        rx, ry = 0, 0

    cx = fw / 2.0 - rx
    cy = fh / 2.0 - ry

    vig_k1 = vignette
    vig_k2 = 0.0
    vig_k3 = 0.0

    if lens_info and "vignetting" in lens_info and lens_info["vignetting"]:
        vig = lens_info["vignetting"]
        if vig.get("model") == "pa":
            vig_k1 += vig.get("k1", 0.0)
            vig_k2 = vig.get("k2", 0.0)
            vig_k3 = vig.get("k3", 0.0)

    if abs(vig_k1) > 1e-6 or abs(vig_k2) > 1e-6 or abs(vig_k3) > 1e-6:
        img = img.copy()  # Avoid modifying source
        vignette_kernel(img, vig_k1, vig_k2, vig_k3, cx, cy, fw, fh)

    return img


def apply_fused_remap(
    img,
    fused_maps,
    out_w,
    out_h,
    interpolation=cv2.INTER_CUBIC,
    crop_offset=(0, 0),
    dest_roi=None,
):
    """Applies one or more fused maps to an image, optionally localized to a ROI."""
    crop_x, crop_y = crop_offset

    if dest_roi is not None:
        out_vx, out_vy, out_vw, out_vh = dest_roi
    else:
        out_vx, out_vy, out_vw, out_vh = 0, 0, out_w, out_h

    if len(fused_maps) == 3:
        # TCA case
        channels = cv2.split(img)
        remapped_channels = []
        for i in range(3):
            mx, my = fused_maps[i]
            if dest_roi is not None:
                mx_patch = (
                    mx[out_vy : out_vy + out_vh, out_vx : out_vx + out_vw] - crop_x
                )
                my_patch = (
                    my[out_vy : out_vy + out_vh, out_vx : out_vx + out_vw] - crop_y
                )
            else:
                mx_patch = mx
                my_patch = my
            remapped_channels.append(
                cv2.remap(
                    channels[i],
                    mx_patch,
                    my_patch,
                    interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            )
        return cv2.merge(remapped_channels)
    elif len(fused_maps) == 1:
        m = fused_maps[0]
        if isinstance(m, tuple):
            # Distortion map
            mx, my = m
            if dest_roi is not None:
                mx_patch = (
                    mx[out_vy : out_vy + out_vh, out_vx : out_vx + out_vw] - crop_x
                )
                my_patch = (
                    my[out_vy : out_vy + out_vh, out_vx : out_vx + out_vw] - crop_y
                )
            else:
                mx_patch = mx
                my_patch = my
            return cv2.remap(
                img,
                mx_patch,
                my_patch,
                interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            # Affine matrix
            if dest_roi is not None:
                Hom_M_full = np.vstack([m, [0, 0, 1]])
                T_src = np.array(
                    [[1, 0, crop_x], [0, 1, crop_y], [0, 0, 1]], dtype=np.float32
                )
                T_dst_inv = np.array(
                    [[1, 0, -out_vx], [0, 1, -out_vy], [0, 0, 1]], dtype=np.float32
                )
                Hom_M_new = T_dst_inv @ Hom_M_full @ T_src
                M_new = Hom_M_new[:2, :]
                return cv2.warpAffine(
                    img,
                    M_new,
                    (out_vw, out_vh),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            else:
                return cv2.warpAffine(
                    img,
                    m,
                    (out_w, out_h),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
    return img


def calculate_histograms(img_array):
    """Calculate and smooth RGB histograms."""
    histograms = {}
    for i, col in enumerate(["r", "g", "b"]):
        histWindow = img_array[:, :, i]
        hist = cv2.calcHist([histWindow], [0], None, [256], [0, 256])

        # Smooth the histogram for better visual appearance
        h = hist.flatten()

        def smooth(h):
            return np.convolve(h, np.ones(5) / 5, mode="same")

        # Apply multiple smoothing passes
        h = smooth(h)
        h = smooth(h)
        h = smooth(h)

        histograms[col] = h
    return histograms
