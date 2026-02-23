import logging
import time

import cv2
import numpy as np
from PySide6 import QtCore, QtGui

from ... import core as pynegative
from ...processing.geometry import GeometryResolver

logger = logging.getLogger(__name__)


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(
        QtGui.QPixmap, int, int, float, object, object, int, object, int, int
    )
    histogramUpdated = QtCore.Signal(dict, int)
    tierGenerated = QtCore.Signal(float, object, int)  # scale, array, image_id
    uneditedPixmapGenerated = QtCore.Signal(QtGui.QPixmap, int)
    error = QtCore.Signal(str, int)


class TierGeneratorWorker(QtCore.QRunnable):
    """Worker to generate scaled image tiers in the background."""

    def __init__(self, signals, img_array, image_id):
        super().__init__()
        self.signals = signals
        self.img_array = img_array
        self.image_id = image_id

    def run(self):
        if self.img_array is None:
            return

        try:
            h, w = self.img_array.shape[:2]

            # 1. Immediate Fast Feedback
            total_pixels = h * w
            scale_fast = 0.0625 if total_pixels >= 40_000_000 else 0.125
            fast_w, fast_h = int(w * scale_fast), int(h * scale_fast)
            preview_fast = cv2.resize(
                self.img_array, (fast_w, fast_h), interpolation=cv2.INTER_LINEAR
            )

            img_uint8 = (np.clip(preview_fast, 0, 1) * 255).astype(np.uint8)
            h_p, w_p, c_p = img_uint8.shape
            qimage = QtGui.QImage(
                img_uint8.data, w_p, h_p, c_p * w_p, QtGui.QImage.Format_RGB888
            )
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.signals.uneditedPixmapGenerated.emit(pixmap, self.image_id)

            # Emit fast tier
            self.signals.tierGenerated.emit(scale_fast, preview_fast, self.image_id)

            # 2. High Quality Pyramid Chain
            scales = [
                round(1.0 / 1.5, 4),
                0.5,
                round(1.0 / 3.0, 4),
                0.25,
                round(1.0 / 6.0, 4),
                0.125,
            ]
            if scale_fast <= 0.0625:
                scales.append(0.0625)

            for scale in scales:
                tw, th = int(w * scale), int(h * scale)
                next_img = cv2.resize(
                    self.img_array, (tw, th), interpolation=cv2.INTER_AREA
                )
                self.signals.tierGenerated.emit(scale, next_img, self.image_id)

        except Exception as e:
            logger.error(f"Tier generation error: {e}")


class ImageProcessorWorker(QtCore.QRunnable):
    """Worker to process a single large ROI in a background thread."""

    def __init__(
        self,
        signals,
        base_img_full,
        tiers,
        settings,
        request_id,
        zoom_scale=1.0,
        is_fitting=True,
        calculate_histogram=False,
        last_heavy_adjusted="de_haze",
        lens_info=None,
        target_on_screen_width=None,
        visible_scene_rect=None,
        tile_key=None,
        render_state_id=0,
        calculate_lowres=True,
        settings_state_id=0,
        dehaze_atmospheric_light=None,
    ):
        super().__init__()
        self.signals = signals
        self.base_img_full = base_img_full
        self.tiers = tiers
        self.settings = settings
        self.request_id = request_id
        self.zoom_scale = zoom_scale
        self.is_fitting = is_fitting
        self.calculate_histogram = calculate_histogram
        self.last_heavy_adjusted = last_heavy_adjusted
        self.lens_info = lens_info
        self.target_on_screen_width = target_on_screen_width
        self.visible_scene_rect = visible_scene_rect
        self.tile_key = tile_key
        self.render_state_id = render_state_id
        self.calculate_lowres = calculate_lowres
        self.settings_state_id = settings_state_id
        self.dehaze_atmospheric_light = dehaze_atmospheric_light

    def run(self):
        try:
            *result, rotate_val, visible_scene_rect_out, bg_lowres_pix = (
                self._update_preview()
            )
            try:
                self.signals.finished.emit(
                    *result,
                    rotate_val,
                    visible_scene_rect_out,
                    bg_lowres_pix,
                    self.request_id,
                    self.tile_key,
                    self.render_state_id,
                    self.settings_state_id,
                )
            except RuntimeError:
                pass  # Signal source deleted (app closing)
        except Exception as e:
            logger.error(f"Image processing worker failed: {e}", exc_info=True)
            try:
                self.signals.error.emit(str(e), self.request_id)
            except RuntimeError:
                pass  # Signal source deleted (app closing)

    def _process_denoise_stage(
        self, img, res_key, heavy_params, zoom_scale, preprocess_key=None
    ):
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
                    if (
                        "High Quality" in requested_method
                        or "Hybrid" in requested_method
                    ):
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

    def _process_heavy_stage(
        self, img, res_key, heavy_params, zoom_scale, preprocess_key=None
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
                fixed_atmospheric_light=self.dehaze_atmospheric_light,
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

        active = self.last_heavy_adjusted
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
        # A 256×256 tile at tier 1.0 is a full-res crop — radius should be unscaled.
        # At tier 0.25 the pixels are 4× larger, so radius should be 0.25×.
        try:
            tier_scale = float(res_key.split("_")[1])
        except (IndexError, ValueError):
            tier_scale = 1.0

        adj_sharpen_p = sharpen_p.copy()
        adj_sharpen_p["sharpen_radius"] = max(
            0.3, sharpen_p["sharpen_radius"] * tier_scale
        )

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

    def _get_fused_geometry(
        self,
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

        if self.settings.get("lens_enabled", True):
            ca_intensity = self.settings.get("lens_ca", 1.0)
            has_tca = (
                self.lens_info and "tca" in self.lens_info and abs(ca_intensity) > 1e-3
            )

            from ...processing.lens import (
                get_lens_distortion_maps,
                get_tca_distortion_maps,
            )

            if has_tca:
                xr, yr, xg, yg, xb, yb, zoom_factor = get_tca_distortion_maps(
                    w_src, h_src, self.settings, self.lens_info, roi_offset, full_size
                )
                # Fuse each channel's map
                fused_maps.append(resolver.get_fused_maps(xr, yr))
                fused_maps.append(resolver.get_fused_maps(xg, yg))
                fused_maps.append(resolver.get_fused_maps(xb, yb))
            else:
                mx, my, zoom_factor = get_lens_distortion_maps(
                    w_src, h_src, self.settings, self.lens_info, roi_offset, full_size
                )
                if mx is not None:
                    fused_maps.append(resolver.get_fused_maps(mx, my))

        if not fused_maps:
            # Fallback to affine only
            fused_maps = [M]

        return fused_maps, out_w, out_h, zoom_factor

    def _resolve_vignette_params(self, roi_offset=None, full_size=None):
        """Resolve vignette parameters from settings and lens_info."""
        if not self.settings.get("lens_enabled", True):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0

        vignette = self.settings.get("lens_vignette", 0.0)
        vig_k1 = vignette
        vig_k2 = 0.0
        vig_k3 = 0.0

        if (
            self.lens_info
            and "vignetting" in self.lens_info
            and self.lens_info["vignetting"]
        ):
            vig = self.lens_info["vignetting"]
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

    def _process_lens_vignette(self, img, scale, roi_offset=None, full_size=None):
        if not self.settings.get("lens_enabled", True):
            return img

        vignette = self.settings.get("lens_vignette", 0.0)
        has_auto_vig = False
        if self.lens_info:
            has_auto_vig = (
                "vignetting" in self.lens_info
                and self.lens_info["vignetting"] is not None
            )

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

        if (
            self.lens_info
            and "vignetting" in self.lens_info
            and self.lens_info["vignetting"]
        ):
            vig = self.lens_info["vignetting"]
            if vig.get("model") == "pa":
                vig_k1 += vig.get("k1", 0.0)
                vig_k2 = vig.get("k2", 0.0)
                vig_k3 = vig.get("k3", 0.0)

        if abs(vig_k1) > 1e-6 or abs(vig_k2) > 1e-6 or abs(vig_k3) > 1e-6:
            img = img.copy()  # Avoid modifying source
            vignette_kernel(img, vig_k1, vig_k2, vig_k3, cx, cy, fw, fh)

        return img

    def _apply_fused_remap(
        self,
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

    def _update_preview(self):
        if self.base_img_full is None:
            return QtGui.QPixmap(), 0, 0, 0.0

        full_h, full_w = self.base_img_full.shape[:2]

        # --- STEP 0: SETTINGS EXTRACTION ---
        rotate_val = self.settings.get("rotation", 0.0)
        flip_h = self.settings.get("flip_h", False)
        flip_v = self.settings.get("flip_v", False)
        crop_val = self.settings.get("crop", None)

        heavy_params = {
            "de_haze": self.settings.get("de_haze", 0) / 50.0,
            "denoise_luma": self.settings.get("denoise_luma", 0),
            "denoise_chroma": self.settings.get("denoise_chroma", 0)
            * 2,  # Scale up max power
            "denoise_method": self.settings.get("denoise_method", "High Quality"),
            "sharpen_value": self.settings.get("sharpen_value", 0),
            "sharpen_radius": self.settings.get("sharpen_radius", 0.5),
            "sharpen_percent": self.settings.get("sharpen_percent", 0.0),
        }

        preprocess_settings = {
            "temperature": self.settings.get("temperature", 0.0),
            "tint": self.settings.get("tint", 0.0),
            "exposure": self.settings.get("exposure", 0.0),
        }

        tone_map_settings = {
            "contrast": self.settings.get("contrast", 0.0),
            "blacks": self.settings.get("blacks", 0.0),
            "whites": self.settings.get("whites", 0.0),
            "shadows": self.settings.get("shadows", 0.0),
            "highlights": self.settings.get("highlights", 0.0),
            "saturation": self.settings.get("saturation", 0.0),
        }

        full_resolver = GeometryResolver(full_w, full_h)
        full_resolver.resolve(
            rotate=rotate_val,
            crop=crop_val,
            flip_h=flip_h,
            flip_v=flip_v,
            expand=True,
        )
        new_full_w, new_full_h = full_resolver.get_output_size()

        # --- STEP 1: RESOLUTION SELECTION ---
        target_on_screen_width = self.target_on_screen_width
        if target_on_screen_width is None:
            target_on_screen_width = full_w

        available_scales = sorted(self.tiers.keys()) + [1.0]
        selected_scale = 1.0
        selected_img = self.base_img_full

        if self.visible_scene_rect is not None:
            # When zooming IN, the self.zoom_scale grows (2.0x, 3.0x, etc).
            # Our tiers are fractions of the original size (0.25, 0.5, 1.0)
            # We want to pick the highest available tier (1.0) when zoom >= 1.0
            if self.zoom_scale >= 1.0:
                selected_scale = 1.0
                selected_img = self.base_img_full
            else:
                for scale in available_scales:
                    tier_img = (
                        self.tiers.get(scale) if scale < 1.0 else self.base_img_full
                    )
                    if tier_img is not None:
                        selected_scale = scale
                        selected_img = tier_img
                        # The highest available scale below or equal to zoom requirement.
                        if scale >= self.zoom_scale:
                            break
        else:
            # Full image bounding calculation
            for scale in available_scales:
                tier_img = self.tiers.get(scale) if scale < 1.0 else self.base_img_full
                if tier_img is not None and tier_img.shape[1] >= target_on_screen_width:
                    selected_scale = scale
                    selected_img = tier_img
                    break

        res_key = f"tier_{selected_scale}"
        h_src, w_src = selected_img.shape[:2]

        crop_x, crop_y, crop_w, crop_h = 0, 0, w_src, h_src
        out_vx, out_vy, out_vw, out_vh = 0, 0, new_full_w, new_full_h

        is_viewport_only = self.visible_scene_rect is not None
        if is_viewport_only:
            out_vx, out_vy, out_vw, out_vh = self.visible_scene_rect

            # Map viewport back to source image to find raw crop bounds
            corners_processed = np.array(
                [
                    [out_vx, out_vy],
                    [out_vx + out_vw, out_vy],
                    [out_vx + out_vw, out_vy + out_vh],
                    [out_vx, out_vy + out_vh],
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            M_full = full_resolver.get_matrix_2x3()
            hom_M = np.vstack([M_full, [0, 0, 1]])
            M_inv = np.linalg.inv(hom_M)[:2, :]

            corners_raw = cv2.transform(corners_processed, M_inv).reshape(-1, 2)
            rx_min, ry_min = corners_raw.min(axis=0)
            rx_max, ry_max = corners_raw.max(axis=0)

            crop_x = int(np.floor(rx_min * selected_scale))
            crop_y = int(np.floor(ry_min * selected_scale))
            crop_w = int(np.ceil((rx_max - rx_min) * selected_scale))
            crop_h = int(np.ceil((ry_max - ry_min) * selected_scale))

            crop_x = max(0, crop_x)
            crop_y = max(0, crop_y)
            crop_w = min(w_src - crop_x, crop_w)
            crop_h = min(h_src - crop_y, crop_h)

            if crop_w <= 0 or crop_h <= 0:
                is_viewport_only = False
                crop_x, crop_y, crop_w, crop_h = 0, 0, w_src, h_src

        vig_params = self._resolve_vignette_params(
            roi_offset=(crop_x, crop_y) if is_viewport_only else None,
            full_size=(w_src, h_src),
        )
        vig_k1, vig_k2, vig_k3, vig_cx, vig_cy, vig_fw, vig_fh = vig_params

        # --- STEP 2: PROCESSING ---
        source_for_preprocess = (
            selected_img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            if is_viewport_only
            else selected_img
        )

        preprocessed_bg = pynegative.apply_preprocess(
            source_for_preprocess,
            temperature=preprocess_settings["temperature"],
            tint=preprocess_settings["tint"],
            exposure=preprocess_settings["exposure"],
            vignette_k1=vig_k1,
            vignette_k2=vig_k2,
            vignette_k3=vig_k3,
            vignette_cx=vig_cx,
            vignette_cy=vig_cy,
            full_width=vig_fw,
            full_height=vig_fh,
        )

        # Denoise only what is seen!
        denoised_bg = self._process_denoise_stage(
            preprocessed_bg, res_key, heavy_params, self.zoom_scale
        )

        fused_maps, o_w, o_h, _ = self._get_fused_geometry(
            w_src,
            h_src,
            rotate_val,
            crop_val,
            flip_h,
            flip_v,
            ts_roi=selected_scale,
            roi_offset=(0, 0),
            full_size=(w_src, h_src),
        )

        if is_viewport_only:
            out_vx_s = int(np.floor(out_vx * selected_scale))
            out_vy_s = int(np.floor(out_vy * selected_scale))
            out_vw_s = int(np.ceil(out_vw * selected_scale))
            out_vh_s = int(np.ceil(out_vh * selected_scale))

            out_vw_s = max(1, min(out_vw_s, o_w - out_vx_s))
            out_vh_s = max(1, min(out_vh_s, o_h - out_vy_s))
            dest_roi_scaled = (out_vx_s, out_vy_s, out_vw_s, out_vh_s)
        else:
            dest_roi_scaled = None

        img_dest = self._apply_fused_remap(
            denoised_bg,
            fused_maps,
            o_w,
            o_h,
            cv2.INTER_CUBIC,
            crop_offset=(crop_x, crop_y) if is_viewport_only else (0, 0),
            dest_roi=dest_roi_scaled,
        )

        processed_bg = self._process_heavy_stage(
            img_dest, res_key, heavy_params, self.zoom_scale
        )

        bg_output, _ = pynegative.apply_tone_map(
            processed_bg, **tone_map_settings, calculate_stats=False
        )
        bg_output = pynegative.apply_defringe(bg_output, self.settings)

        img_uint8 = pynegative.float32_to_uint8(bg_output)

        # -- STEP 3: Handle LowRes Pan Background & Histograms --
        bg_lowres_pix = None
        if self.calculate_lowres:
            if is_viewport_only:
                low_scale = 0.0625
                low_img = self.tiers.get(low_scale)
                if low_img is not None:
                    low_h, low_w = low_img.shape[:2]
                    low_vig_params = self._resolve_vignette_params(None, (low_w, low_h))
                    low_pre = pynegative.apply_preprocess(
                        low_img,
                        temperature=preprocess_settings["temperature"],
                        tint=preprocess_settings["tint"],
                        exposure=preprocess_settings["exposure"],
                        vignette_k1=low_vig_params[0],
                        vignette_k2=low_vig_params[1],
                        vignette_k3=low_vig_params[2],
                        vignette_cx=low_vig_params[3],
                        vignette_cy=low_vig_params[4],
                        full_width=low_vig_params[5],
                        full_height=low_vig_params[6],
                    )

                    low_maps, low_o_w, low_o_h, _ = self._get_fused_geometry(
                        low_w,
                        low_h,
                        rotate_val,
                        crop_val,
                        flip_h,
                        flip_v,
                        ts_roi=low_scale,
                    )

                    low_dest = self._apply_fused_remap(
                        low_pre, low_maps, low_o_w, low_o_h, cv2.INTER_LINEAR
                    )

                    low_heavy = self._process_heavy_stage(
                        low_dest, f"tier_{low_scale}", heavy_params, low_scale
                    )

                    low_out, _ = pynegative.apply_tone_map(
                        low_heavy, **tone_map_settings, calculate_stats=False
                    )
                    low_out = pynegative.apply_defringe(low_out, self.settings)

                    low_uint8 = pynegative.float32_to_uint8(low_out)

                    if self.calculate_histogram:
                        hist_data = self._calculate_histograms(low_uint8)
                        self.signals.histogramUpdated.emit(hist_data, self.request_id)

                    lh, lw = low_uint8.shape[:2]
                    qimg_low = QtGui.QImage(
                        low_uint8.data, lw, lh, 3 * lw, QtGui.QImage.Format_RGB888
                    )
                    bg_lowres_pix = QtGui.QPixmap.fromImage(qimg_low)
            else:
                if self.calculate_histogram:
                    hist_data = self._calculate_histograms(img_uint8)
                    self.signals.histogramUpdated.emit(hist_data, self.request_id)

        h_bg, w_bg = img_uint8.shape[:2]
        qimg_bg = QtGui.QImage(
            img_uint8.data, w_bg, h_bg, 3 * w_bg, QtGui.QImage.Format_RGB888
        )
        pix_bg = QtGui.QPixmap.fromImage(qimg_bg)

        visible_scene_rect_out = (
            (out_vx, out_vy, out_vw, out_vh) if is_viewport_only else None
        )

        return (
            pix_bg,
            new_full_w,
            new_full_h,
            rotate_val,
            visible_scene_rect_out,
            bg_lowres_pix,
        )

    def _calculate_histograms(self, img_array):
        start_time = time.perf_counter()
        h, w = img_array.shape[:2]
        stride = max(1, int(np.sqrt((h * w) / 65536)))
        img_rgb = img_array
        hr, hg, hb, hy, hu, hv = pynegative.numba_histogram_kernel(img_rgb, stride)

        def smooth(h):
            return cv2.GaussianBlur(h.reshape(-1, 1), (5, 5), 0).flatten()

        result = {
            "R": smooth(hr),
            "G": smooth(hg),
            "B": smooth(hb),
            "Y": smooth(hy),
            "U": smooth(hu),
            "V": smooth(hv),
        }
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Histogram calculation: {elapsed:.2f}ms")
        return result
