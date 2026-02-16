import numpy as np
from PySide6 import QtCore, QtGui
import cv2
import time
import logging
from ... import core as pynegative
from ...processing.geometry import GeometryResolver

logger = logging.getLogger(__name__)


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(
        QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int, int, float
    )
    histogramUpdated = QtCore.Signal(dict, int)
    tierGenerated = QtCore.Signal(float, np.ndarray)  # scale, array
    uneditedPixmapGenerated = QtCore.Signal(QtGui.QPixmap)
    error = QtCore.Signal(str, int)


class TierGeneratorWorker(QtCore.QRunnable):
    """Worker to generate scaled image tiers in the background."""

    def __init__(self, signals, img_array):
        super().__init__()
        self.signals = signals
        self.img_array = img_array

    def run(self):
        if self.img_array is None:
            return

        try:
            h, w = self.img_array.shape[:2]

            # 1. Immediate Fast Feedback (1:16)
            scale_fast = 0.0625
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
            self.signals.uneditedPixmapGenerated.emit(pixmap)

            # Emit fast tier
            self.signals.tierGenerated.emit(scale_fast, preview_fast)

            # 2. High Quality Pyramid Chain
            scales = [0.5, 0.25, 0.125, 0.0625]
            current_img = self.img_array

            for scale in scales:
                tw, th = int(w * scale), int(h * scale)
                next_img = cv2.resize(
                    current_img, (tw, th), interpolation=cv2.INTER_AREA
                )
                self.signals.tierGenerated.emit(scale, next_img)
                current_img = next_img

        except Exception as e:
            logger.error(f"Tier generation error: {e}")


class ImageProcessorWorker(QtCore.QRunnable):
    """Worker to process a single large ROI in a background thread."""

    def __init__(
        self,
        signals,
        view_ref,
        base_img_full,
        tiers,
        settings,
        request_id,
        calculate_histogram=False,
        cache=None,
        last_heavy_adjusted="de_haze",
        expand_roi=False,
        lens_info=None,
    ):
        super().__init__()
        self.signals = signals
        self._view_ref = view_ref
        self.base_img_full = base_img_full
        self.tiers = tiers
        self.settings = settings
        self.request_id = request_id
        self.calculate_histogram = calculate_histogram
        self.cache = cache
        self.last_heavy_adjusted = last_heavy_adjusted
        self.expand_roi = expand_roi
        self.lens_info = lens_info

    def run(self):
        try:
            *result, rotate_val = self._update_preview()
            self.signals.finished.emit(*result, rotate_val, self.request_id)
        except Exception as e:
            logger.error(f"Image processing worker failed: {e}", exc_info=True)
            self.signals.error.emit(str(e), self.request_id)

    def _process_denoise_stage(
        self, img, res_key, heavy_params, zoom_scale, preprocess_key=None
    ):
        """Processes and caches the denoising stage."""
        l_str = float(heavy_params.get("denoise_luma", 0))
        c_str = float(heavy_params.get("denoise_chroma", 0))

        if l_str <= 0 and c_str <= 0:
            return img

        requested_method = heavy_params.get("denoise_method", "NLMeans (Numba Fast+)")
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

        denoise_p = {
            "denoise_luma": l_str,
            "denoise_chroma": c_str,
            "denoise_method": effective_method,
        }
        if preprocess_key:
            denoise_p["_preprocess_key"] = preprocess_key

        if self.cache:
            cached = self.cache.get(res_key, "denoise_stage", denoise_p)
            if cached is not None:
                return cached

        processed = pynegative.de_noise_image(
            img,
            luma_strength=l_str,
            chroma_strength=c_str,
            method=effective_method,
            zoom=zoom_scale,
            tier=res_key,
        )

        if self.cache:
            self.cache.put(res_key, "denoise_stage", denoise_p, processed)

        return processed

    def _process_heavy_stage(self, img, res_key, heavy_params, zoom_scale):
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
            atmos_fixed = (
                self.cache.estimated_params.get("atmospheric_light")
                if self.cache
                else None
            )
            processed, atmos = pynegative.de_haze_image(
                image,
                dehaze_p["de_haze"],
                zoom=zoom_scale,
                fixed_atmospheric_light=atmos_fixed,
            )
            if res_key == "preview" and self.cache and atmos_fixed is None:
                self.cache.estimated_params["atmospheric_light"] = atmos
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

        img_w = img.shape[1]
        full_w = self.base_img_full.shape[1]
        scale_factor = img_w / full_w

        adj_sharpen_p = sharpen_p.copy()
        adj_sharpen_p["sharpen_radius"] *= scale_factor

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

        for i, (name, params, func) in enumerate(pipeline_scaled):
            accumulated_params.update(params)
            stage_id = f"heavy_stage_{i + 1}_{name}"

            if self.cache:
                cached = self.cache.get(res_key, stage_id, accumulated_params)
                if cached is not None:
                    processed = cached
                    continue

            processed = func(processed)
            if self.cache:
                self.cache.put(res_key, stage_id, accumulated_params, processed)

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
        self, img, fused_maps, out_w, out_h, interpolation=cv2.INTER_CUBIC
    ):
        """Applies one or more fused maps to an image."""
        if len(fused_maps) == 3:
            # TCA case
            channels = cv2.split(img)
            remapped_channels = []
            for i in range(3):
                mx, my = fused_maps[i]
                remapped_channels.append(
                    cv2.remap(
                        channels[i],
                        mx,
                        my,
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
                return cv2.remap(
                    img,
                    m[0],
                    m[1],
                    interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            else:
                # Affine matrix
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
        if self.base_img_full is None or self._view_ref is None:
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0, 0.0

        full_h, full_w = self.base_img_full.shape[:2]

        # --- STEP 0: SETTINGS EXTRACTION ---
        rotate_val = self.settings.get("rotation", 0.0)
        flip_h = self.settings.get("flip_h", False)
        flip_v = self.settings.get("flip_v", False)
        crop_val = self.settings.get("crop", None)

        heavy_params = {
            "de_haze": self.settings.get("de_haze", 0) / 50.0,
            "denoise_luma": self.settings.get("denoise_luma", 0),
            "denoise_chroma": self.settings.get("denoise_chroma", 0),
            "denoise_method": self.settings.get(
                "denoise_method", "NLMeans (Numba Fast+)"
            ),
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
            "contrast": self.settings.get("contrast", 1.0),
            "blacks": self.settings.get("blacks", 0.0),
            "whites": self.settings.get("whites", 1.0),
            "shadows": self.settings.get("shadows", 0.0),
            "highlights": self.settings.get("highlights", 0.0),
            "saturation": self.settings.get("saturation", 1.0),
        }

        full_vig_params = self._resolve_vignette_params(
            roi_offset=None, full_size=(full_w, full_h)
        )

        # --- STEP 1: RESOLUTION SELECTION ---
        try:
            zoom_scale = self._view_ref.transform().m11()
            view_rect = self._view_ref.viewport().rect()
            v_w = view_rect.width()
            is_fitting = getattr(self._view_ref, "_is_fitting", False)
        except (AttributeError, RuntimeError):
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0, 0.0

        # Optimization: ROI logic
        is_zoomed_in = not is_fitting and zoom_scale > 0.5
        target_on_screen_width = full_w * zoom_scale if not is_fitting else v_w

        # Master Geometry Resolver (The UI scene truth)
        full_resolver = GeometryResolver(full_w, full_h)
        full_resolver.resolve(
            rotate=rotate_val,
            crop=crop_val,
            flip_h=flip_h,
            flip_v=flip_v,
            expand=True,
        )
        new_full_w, new_full_h = full_resolver.get_output_size()

        # --- Part 1: Global Background ---
        # Optimization: Cap background resolution when zoomed in to keep interaction fast
        bg_limit_w = (
            min(target_on_screen_width, 1500)
            if is_zoomed_in
            else target_on_screen_width
        )

        available_scales = sorted(self.tiers.keys()) + [1.0]
        selected_scale = 1.0
        selected_img = self.base_img_full

        for scale in available_scales:
            tier_img = self.tiers.get(scale) if scale < 1.0 else self.base_img_full
            if tier_img is not None and tier_img.shape[1] >= bg_limit_w:
                selected_scale = scale
                selected_img = tier_img
                break

        res_key = f"tier_{selected_scale}"
        h_src, w_src = selected_img.shape[:2]

        preprocess_key = (
            preprocess_settings["temperature"],
            preprocess_settings["tint"],
            preprocess_settings["exposure"],
            full_vig_params[0],
            full_vig_params[1],
            full_vig_params[2],
        )

        cached_bg, cached_w, cached_h = (None, 0, 0)
        if self.cache:
            cached_bg, cached_w, cached_h = self.cache.get_cached_bg_pixmap(
                preprocess_key
            )

        if (
            is_zoomed_in
            and cached_bg is not None
            and cached_w == new_full_w
            and cached_h == new_full_h
        ):
            pix_bg = cached_bg
        else:
            # Render background (potentially low-res if zoomed in)
            if is_zoomed_in:
                vig_k1, vig_k2, vig_k3, vig_cx, vig_cy, vig_fw, vig_fh = full_vig_params
                processed_bg = pynegative.apply_preprocess(
                    selected_img,
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

                img_dest = self._apply_fused_remap(
                    processed_bg, fused_maps, o_w, o_h, cv2.INTER_LINEAR
                )

                bg_output, _ = pynegative.apply_tone_map(
                    img_dest, **tone_map_settings, calculate_stats=False
                )
                bg_output = pynegative.apply_defringe(bg_output, self.settings)
            else:
                vig_k1, vig_k2, vig_k3, vig_cx, vig_cy, vig_fw, vig_fh = full_vig_params
                preprocess_key = (
                    preprocess_settings["temperature"],
                    preprocess_settings["tint"],
                    preprocess_settings["exposure"],
                    vig_k1,
                    vig_k2,
                    vig_k3,
                )
                preprocessed_bg = pynegative.apply_preprocess(
                    selected_img,
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

                denoised_bg = self._process_denoise_stage(
                    preprocessed_bg, res_key, heavy_params, zoom_scale, preprocess_key
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

                img_dest = self._apply_fused_remap(
                    denoised_bg, fused_maps, o_w, o_h, cv2.INTER_CUBIC
                )

                processed_bg = self._process_heavy_stage(
                    img_dest, res_key, heavy_params, zoom_scale
                )

                bg_output, _ = pynegative.apply_tone_map(
                    processed_bg, **tone_map_settings, calculate_stats=False
                )
                bg_output = pynegative.apply_defringe(bg_output, self.settings)

            img_uint8 = (np.clip(bg_output, 0, 1) * 255).astype(np.uint8)

            if self.calculate_histogram:
                try:
                    hist_data = self._calculate_histograms(img_uint8)
                    self.signals.histogramUpdated.emit(hist_data, self.request_id)
                except Exception as e:
                    logger.error(f"Histogram error: {e}")

            h_bg, w_bg = img_uint8.shape[:2]
            qimg_bg = QtGui.QImage(
                img_uint8.data, w_bg, h_bg, 3 * w_bg, QtGui.QImage.Format_RGB888
            )
            pix_bg = QtGui.QPixmap.fromImage(qimg_bg)
            if self.cache:
                self.cache.set_cached_bg_pixmap(
                    pix_bg, new_full_w, new_full_h, preprocess_key
                )

        # --- Part 2: Detail ROI ---
        pix_roi, roi_x, roi_y, roi_w, roi_h = QtGui.QPixmap(), 0, 0, 0, 0

        if is_zoomed_in:
            roi_scene = self._view_ref.mapToScene(
                self._view_ref.viewport().rect()
            ).boundingRect()
            rx, ry, rw, rh = (
                roi_scene.x(),
                roi_scene.y(),
                roi_scene.width(),
                roi_scene.height(),
            )

            # ROI PADDING
            pad_ratio = 0.5 if self.expand_roi else 0.05
            p_w, p_h = rw * pad_ratio, rh * pad_ratio
            rx, ry, rw, rh = rx - p_w, ry - p_h, rw + 2 * p_w, rh + 2 * p_h

            # Clamp and round to ensure consistent integer dimensions
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(new_full_w - rx, rw), min(new_full_h - ry, rh)
            # Round early to ensure consistent dimensions throughout processing
            rx, ry = int(round(rx)), int(round(ry))
            rw, rh = int(round(rw)), int(round(rh))

            if rw > 10 and rh > 10:
                req_display_w = full_w * zoom_scale
                res_key_roi, base_roi_img, ts_roi = "full", self.base_img_full, 1.0

                for scale in available_scales:
                    tier_img = (
                        self.tiers.get(scale) if scale < 1.0 else self.base_img_full
                    )
                    if tier_img is not None and tier_img.shape[1] >= req_display_w:
                        res_key_roi, base_roi_img, ts_roi = (
                            f"tier_{scale}",
                            tier_img,
                            scale,
                        )
                        break

                M_inv = full_resolver.get_inverse_matrix()
                corners = np.array(
                    [
                        [rx, ry, 1],
                        [rx + rw, ry, 1],
                        [rx + rw, ry + rh, 1],
                        [rx, ry + rh, 1],
                    ],
                    dtype=np.float32,
                )
                src_corners = (M_inv @ corners.T).T

                s_xmin, s_ymin = np.floor(
                    np.min(src_corners[:, :2], axis=0) * ts_roi
                ).astype(int)
                s_xmax, s_ymax = np.ceil(
                    np.max(src_corners[:, :2], axis=0) * ts_roi
                ).astype(int)

                h_rs, w_rs = base_roi_img.shape[:2]

                # Calculate expanded bounds for spatial caching (3x visible area)
                # This enables smooth panning within the cached region
                spatial_pad_x = (
                    s_xmax - s_xmin
                ) * 2  # 2x additional padding on each side
                spatial_pad_y = (s_ymax - s_ymin) * 2
                cache_xmin = max(0, s_xmin - spatial_pad_x)
                cache_ymin = max(0, s_ymin - spatial_pad_y)
                cache_xmax = min(w_rs, s_xmax + spatial_pad_x)
                cache_ymax = min(h_rs, s_ymax + spatial_pad_y)

                # Check spatial cache for denoised data
                requested_denoise_rect = (s_xmin, s_ymin, s_xmax, s_ymax)
                denoise_spatial_key = {
                    "denoise_luma": heavy_params["denoise_luma"],
                    "denoise_chroma": heavy_params["denoise_chroma"],
                    "denoise_method": heavy_params["denoise_method"],
                }

                denoised_chunk = None
                if self.cache:
                    denoised_chunk = self.cache.get_spatial_roi(
                        f"{res_key_roi}_denoise",
                        requested_denoise_rect,
                        denoise_spatial_key,
                    )
                    if denoised_chunk is not None:
                        logger.debug(
                            f"Spatial denoise cache HIT for tier {res_key_roi}"
                        )

                # Use expanded bounds for extraction to enable spatial caching
                extract_xmin, extract_ymin = cache_xmin, cache_ymin
                extract_xmax, extract_ymax = cache_xmax, cache_ymax

                # Add small padding for processing
                pad_src = 16
                extract_xmin = max(0, extract_xmin - pad_src)
                extract_ymin = max(0, extract_ymin - pad_src)
                extract_xmax = min(w_rs, extract_xmax + pad_src)
                extract_ymax = min(h_rs, extract_ymax + pad_src)

                # Adjust coordinates for the extracted chunk
                s_xmin_adj = s_xmin - extract_xmin
                s_ymin_adj = s_ymin - extract_ymin
                s_xmax_adj = s_xmax - extract_xmin
                s_ymax_adj = s_ymax - extract_ymin

                raw_chunk = base_roi_img[
                    extract_ymin:extract_ymax, extract_xmin:extract_xmax
                ]
                if raw_chunk.size > 0:
                    roi_res_id = (res_key_roi, s_xmin, s_ymin, s_xmax, s_ymax)

                    roi_vig_params = self._resolve_vignette_params(
                        roi_offset=(s_xmin, s_ymin), full_size=(w_rs, h_rs)
                    )
                    vig_k1, vig_k2, vig_k3, vig_cx, vig_cy, vig_fw, vig_fh = (
                        roi_vig_params
                    )

                    preprocess_p = {
                        "temperature": preprocess_settings["temperature"],
                        "tint": preprocess_settings["tint"],
                        "exposure": preprocess_settings["exposure"],
                        "vig_k1": vig_k1,
                        "vig_k2": vig_k2,
                        "vig_k3": vig_k3,
                    }
                    denoise_p = {
                        "denoise_luma": heavy_params["denoise_luma"],
                        "denoise_chroma": heavy_params["denoise_chroma"],
                        "denoise_method": heavy_params["denoise_method"],
                        "_preprocess_key": (
                            preprocess_settings["temperature"],
                            preprocess_settings["tint"],
                            preprocess_settings["exposure"],
                            vig_k1,
                            vig_k2,
                            vig_k3,
                        ),
                    }

                    preprocessed_chunk = (
                        self.cache.get(res_key_roi, "preprocess_chunk", preprocess_p)
                        if self.cache
                        else None
                    )

                    if preprocessed_chunk is None:
                        preprocessed_chunk = pynegative.apply_preprocess(
                            raw_chunk,
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
                        if self.cache:
                            self.cache.put(
                                res_key_roi,
                                "preprocess_chunk",
                                preprocess_p,
                                preprocessed_chunk,
                            )

                    # Only process denoise if we don't have spatial cache hit
                    if denoised_chunk is None:
                        denoised_full_chunk = (
                            self.cache.get(res_key_roi, "denoise_chunk", denoise_p)
                            if self.cache
                            else None
                        )

                        if denoised_full_chunk is None:
                            denoised_full_chunk = self._process_denoise_stage(
                                preprocessed_chunk, roi_res_id, heavy_params, zoom_scale
                            )
                            if self.cache:
                                self.cache.put(
                                    res_key_roi,
                                    "denoise_chunk",
                                    denoise_p,
                                    denoised_full_chunk,
                                )
                                # Also store in spatial cache for panning
                                cache_rect = (
                                    extract_xmin,
                                    extract_ymin,
                                    extract_xmax,
                                    extract_ymax,
                                )
                                self.cache.put_spatial_roi(
                                    f"{res_key_roi}_denoise",
                                    cache_rect,
                                    denoise_spatial_key,
                                    denoised_full_chunk,
                                )

                        # Extract the region we need from the full processed chunk
                        denoised_chunk = denoised_full_chunk[
                            s_ymin_adj:s_ymax_adj, s_xmin_adj:s_xmax_adj
                        ]

                    # 3. Mega-Remap: Lens + Affine in one go
                    h_chunk, w_chunk = denoised_chunk.shape[:2]

                    # Calculate M_local for this chunk
                    resolver_roi = GeometryResolver(w_rs, h_rs)
                    resolver_roi.resolve(
                        rotate=rotate_val,
                        crop=crop_val,
                        flip_h=flip_h,
                        flip_v=flip_v,
                        expand=True,
                    )
                    M_roi_full_tier = resolver_roi.get_matrix_2x3()

                    M_local = M_roi_full_tier.copy()
                    # Scale the matrix first to work in tier-scale coordinates
                    M_local[:2, :] /= ts_roi
                    # Apply translation using the scaled matrix with adjusted chunk coordinates
                    M_local[0, 2] += M_local[0, 0] * (
                        s_xmin_adj + extract_xmin
                    ) + M_local[0, 1] * (s_ymin_adj + extract_ymin)
                    M_local[1, 2] += M_local[1, 0] * (
                        s_xmin_adj + extract_xmin
                    ) + M_local[1, 1] * (s_ymin_adj + extract_ymin)
                    # Adjust for ROI offset in destination
                    M_local[0, 2] -= rx
                    M_local[1, 2] -= ry

                    # Use pre-rounded dimensions
                    fused_maps, o_w, o_h, _ = self._get_fused_geometry(
                        w_chunk,
                        h_chunk,
                        rotate_val,
                        crop_val,
                        flip_h,
                        flip_v,
                        ts_roi=ts_roi,
                        roi_offset=(s_xmin, s_ymin),
                        full_size=(w_rs, h_rs),
                        M_override=M_local,
                        out_size_override=(rw, rh),
                    )

                    remapped_roi = self._apply_fused_remap(
                        denoised_chunk, fused_maps, rw, rh, cv2.INTER_CUBIC
                    )

                    chunk_processed = self._process_heavy_stage(
                        remapped_roi, "roi_final", heavy_params, zoom_scale
                    )

                    roi_output, _ = pynegative.apply_tone_map(
                        chunk_processed, **tone_map_settings, calculate_stats=False
                    )
                    roi_output = pynegative.apply_defringe(roi_output, self.settings)

                    roi_uint8 = (np.clip(roi_output, 0, 1) * 255).astype(np.uint8)
                    qimg_roi = QtGui.QImage(
                        roi_uint8.data, rw, rh, 3 * rw, QtGui.QImage.Format_RGB888
                    )
                    pix_roi = QtGui.QPixmap.fromImage(qimg_roi)
                    roi_x, roi_y, roi_w, roi_h = rx, ry, rw, rh

        return (
            pix_bg,
            new_full_w,
            new_full_h,
            pix_roi,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            rotate_val,
        )

    def _calculate_histograms(self, img_array):
        start_time = time.perf_counter()
        h, w = img_array.shape[:2]
        stride = max(1, int(np.sqrt((h * w) / 65536)))
        if img_array.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
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
