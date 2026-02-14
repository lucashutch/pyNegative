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

    def _process_lens_stage(self, img, res_key, scale, roi_offset=None, full_size=None):
        """Processes lens correction stage."""
        if not self.settings.get("lens_enabled", True):
            return img

        k1 = self.settings.get("lens_distortion", 0.0)
        vignette = self.settings.get("lens_vignette", 0.0)

        has_auto_dist = False
        has_auto_vig = False
        if self.lens_info:
            has_auto_dist = (
                "distortion" in self.lens_info
                and self.lens_info["distortion"] is not None
            )
            has_auto_vig = (
                "vignetting" in self.lens_info
                and self.lens_info["vignetting"] is not None
            )

        if (
            abs(k1) < 1e-5
            and abs(vignette) < 1e-5
            and not has_auto_dist
            and not has_auto_vig
        ):
            return img

        t0 = time.perf_counter()
        result = pynegative.apply_lens_correction(
            img, self.settings, self.lens_info, scale, roi_offset, full_size
        )
        elapsed = (time.perf_counter() - t0) * 1000

        logger.debug(
            f"Lens Correction ({res_key}): k1={k1:.4f}, vig={vignette:.4f} ({elapsed:.2f}ms)"
        )
        return result

    def _process_denoise_stage(self, img, res_key, heavy_params, zoom_scale):
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

        tone_map_settings = {
            "temperature": self.settings.get("temperature", 0.0),
            "tint": self.settings.get("tint", 0.0),
            "exposure": self.settings.get("exposure", 0.0),
            "contrast": self.settings.get("contrast", 1.0),
            "blacks": self.settings.get("blacks", 0.0),
            "whites": self.settings.get("whites", 1.0),
            "shadows": self.settings.get("shadows", 0.0),
            "highlights": self.settings.get("highlights", 0.0),
            "saturation": self.settings.get("saturation", 1.0),
        }

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

        tier_resolver = GeometryResolver(w_src, h_src)
        tier_resolver.resolve(
            rotate=rotate_val,
            crop=crop_val,
            flip_h=flip_h,
            flip_v=flip_v,
            expand=True,
        )
        M_tier = tier_resolver.get_matrix_2x3()
        out_w_tier, out_h_tier = tier_resolver.get_output_size()

        cached_bg, cached_w, cached_h = (None, 0, 0)
        if self.cache:
            cached_bg, cached_w, cached_h = self.cache.get_cached_bg_pixmap()

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
                # OPTIMIZATION: Skip heavy effects for background when zoomed in.
                # The background is just context/blur, so we don't need expensive effects.
                processed_bg = selected_img
                # Still need lens correction for geometry alignment
                corrected_bg = self._process_lens_stage(
                    processed_bg, res_key, selected_scale, (0, 0), (w_src, h_src)
                )
                bg_output, _ = pynegative.apply_tone_map(
                    corrected_bg, **tone_map_settings, calculate_stats=False
                )
                bg_output = pynegative.apply_defringe(bg_output, self.settings)
            else:
                denoised_bg = self._process_denoise_stage(
                    selected_img, res_key, heavy_params, zoom_scale
                )
                corrected_bg = self._process_lens_stage(
                    denoised_bg, res_key, selected_scale, (0, 0), (w_src, h_src)
                )
                processed_bg = self._process_heavy_stage(
                    corrected_bg, res_key, heavy_params, zoom_scale
                )
                bg_output, _ = pynegative.apply_tone_map(
                    processed_bg, **tone_map_settings, calculate_stats=False
                )
                bg_output = pynegative.apply_defringe(bg_output, self.settings)

            img_dest = cv2.warpAffine(
                bg_output,
                M_tier,
                (int(round(out_w_tier)), int(round(out_h_tier))),
                flags=cv2.INTER_LINEAR,
            )
            img_uint8 = (np.clip(img_dest, 0, 1) * 255).astype(np.uint8)

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
                self.cache.set_cached_bg_pixmap(pix_bg, new_full_w, new_full_h)

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

            # Clamp
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(new_full_w - rx, rw), min(new_full_h - ry, rh)

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
                pad_src = 16
                s_xmin, s_ymin = max(0, s_xmin - pad_src), max(0, s_ymin - pad_src)
                s_xmax, s_ymax = (
                    min(w_rs, s_xmax + pad_src),
                    min(h_rs, s_ymax + pad_src),
                )

                raw_chunk = base_roi_img[s_ymin:s_ymax, s_xmin:s_xmax]
                if raw_chunk.size > 0:
                    roi_res_id = (res_key_roi, s_xmin, s_ymin, s_xmax, s_ymax)
                    chunk_processed = (
                        self.cache.get_spatial_roi(
                            res_key_roi, (s_xmin, s_ymin, s_xmax, s_ymax), heavy_params
                        )
                        if self.cache
                        else None
                    )

                    if chunk_processed is None:
                        c = self._process_denoise_stage(
                            raw_chunk, roi_res_id, heavy_params, zoom_scale
                        )
                        c = self._process_lens_stage(
                            c, res_key_roi, ts_roi, (s_xmin, s_ymin), (w_rs, h_rs)
                        )
                        chunk_processed = self._process_heavy_stage(
                            c, roi_res_id, heavy_params, zoom_scale
                        )
                        if self.cache:
                            self.cache.put_spatial_roi(
                                res_key_roi,
                                (s_xmin, s_ymin, s_xmax, s_ymax),
                                heavy_params,
                                chunk_processed,
                            )

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
                    M_local[0, 2] += (
                        M_roi_full_tier[0, 0] * s_xmin + M_roi_full_tier[0, 1] * s_ymin
                    )
                    M_local[1, 2] += (
                        M_roi_full_tier[1, 0] * s_xmin + M_roi_full_tier[1, 1] * s_ymin
                    )
                    M_local[:2, :] /= ts_roi
                    M_local[0, 2] -= rx
                    M_local[1, 2] -= ry

                    rw_i, rh_i = int(round(rw)), int(round(rh))
                    roi_output, _ = pynegative.apply_tone_map(
                        chunk_processed, **tone_map_settings, calculate_stats=False
                    )
                    roi_output = pynegative.apply_defringe(roi_output, self.settings)
                    roi_dest = cv2.warpAffine(
                        roi_output, M_local, (rw_i, rh_i), flags=cv2.INTER_LINEAR
                    )

                    roi_uint8 = (np.clip(roi_dest, 0, 1) * 255).astype(np.uint8)
                    qimg_roi = QtGui.QImage(
                        roi_uint8.data, rw_i, rh_i, 3 * rw_i, QtGui.QImage.Format_RGB888
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
