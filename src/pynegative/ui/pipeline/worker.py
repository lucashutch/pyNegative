import numpy as np
from PySide6 import QtCore, QtGui
import cv2
import time
import logging
from ... import core as pynegative

logger = logging.getLogger(__name__)


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(
        QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int, int
    )
    histogramUpdated = QtCore.Signal(dict, int)
    tiersGenerated = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray)
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

            # 1. Preview (High Priority for UI)
            scale = 2048 / max(h, w)
            target_h, target_w = int(h * scale), int(w * scale)
            preview = cv2.resize(
                self.img_array, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

            # Generate unedited pixmap from preview for fast initial display
            img_uint8 = (np.clip(preview, 0, 1) * 255).astype(np.uint8)
            h_p, w_p, c_p = img_uint8.shape
            qimage = QtGui.QImage(
                img_uint8.data, w_p, h_p, c_p * w_p, QtGui.QImage.Format_RGB888
            )
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.signals.uneditedPixmapGenerated.emit(pixmap)

            # 2. Half
            half = cv2.resize(
                self.img_array, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR
            )

            # 3. Quarter
            quarter = cv2.resize(half, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)

            self.signals.tiersGenerated.emit(half, quarter, preview)

        except Exception as e:
            print(f"Tier generation error: {e}")


class ImageProcessorWorker(QtCore.QRunnable):
    """Worker to process a single large ROI in a background thread."""

    def __init__(
        self,
        signals,
        view_ref,
        base_img_full,
        base_img_half,
        base_img_quarter,
        base_img_preview,
        settings,
        request_id,
        calculate_histogram=False,
        cache=None,
        last_heavy_adjusted="de_haze",
        expand_roi=False,
    ):
        super().__init__()
        self.signals = signals
        self._view_ref = view_ref
        self.base_img_full = base_img_full
        self.base_img_half = base_img_half
        self.base_img_quarter = base_img_quarter
        self.base_img_preview = base_img_preview
        self.settings = settings
        self.request_id = request_id
        self.calculate_histogram = calculate_histogram
        self.cache = cache
        self.last_heavy_adjusted = last_heavy_adjusted
        self.expand_roi = expand_roi

    def run(self):
        try:
            result = self._update_preview()
            self.signals.finished.emit(*result, self.request_id)
        except Exception as e:
            self.signals.error.emit(str(e), self.request_id)

    def _process_heavy_stage(self, img, res_key, heavy_params, zoom_scale):
        """Processes and caches the heavy effects stage for a full image tier."""

        # 1. Determine Effective Denoise Method based on Image Tier
        requested_method = heavy_params.get("denoise_method", "NLMeans (Numba Fast+)")
        effective_method = requested_method

        # Only override NLMeans methods (Bilateral is already fast)
        if "NLMeans" in requested_method:
            is_roi = isinstance(res_key, tuple)

            if is_roi:
                # True Quality for ROI (Zoomed-in Detail)
                effective_method = requested_method
            elif res_key in ["preview", "quarter"]:
                effective_method = "NLMeans (Numba Ultra Fast YUV)"
            elif res_key == "half":
                effective_method = "NLMeans (Numba Fast+ YUV)"
            elif res_key == "full":
                # Cap Background at Hybrid
                if "High Quality" in requested_method or "Hybrid" in requested_method:
                    effective_method = "NLMeans (Numba Hybrid YUV)"
                else:
                    effective_method = requested_method

        if effective_method != requested_method:
            logger.debug(
                f"Tier-aware denoise override: {requested_method} -> {effective_method} (tier: {res_key})"
            )

        # 2. Group parameters by effect

        dehaze_p = {"de_haze": heavy_params["de_haze"]}
        denoise_p = {
            "denoise_luma": heavy_params.get("denoise_luma", 0),
            "denoise_chroma": heavy_params.get("denoise_chroma", 0),
            "denoise_method": effective_method,
        }
        sharpen_p = {
            "sharpen_value": heavy_params["sharpen_value"],
            "sharpen_radius": heavy_params["sharpen_radius"],
            "sharpen_percent": heavy_params["sharpen_percent"],
        }

        # 2. Define application functions
        def apply_dehaze(image):
            if dehaze_p["de_haze"] <= 0:
                return image
            # Always sync atmospheric light from preview if possible
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
            # If we are processing preview, store the estimated light for other tiers
            if res_key == "preview" and self.cache and atmos_fixed is None:
                self.cache.estimated_params["atmospheric_light"] = atmos
            return processed

        def apply_denoise(image):
            if denoise_p["denoise_luma"] <= 0 and denoise_p["denoise_chroma"] <= 0:
                return image
            return pynegative.de_noise_image(
                image,
                luma_strength=denoise_p["denoise_luma"],
                chroma_strength=denoise_p["denoise_chroma"],
                method=denoise_p["denoise_method"],
                zoom=zoom_scale,
            )

        def apply_sharpen(image):
            if sharpen_p["sharpen_value"] <= 0:
                return image
            return pynegative.sharpen_image(
                image,
                sharpen_p["sharpen_radius"],
                sharpen_p["sharpen_percent"],
                "High Quality",
            )

        # 3. Determine execution order based on the last adjusted parameter.
        active = self.last_heavy_adjusted
        if active == "de_haze":
            pipeline = [
                ("denoise", denoise_p, apply_denoise),
                ("sharpen", sharpen_p, apply_sharpen),
                ("dehaze", dehaze_p, apply_dehaze),
            ]
        elif active in ["denoise_luma", "denoise_chroma"]:
            pipeline = [
                ("dehaze", dehaze_p, apply_dehaze),
                ("sharpen", sharpen_p, apply_sharpen),
                ("denoise", denoise_p, apply_denoise),
            ]
        else:
            pipeline = [
                ("dehaze", dehaze_p, apply_dehaze),
                ("denoise", denoise_p, apply_denoise),
                ("sharpen", sharpen_p, apply_sharpen),
            ]

        # 4. Execute pipeline with multi-stage caching
        processed = img
        accumulated_params = {}
        for i, (name, params, func) in enumerate(pipeline):
            accumulated_params.update(params)
            stage_id = f"heavy_stage_{i + 1}_{name}"

            if self.cache:
                cached = self.cache.get(res_key, stage_id, accumulated_params)
                if cached is not None:
                    processed = cached
                    continue

            # Cache miss: compute this stage
            processed = func(processed)

            # Store in cache
            if self.cache:
                self.cache.put(res_key, stage_id, accumulated_params, processed)

        return processed

    def _update_preview(self):
        if self.base_img_full is None or self._view_ref is None:
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0

        full_h, full_w, _ = self.base_img_full.shape

        try:
            zoom_scale = self._view_ref.transform().m11()
        except (AttributeError, RuntimeError):
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0

        preview_scale = 2048 / max(full_w, full_h)
        is_fitting = getattr(self._view_ref, "_is_fitting", False)
        rotate_val = self.settings.get("rotation", 0.0)
        flip_h = self.settings.get("flip_h", False)
        flip_v = self.settings.get("flip_v", False)
        crop_val = self.settings.get("crop", None)

        is_zoomed_in = not is_fitting and (zoom_scale > preview_scale * 1.1)

        # --- Part 1: Global Background ---
        res_key = "preview"
        img_render_base = self.base_img_preview

        # If tiers aren't ready, we MUST NOT process the full image for a "preview"
        # as it will take seconds to denoise/dehaze.
        if img_render_base is None:
            # Quick downsample for immediate responsiveness
            h_f, w_f = self.base_img_full.shape[:2]
            scale = 2048 / max(h_f, w_f)
            img_render_base = cv2.resize(
                self.base_img_full,
                (int(w_f * scale), int(h_f * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            # res_key is "preview" anyway

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

        cached_bg, cached_w, cached_h = (None, 0, 0)
        if self.cache:
            cached_bg, cached_w, cached_h = self.cache.get_cached_bg_pixmap()

        if is_zoomed_in and cached_bg is not None:
            pix_bg = cached_bg
            new_full_w = cached_w
            new_full_h = cached_h
        else:
            processed_bg = self._process_heavy_stage(
                img_render_base, res_key, heavy_params, zoom_scale
            )
            bg_output, _ = pynegative.apply_tone_map(
                processed_bg, **tone_map_settings, calculate_stats=False
            )
            bg_output = pynegative.apply_geometry(
                bg_output,
                rotate=rotate_val,
                crop=crop_val,
                flip_h=flip_h,
                flip_v=flip_v,
            )
            img_uint8 = (np.clip(bg_output, 0, 1) * 255).astype(np.uint8)

            if self.base_img_preview is not None:
                preview_h_orig, preview_w_orig = self.base_img_preview.shape[:2]
            else:
                # Fallback if tiers not ready yet
                preview_h_orig, preview_w_orig = img_uint8.shape[:2]

            scale_x = full_w / preview_w_orig
            scale_y = full_h / preview_h_orig

            # Using uint8 shape directly
            h_bg, w_bg, c_bg = img_uint8.shape
            new_full_w = int(w_bg * scale_x)
            new_full_h = int(h_bg * scale_y)

            if self.calculate_histogram:
                try:
                    hist_data = self._calculate_histograms(img_uint8)
                    self.signals.histogramUpdated.emit(hist_data, self.request_id)
                except Exception as e:
                    print(f"Histogram calculation error: {e}")

            # Direct NumPy to QImage conversion
            qimg_bg = QtGui.QImage(
                img_uint8.data, w_bg, h_bg, c_bg * w_bg, QtGui.QImage.Format_RGB888
            )
            pix_bg = QtGui.QPixmap.fromImage(qimg_bg)
            if self.cache:
                self.cache.set_cached_bg_pixmap(pix_bg, new_full_w, new_full_h)

        # --- Part 2: Detail ROI ---
        pix_roi, roi_x, roi_y, roi_w, roi_h = QtGui.QPixmap(), 0, 0, 0, 0

        if is_zoomed_in:
            roi = self._view_ref.mapToScene(
                self._view_ref.viewport().rect()
            ).boundingRect()
            v_x, v_y, v_w, v_h = roi.x(), roi.y(), roi.width(), roi.height()
            offset_x, offset_y = 0, 0
            if crop_val:
                offset_x = int(crop_val[0] * full_w)
                offset_y = int(crop_val[1] * full_h)

            src_x, src_y = int(v_x + offset_x), int(v_y + offset_y)
            src_w, src_h = int(v_w), int(v_h)

            if flip_h:
                src_x = full_w - (src_x + src_w)
            if flip_v:
                src_y = full_h - (src_y + src_h)

            src_x, src_y = max(0, src_x), max(0, src_y)
            src_x2, src_y2 = min(full_w, src_x + src_w), min(full_h, src_y + src_h)

            # --- ROI PADDING LOGIC ---
            # If idle (expand_roi=True), we request a much larger area to allow smooth panning
            pad_ratio = 0.5 if self.expand_roi else 0.05
            p_w, p_h = src_x2 - src_x, src_y2 - src_y
            pad_w, pad_h = int(p_w * pad_ratio), int(p_h * pad_ratio)

            # Apply padding but stay within image bounds
            src_x = max(0, src_x - pad_w)
            src_y = max(0, src_y - pad_h)
            src_x2 = min(full_w, src_x2 + pad_w)
            src_y2 = min(full_h, src_y2 + pad_h)
            # Re-calculate required width/height after padding
            req_w, req_h = src_x2 - src_x, src_y2 - src_y

            if req_w > 10 and req_h > 10:
                roi_area = req_w * req_h
                full_area = full_w * full_h
                # If we're rendering almost the whole image, just stop
                if roi_area / full_area > 0.95 and not self.expand_roi:
                    return (
                        pix_bg,
                        new_full_w,
                        new_full_h,
                        pix_roi,
                        roi_x,
                        roi_y,
                        roi_w,
                        roi_h,
                    )

                preview_w_res = (
                    self.base_img_preview.shape[1]
                    if self.base_img_preview is not None
                    else 2048
                )
                res_key_roi = "full"
                base_roi_img = self.base_img_full

                if self.base_img_half is not None:
                    h_w = self.base_img_half.shape[1]
                    if h_w > preview_w_res and zoom_scale < 1.5:
                        res_key_roi = "half"
                        base_roi_img = self.base_img_half

                if self.base_img_quarter is not None:
                    q_w = self.base_img_quarter.shape[1]
                    if q_w > preview_w_res and zoom_scale < 0.5:
                        res_key_roi = "quarter"
                        base_roi_img = self.base_img_quarter

                if base_roi_img.shape[1] <= preview_w_res:
                    return (
                        pix_bg,
                        new_full_w,
                        new_full_h,
                        pix_roi,
                        roi_x,
                        roi_y,
                        roi_w,
                        roi_h,
                    )

                h_tier, w_tier = base_roi_img.shape[:2]
                s_x = int(src_x * (w_tier / full_w))
                s_y = int(src_y * (h_tier / full_h))
                s_x2 = int(src_x2 * (w_tier / full_w))
                s_y2 = int(src_y2 * (h_tier / full_h))

                pad = 16
                p_x1, p_y1 = max(0, s_x - pad), max(0, s_y - pad)
                p_x2, p_y2 = min(w_tier, s_x2 + pad), min(h_tier, s_y2 + pad)

                raw_chunk = base_roi_img[p_y1:p_y2, p_x1:p_x2]
                roi_res_key = (res_key_roi, s_x, s_y, s_x2, s_y2)
                processed_chunk_padded = self._process_heavy_stage(
                    raw_chunk, roi_res_key, heavy_params, zoom_scale
                )

                c_y1, c_x1 = s_y - p_y1, s_x - p_x1
                c_y2, c_x2 = c_y1 + (s_y2 - s_y), c_x1 + (s_x2 - s_x)
                crop_chunk = processed_chunk_padded[c_y1:c_y2, c_x1:c_x2]

                if flip_h or flip_v:
                    flip_code = -1 if (flip_h and flip_v) else (1 if flip_h else 0)
                    crop_chunk = cv2.flip(crop_chunk, flip_code)

                processed_roi, _ = pynegative.apply_tone_map(
                    crop_chunk, **tone_map_settings, calculate_stats=False
                )
                roi_uint8 = (np.clip(processed_roi, 0, 1) * 255).astype(np.uint8)
                h_r, w_r, c_r = roi_uint8.shape
                qimg_roi = QtGui.QImage(
                    roi_uint8.data, w_r, h_r, c_r * w_r, QtGui.QImage.Format_RGB888
                )
                pix_roi = QtGui.QPixmap.fromImage(qimg_roi)
                roi_x, roi_y = src_x - offset_x, src_y - offset_y
                roi_w, roi_h = req_w, req_h

        return pix_bg, new_full_w, new_full_h, pix_roi, roi_x, roi_y, roi_w, roi_h

    def _calculate_histograms(self, img_array):
        # Use strided Numba kernel for maximum speed
        # Stride based on image size to keep samples roughly constant (~65k samples)
        start_time = time.perf_counter()
        h, w = img_array.shape[:2]
        area = h * w
        stride = max(1, int(np.sqrt(area / 65536)))

        # Ensure RGB (OpenCV might provide RGBA in some cases, though pipeline uses RGB)
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
        logger.debug(
            f"Histogram calculation: {elapsed:.2f}ms (stride: {stride}, size: {w}x{h})"
        )
        return result
