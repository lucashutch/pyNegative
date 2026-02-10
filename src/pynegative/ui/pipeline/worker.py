import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui
import cv2
from ... import core as pynegative


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(
        QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int, int
    )
    histogramUpdated = QtCore.Signal(dict, int)
    error = QtCore.Signal(str, int)


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

    def run(self):
        try:
            result = self._update_preview()
            self.signals.finished.emit(*result, self.request_id)
        except Exception as e:
            self.signals.error.emit(str(e), self.request_id)

    def _process_heavy_stage(self, img, res_key, heavy_params, zoom_scale):
        """Processes and caches the heavy effects stage for a full image tier."""

        # 1. Group parameters by effect
        dehaze_p = {"de_haze": heavy_params["de_haze"]}
        denoise_p = {
            "denoise_luma": heavy_params.get("denoise_luma", 0),
            "denoise_chroma": heavy_params.get("denoise_chroma", 0),
            "denoise_method": heavy_params.get(
                "denoise_method", "NLMeans (Numba Fast+)"
            ),
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
            pil_bg = Image.fromarray(img_uint8)
            preview_h, preview_w = self.base_img_preview.shape[:2]
            scale_x = full_w / preview_w
            scale_y = full_h / preview_h
            new_full_w = int(pil_bg.width * scale_x)
            new_full_h = int(pil_bg.height * scale_y)

            if self.calculate_histogram:
                try:
                    hist_data = self._calculate_histograms(img_uint8)
                    self.signals.histogramUpdated.emit(hist_data, self.request_id)
                except Exception as e:
                    print(f"Histogram calculation error: {e}")

            pix_bg = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_bg))
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

            if (req_w := src_x2 - src_x) > 10 and (req_h := src_y2 - src_y) > 10:
                roi_area = req_w * req_h
                full_area = full_w * full_h
                if roi_area / full_area > 0.85:
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

                preview_w_res = self.base_img_preview.shape[1]
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
                pil_roi = Image.fromarray(
                    (np.clip(processed_roi, 0, 1) * 255).astype(np.uint8)
                )
                pix_roi = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_roi))
                roi_x, roi_y = src_x - offset_x, src_y - offset_y
                roi_w, roi_h = req_w, req_h

        return pix_bg, new_full_w, new_full_h, pix_roi, roi_x, roi_y, roi_w, roi_h

    def _calculate_histograms(self, img_array):
        bins = 256
        h, w = img_array.shape[:2]
        if max(h, w) > 512:
            scale = 256 / max(h, w)
            small_img = cv2.resize(
                img_array,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            small_img = img_array

        if small_img.shape[2] == 4:
            small_img = cv2.cvtColor(small_img, cv2.COLOR_RGBA2RGB)

        hist_r = cv2.calcHist([small_img], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([small_img], [1], None, [bins], [0, 256]).flatten()
        hist_b = cv2.calcHist([small_img], [2], None, [bins], [0, 256]).flatten()

        img_yuv = cv2.cvtColor(small_img, cv2.COLOR_RGB2YUV)
        hist_y = cv2.calcHist([img_yuv], [0], None, [bins], [0, 256]).flatten()
        hist_u = cv2.calcHist([img_yuv], [1], None, [bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([img_yuv], [2], None, [bins], [0, 256]).flatten()

        def smooth(h):
            return cv2.GaussianBlur(h.reshape(-1, 1), (5, 5), 0).flatten()

        return {
            "R": smooth(hist_r),
            "G": smooth(hist_g),
            "B": smooth(hist_b),
            "Y": smooth(hist_y),
            "U": smooth(hist_u),
            "V": smooth(hist_v),
        }
