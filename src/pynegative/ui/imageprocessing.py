import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui
import time
import cv2
from .. import core as pynegative


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(
        QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int, int
    )
    histogramUpdated = QtCore.Signal(dict, int)
    error = QtCore.Signal(str, int)


class PipelineCache:
    """Manages cached stages of the image processing pipeline."""

    def __init__(self):
        # caches[resolution_key][stage_id] = (parameters_dict, numpy_array)
        self.caches = {}
        # Effect parameters that are estimated once on the preview and synced
        self.estimated_params = {}
        # Cached background pixmap for ROI optimization
        self._cached_bg_pixmap = None
        self._cached_bg_full_w = 0
        self._cached_bg_full_h = 0

    def get(self, resolution, stage_id, current_params):
        """Returns the cached array if parameters match exactly."""
        res_cache = self.caches.get(resolution, {})
        cached_data = res_cache.get(stage_id)

        if cached_data:
            cached_params, cached_array = cached_data
            # Check if all relevant parameters for this stage match
            if all(
                current_params.get(k) == cached_params.get(k) for k in cached_params
            ):
                return cached_array
        return None

    def put(self, resolution, stage_id, params, array):
        """Stores a stage in the cache."""
        if resolution not in self.caches:
            self.caches[resolution] = {}
        self.caches[resolution][stage_id] = (params.copy(), array)

    def invalidate(self, stage_id=None):
        """Invalidates stages. If stage_id is None, invalidates everything."""
        if stage_id is None:
            self.caches = {}
            self.estimated_params = {}
            self._cached_bg_pixmap = None

    def clear(self):
        self.caches = {}
        self.estimated_params = {}
        self._cached_bg_pixmap = None

    def get_cached_bg_pixmap(self):
        """Returns cached background pixmap if it exists (no param check).

        When ROI is active, we use any cached background since it's not visible.
        """
        if self._cached_bg_pixmap is None:
            return None, 0, 0
        return self._cached_bg_pixmap, self._cached_bg_full_w, self._cached_bg_full_h

    def set_cached_bg_pixmap(self, pixmap, full_w, full_h):
        """Store the cached background pixmap."""
        self._cached_bg_pixmap = pixmap
        self._cached_bg_full_w = full_w
        self._cached_bg_full_h = full_h


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
        # The goal is to keep the "active" parameter at the end of the chain
        # so that earlier stages can be retrieved from cache.
        active = self.last_heavy_adjusted
        if active == "de_haze":
            # Adjusting Dehaze -> Denoise and Sharpen come first
            pipeline = [
                ("denoise", denoise_p, apply_denoise),
                ("sharpen", sharpen_p, apply_sharpen),
                ("dehaze", dehaze_p, apply_dehaze),
            ]
        elif active in ["denoise_luma", "denoise_chroma"]:
            # Adjusting Denoise -> Dehaze and Sharpen come first
            pipeline = [
                ("dehaze", dehaze_p, apply_dehaze),
                ("sharpen", sharpen_p, apply_sharpen),
                ("denoise", denoise_p, apply_denoise),
            ]
        else:
            # Adjusting Sharpen or other -> Default order
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
        crop_val = self.settings.get(
            "crop", None
        )  # (left, top, right, bottom) normalized

        # Only trigger ROI processing if we are actually zoomed in MORE than what the
        # background preview (2048px) can already provide with good quality.
        is_zoomed_in = not is_fitting and (zoom_scale > preview_scale * 1.1)

        # --- Part 1: Global Background ---
        # Resolution key for caching
        res_key = "preview"
        img_render_base = self.base_img_preview

        # Stage 1: Heavy Effects (Dehaze, Denoise, Sharpen)
        heavy_params = {
            # Normalize dehaze to 0-1 range (UI slider is 0-50)
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

        # Stage 2: Tone Mapping settings (used by both background and ROI)
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

        # OPTIMIZATION: When zoomed in and ROI will be rendered, skip ALL background
        # processing. The ROI overlay covers the visible area, so processing the
        # 2048px background is wasteful. We use any cached pixmap (even if stale).
        cached_bg, cached_w, cached_h = (None, 0, 0)
        if self.cache:
            cached_bg, cached_w, cached_h = self.cache.get_cached_bg_pixmap()

        if is_zoomed_in and cached_bg is not None:
            # Use cached background - skip all processing (background is hidden by ROI)
            pix_bg = cached_bg
            new_full_w = cached_w
            new_full_h = cached_h
        else:
            # Full background processing required
            # Use helper to get/calculate cached heavy background
            processed_bg = self._process_heavy_stage(
                img_render_base, res_key, heavy_params, zoom_scale
            )

            # Apply Tone Map to the result of heavy stage
            bg_output, _ = pynegative.apply_tone_map(
                processed_bg, **tone_map_settings, calculate_stats=False
            )

            # Apply Geometry (optimized OpenCV implementation)
            bg_output = pynegative.apply_geometry(
                bg_output,
                rotate=rotate_val,
                crop=crop_val,
                flip_h=flip_h,
                flip_v=flip_v,
            )

            # Convert to uint8 for display and histograms
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

            # Cache the background for use when ROI is active
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
                # Coverage check: If the ROI covers most of the image, the background
                # preview (2048px) is already good enough. Computing a second layer
                # at a similar resolution is wasteful.
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

                # ROI Resolution Selection
                # We only want to use an ROI tier if it offers MORE detail than
                # the background preview (2048px).
                preview_w_res = self.base_img_preview.shape[1]

                res_key_roi = "full"
                base_roi_img = self.base_img_full

                # Check if half-res is suitable and better than preview
                if self.base_img_half is not None:
                    h_w = self.base_img_half.shape[1]
                    if h_w > preview_w_res and zoom_scale < 1.5:
                        res_key_roi = "half"
                        base_roi_img = self.base_img_half

                # Check if quarter-res is suitable and better than preview
                # (Only likely for > 32MP images)
                if self.base_img_quarter is not None:
                    q_w = self.base_img_quarter.shape[1]
                    if q_w > preview_w_res and zoom_scale < 0.5:
                        res_key_roi = "quarter"
                        base_roi_img = self.base_img_quarter

                # Final safety: If the best tier we found isn't actually better than
                # the preview, skip ROI entirely.
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

                # Resolution-scaled coordinates for the ROI tier
                h_tier, w_tier = base_roi_img.shape[:2]
                s_x = int(src_x * (w_tier / full_w))
                s_y = int(src_y * (h_tier / full_h))
                s_x2 = int(src_x2 * (w_tier / full_w))
                s_y2 = int(src_y2 * (h_tier / full_h))

                # COMPUTE ROI CHUNK ONLY
                # We crop the raw data first to avoid processing the whole large image.
                # We add some padding to the crop so that local filters (like bilateral)
                # don't have artifacts at the edges of the visible ROI.
                pad = 16
                p_x1 = max(0, s_x - pad)
                p_y1 = max(0, s_y - pad)
                p_x2 = min(w_tier, s_x2 + pad)
                p_y2 = min(h_tier, s_y2 + pad)

                raw_chunk = base_roi_img[p_y1:p_y2, p_x1:p_x2]

                # We use a unique resolution key for this ROI position to enable caching
                # of intermediate stages while panning.
                roi_res_key = (res_key_roi, s_x, s_y, s_x2, s_y2)

                processed_chunk_padded = self._process_heavy_stage(
                    raw_chunk, roi_res_key, heavy_params, zoom_scale
                )

                # Remove the padding we added earlier
                c_y1 = s_y - p_y1
                c_x1 = s_x - p_x1
                c_y2 = c_y1 + (s_y2 - s_y)
                c_x2 = c_x1 + (s_x2 - s_x)

                crop_chunk = processed_chunk_padded[c_y1:c_y2, c_x1:c_x2]

                if flip_h or flip_v:
                    flip_code = -1 if (flip_h and flip_v) else (1 if flip_h else 0)
                    crop_chunk = cv2.flip(crop_chunk, flip_code)

                # Tone Map for ROI (Fast) - operates on the already heavy-processed chunk
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
        """Calculate RGB and YUV histograms efficiently."""
        bins = 256
        h, w = img_array.shape[:2]

        # If image is still large (background preview is ~2048px), downsample for histogram speed
        if max(h, w) > 512:
            scale = 256 / max(h, w)
            small_img = cv2.resize(
                img_array,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            small_img = img_array

        # Handle RGBA from rotation
        if small_img.shape[2] == 4:
            small_img = cv2.cvtColor(small_img, cv2.COLOR_RGBA2RGB)

        # RGB Histograms (Use calcHist on uint8)
        hist_r = cv2.calcHist([small_img], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([small_img], [1], None, [bins], [0, 256]).flatten()
        hist_b = cv2.calcHist([small_img], [2], None, [bins], [0, 256]).flatten()

        # YUV Histograms
        img_yuv = cv2.cvtColor(small_img, cv2.COLOR_RGB2YUV)
        hist_y = cv2.calcHist([img_yuv], [0], None, [bins], [0, 256]).flatten()
        hist_u = cv2.calcHist([img_yuv], [1], None, [bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([img_yuv], [2], None, [bins], [0, 256]).flatten()

        # Apply smoothing
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


class ImageProcessingPipeline(QtCore.QObject):
    previewUpdated = QtCore.Signal(
        QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int
    )
    histogramUpdated = QtCore.Signal(dict)
    performanceMeasured = QtCore.Signal(float)
    uneditedPixmapUpdated = QtCore.Signal(QtGui.QPixmap)
    editedPixmapUpdated = QtCore.Signal(QtGui.QPixmap)

    def __init__(self, thread_pool, parent=None):
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.render_timer = QtCore.QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._on_render_timer_timeout)
        self._render_pending = False
        self._is_rendering_locked = False
        self.base_img_full = None
        self.base_img_half = None
        self.base_img_quarter = None
        self.base_img_preview = None
        self._unedited_img_full = None  # Original raw data for unedited comparison
        self._processing_params = {}
        self._last_heavy_adjusted = "de_haze"
        self._view_ref = None
        self.perf_start_time = 0
        self.histogram_enabled = False

        # Request ID tracking to prevent out-of-order frames
        self._current_request_id = 0
        self._last_processed_id = -1
        self.cache = PipelineCache()

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.histogramUpdated.connect(self._on_histogram_updated)
        self.signals.error.connect(self._on_worker_error)

    def set_image(self, img_array):
        self.base_img_full = img_array
        # Store original unedited data for comparison
        if img_array is not None:
            self._unedited_img_full = img_array.copy()
        else:
            self._unedited_img_full = None
        self.cache.clear()
        # Reset processing parameters for the new image to avoid carrying over
        # edits from the previous one, unless we explicitly load them.
        self._processing_params = {}
        if img_array is not None:
            h, w, _ = img_array.shape

            # 1. Create a 50% scale RAW for intermediate zooms (75% <= Zoom < 200%)
            self.base_img_half = cv2.resize(
                img_array, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR
            )

            # 2. Create a 25% scale RAW for lower zooms (Fit < Zoom < 75%)
            self.base_img_quarter = cv2.resize(
                img_array, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR
            )

            # 3. Create a 2048px float32 preview for global background.
            scale = 2048 / max(h, w)
            target_h, target_w = int(h * scale), int(w * scale)
            self.base_img_preview = cv2.resize(
                img_array, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

            # Emit unedited pixmap update
            unedited_pixmap = self.get_unedited_pixmap()
            self.uneditedPixmapUpdated.emit(unedited_pixmap)
        else:
            self.base_img_half = None
            self.base_img_quarter = None
            self.base_img_preview = None

    def set_view_reference(self, view):
        self._view_ref = view

    def get_unedited_pixmap(self) -> QtGui.QPixmap:
        """Convert the unedited raw image to a QPixmap for comparison view."""
        if self._unedited_img_full is None:
            return QtGui.QPixmap()

        try:
            # Normalize to 0-1 range if needed
            img = self._unedited_img_full.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

            # Clamp to 0-1 range
            img = np.clip(img, 0.0, 1.0)

            # Convert to uint8 for display
            img_uint8 = (img * 255).astype(np.uint8)

            # Convert to RGB if needed
            if img_uint8.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = img_uint8

            # Convert to QImage then QPixmap
            h, w, c = img_rgb.shape
            bytes_per_line = c * w
            qimage = QtGui.QImage(
                img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
            )
            return QtGui.QPixmap.fromImage(qimage)
        except Exception:
            return QtGui.QPixmap()

    def set_histogram_enabled(self, enabled):
        self.histogram_enabled = enabled
        if enabled:
            self.request_update()

    def set_processing_params(self, **kwargs):
        # Track which heavy effect was last adjusted to optimize pipeline order
        heavy_keys = {"de_haze", "denoise_luma", "denoise_chroma", "sharpen_value"}
        for k in kwargs:
            if k in heavy_keys:
                self._last_heavy_adjusted = k
                break
        self._processing_params.update(kwargs)

    def get_current_settings(self):
        return self._processing_params.copy()

    def request_update(self):
        if self.base_img_full is None:
            return
        self._render_pending = True
        if not self._is_rendering_locked:
            self._process_pending_update()

    def _process_pending_update(self):
        if (
            not self._render_pending
            or self.base_img_full is None
            or self._view_ref is None
        ):
            return
        self._render_pending = False
        self._is_rendering_locked = True
        self.perf_start_time = time.perf_counter()

        self._current_request_id += 1
        worker = ImageProcessorWorker(
            self.signals,
            self._view_ref,
            self.base_img_full,
            self.base_img_half,
            self.base_img_quarter,
            self.base_img_preview,
            self.get_current_settings(),
            self._current_request_id,
            calculate_histogram=self.histogram_enabled,
            cache=self.cache,
            last_heavy_adjusted=self._last_heavy_adjusted,
        )
        self.thread_pool.start(worker)


    def _measure_and_emit_perf(self):
        elapsed_ms = (time.perf_counter() - self.perf_start_time) * 1000
        self.performanceMeasured.emit(elapsed_ms)

    @QtCore.Slot(QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int, int)
    def _on_worker_finished(
        self,
        pix_bg,
        full_w,
        full_h,
        pix_roi,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        request_id,
    ):
        # Unlock rendering since the worker has finished
        self._is_rendering_locked = False

        if request_id < self._last_processed_id:
            # If we were locked and a new request came in, process it now
            if self._render_pending:
                self._process_pending_update()
            return
        self._last_processed_id = request_id

        # Emit preview update (original signal)
        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )

        # Emit edited pixmap update for comparison overlay
        self.editedPixmapUpdated.emit(pix_bg)

        self._measure_and_emit_perf()

        # If a new request came in while this one was processing, start it now
        if self._render_pending:
            self._process_pending_update()

    @QtCore.Slot(dict, int)
    def _on_histogram_updated(self, hist_data, request_id):
        if request_id < self._last_processed_id:
            return
        self.histogramUpdated.emit(hist_data)

    @QtCore.Slot(str, int)
    def _on_worker_error(self, error_message, request_id):
        # Always unlock on error so we can try again
        self._is_rendering_locked = False
        if self._render_pending:
            self._process_pending_update()

        if request_id < self._last_processed_id:
            return
        print(f"Image processing error (ID {request_id}): {error_message}")
