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

    def run(self):
        try:
            result = self._update_preview()
            self.signals.finished.emit(*result, self.request_id)
        except Exception as e:
            self.signals.error.emit(str(e), self.request_id)

    def _update_preview(self):
        if self.base_img_full is None or self._view_ref is None:
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0

        full_h, full_w, _ = self.base_img_full.shape

        try:
            zoom_scale = self._view_ref.transform().m11()
            viewport = self._view_ref.viewport()
            vw, vh = viewport.width(), viewport.height()
        except (AttributeError, RuntimeError):
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0

        fit_scale = min(vw / full_w, vh / full_h) if vw > 0 and vh > 0 else 1.0
        is_fitting = getattr(self._view_ref, "_is_fitting", False)
        is_zoomed_in = not is_fitting and (
            zoom_scale > fit_scale * 1.01 or zoom_scale > 0.99
        )

        # --- Part 1: Global Background ---
        # Use the pre-resized preview for the background and histogram.
        # This is MUCH faster than resizing from full-res on every slider move.
        img_render_base = self.base_img_preview

        tone_map_settings = {
            k: v
            for k, v in self.settings.items()
            if k
            in [
                "temperature",
                "tint",
                "exposure",
                "contrast",
                "blacks",
                "whites",
                "shadows",
                "highlights",
                "saturation",
            ]
        }
        # Optimization: Pass calculate_stats=False for live preview
        processed_bg, _ = pynegative.apply_tone_map(
            img_render_base, **tone_map_settings, calculate_stats=False
        )

        # Apply De-haze to background
        if not is_zoomed_in and self.settings.get("de_haze", 0) > 0:
            processed_bg = pynegative.de_haze_image(
                processed_bg,
                self.settings["de_haze"],
            )

        # Apply De-noise to background (Skip if zoomed in to save performance, as ROI will cover it)
        if not is_zoomed_in and self.settings.get("de_noise", 0) > 0:
            processed_bg = pynegative.de_noise_image(
                processed_bg,
                self.settings["de_noise"],
                self.settings.get("denoise_method", "High Quality"),
                zoom=zoom_scale,
            )

        # Apply Sharpening to background (Skip if zoomed in)
        if not is_zoomed_in and self.settings.get("sharpen_value", 0) > 0:
            processed_bg = pynegative.sharpen_image(
                processed_bg,
                self.settings.get("sharpen_radius", 0.5),
                self.settings.get("sharpen_percent", 0.0),
                "High Quality",
            )

        # Prepare image for geometry (convert to uint8 for OpenCV)
        img_uint8 = (processed_bg * 255).astype(np.uint8)

        rotate_val = self.settings.get("rotation", 0.0)
        flip_h = self.settings.get("flip_h", False)
        flip_v = self.settings.get("flip_v", False)
        crop_val = self.settings.get(
            "crop", None
        )  # (left, top, right, bottom) normalized

        # 0. Apply Flip
        if flip_h or flip_v:
            flip_code = -1 if (flip_h and flip_v) else (1 if flip_h else 0)
            img_uint8 = cv2.flip(img_uint8, flip_code)

        # 1. Apply Rotation using OpenCV (Much faster than PIL)
        if abs(rotate_val) > 0.01:
            h, w = img_uint8.shape[:2]
            center = (w / 2, h / 2)

            # Add alpha channel for transparency if rotating
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2RGBA)

            # Rotation Matrix
            # OpenCV rotates CCW for positive angles.
            # The user wants negative to be CW, so positive is CCW.
            # This matches OpenCV's behavior.
            M = cv2.getRotationMatrix2D(center, rotate_val, 1.0)

            # Calculate new bounding box to avoid clipping (equivalent to expand=True)
            cos_val = np.abs(M[0, 0])
            sin_val = np.abs(M[0, 1])
            new_w = int((h * sin_val) + (w * cos_val))
            new_h = int((h * cos_val) + (w * sin_val))

            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # Perform rotation (INTER_NEAREST for speed during interactive preview)
            # Use borderValue=(0,0,0,0) for transparent background
            img_uint8 = cv2.warpAffine(
                img_uint8,
                M,
                (new_w, new_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

        # 2. Apply Crop using Numpy Slicing
        if crop_val is not None:
            h, w = img_uint8.shape[:2]
            c_left, c_top, c_right, c_bottom = crop_val

            x1 = int(c_left * w)
            y1 = int(c_top * h)
            x2 = int(c_right * w)
            y2 = int(c_bottom * h)

            # Clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 > x1 and y2 > y1:
                img_uint8 = img_uint8[y1:y2, x1:x2]

        # Convert back to RGB for PIL if we added an alpha channel for rotation
        if img_uint8.shape[2] == 4:
            # But wait, we want to keep transparency in the background if possible?
            # Actually ZoomableGraphicsView background is black anyway.
            # If we want transparent export later, we need RGBA.
            # PIL supports RGBA.
            pass

        pil_bg = Image.fromarray(img_uint8)

        # Calculate new virtual full dimensions
        # The preview is a scaled down version. We need to project the new dimensions back to full resolution.
        preview_h, preview_w = img_render_base.shape[:2]
        full_h, full_w = self.base_img_full.shape[:2]
        scale_x = full_w / preview_w
        scale_y = full_h / preview_h

        # New "Full" dimensions after geometry
        new_full_w = int(pil_bg.width * scale_x)
        new_full_h = int(pil_bg.height * scale_y)

        # --- Part 1.5: Histogram ---
        if self.calculate_histogram:
            try:
                # Use the already processed uint8 image for histogram (much faster)
                hist_data = self._calculate_histograms(img_uint8)
                self.signals.histogramUpdated.emit(hist_data, self.request_id)
            except Exception as e:
                print(f"Histogram calculation error: {e}")

        pix_bg = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_bg))

        # --- Part 2: Detail ROI ---
        pix_roi, roi_x, roi_y, roi_w, roi_h = QtGui.QPixmap(), 0, 0, 0, 0

        if is_zoomed_in and abs(rotate_val) < 0.1:
            roi = self._view_ref.mapToScene(
                self._view_ref.viewport().rect()
            ).boundingRect()

            v_x, v_y, v_w, v_h = roi.x(), roi.y(), roi.width(), roi.height()

            offset_x, offset_y = 0, 0
            if crop_val:
                orig_full_w = self.base_img_full.shape[1]
                orig_full_h = self.base_img_full.shape[0]
                offset_x = int(crop_val[0] * orig_full_w)
                offset_y = int(crop_val[1] * orig_full_h)

            src_x, src_y = int(v_x + offset_x), int(v_y + offset_y)
            src_w, src_h = int(v_w), int(v_h)

            orig_w, orig_h = self.base_img_full.shape[1], self.base_img_full.shape[0]
            if flip_h:
                src_x = orig_w - (src_x + src_w)
            if flip_v:
                src_y = orig_h - (src_y + src_h)

            src_x, src_y = max(0, src_x), max(0, src_y)
            src_x2, src_y2 = min(orig_w, src_x + src_w), min(orig_h, src_y + src_h)

            if (req_w := src_x2 - src_x) > 10 and (req_h := src_y2 - src_y) > 10:
                # Decide which source to use based on zoom level
                # Tiered approach for performance:
                # 1. Zoom < 50%: Use 1/4 size RAW
                # 2. 50% <= Zoom < 150%: Use 1/2 size RAW
                # 3. Zoom >= 150%: Use Full size RAW
                if zoom_scale < 0.5 and self.base_img_quarter is not None:
                    # Scale coordinates to quarter-res
                    q_src_x, q_src_y = src_x // 4, src_y // 4
                    q_src_x2, q_src_y2 = src_x2 // 4, src_y2 // 4
                    crop_chunk = self.base_img_quarter[
                        q_src_y:q_src_y2, q_src_x:q_src_x2
                    ]
                elif zoom_scale < 1.5 and self.base_img_half is not None:
                    # Scale coordinates to half-res
                    h_src_x, h_src_y = src_x // 2, src_y // 2
                    h_src_x2, h_src_y2 = src_x2 // 2, src_y2 // 2
                    crop_chunk = self.base_img_half[h_src_y:h_src_y2, h_src_x:h_src_x2]
                else:
                    crop_chunk = self.base_img_full[src_y:src_y2, src_x:src_x2]

                if flip_h or flip_v:
                    flip_code = -1 if (flip_h and flip_v) else (1 if flip_h else 0)
                    crop_chunk = cv2.flip(crop_chunk, flip_code)

                # Process ROI in float32 Numpy
                processed_roi, _ = pynegative.apply_tone_map(
                    crop_chunk, **tone_map_settings, calculate_stats=False
                )

                # 1.5 De-haze
                if self.settings.get("de_haze", 0) > 0:
                    processed_roi = pynegative.de_haze_image(
                        processed_roi,
                        self.settings["de_haze"],
                    )

                # 2. De-noise first
                if self.settings.get("de_noise", 0) > 0:
                    processed_roi = pynegative.de_noise_image(
                        processed_roi,
                        self.settings["de_noise"],
                        self.settings.get("denoise_method", "High Quality"),
                        zoom=zoom_scale,
                    )

                # 3. Sharpen second (to avoid sharpening noise)
                if self.settings.get("sharpen_value", 0) > 0:
                    processed_roi = pynegative.sharpen_image(
                        processed_roi,
                        self.settings["sharpen_radius"],
                        self.settings["sharpen_percent"],
                        "High Quality",
                    )

                # Final ROI conversion to QPixmap
                pil_roi = Image.fromarray((processed_roi * 255).astype(np.uint8))
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
        self._processing_params = {}
        self._view_ref = None
        self.perf_start_time = 0
        self.histogram_enabled = False

        # Request ID tracking to prevent out-of-order frames
        self._current_request_id = 0
        self._last_processed_id = -1

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.histogramUpdated.connect(self._on_histogram_updated)
        self.signals.error.connect(self._on_worker_error)

    def set_image(self, img_array):
        self.base_img_full = img_array
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
        else:
            self.base_img_half = None
            self.base_img_quarter = None
            self.base_img_preview = None

    def set_view_reference(self, view):
        self._view_ref = view

    def set_histogram_enabled(self, enabled):
        self.histogram_enabled = enabled
        if enabled:
            self.request_update()

    def set_processing_params(self, **kwargs):
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
        )
        self.thread_pool.start(worker)

    def _on_render_timer_timeout(self):
        pass

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

        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )
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
