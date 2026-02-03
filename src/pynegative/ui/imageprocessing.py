import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui
import time
import cv2
from .. import core as pynegative


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int)
    histogramUpdated = QtCore.Signal(dict)
    error = QtCore.Signal(str)


class ImageProcessorWorker(QtCore.QRunnable):
    """Worker to process a single large ROI in a background thread."""

    def __init__(
        self,
        signals,
        view_ref,
        base_img_full,
        base_img_preview,
        settings,
        calculate_histogram=False,
    ):
        super().__init__()
        self.signals = signals
        self._view_ref = view_ref
        self.base_img_full = base_img_full
        self.base_img_preview = base_img_preview
        self.settings = settings
        self.calculate_histogram = calculate_histogram

    def run(self):
        try:
            result = self._update_preview()
            self.signals.finished.emit(*result)
        except Exception as e:
            self.signals.error.emit(str(e))

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
                "exposure",
                "contrast",
                "blacks",
                "whites",
                "shadows",
                "highlights",
                "saturation",
            ]
        }
        processed_bg, _ = pynegative.apply_tone_map(
            img_render_base, **tone_map_settings
        )

        # --- Part 1.5: Histogram (calculated on background preview for speed) ---
        if self.calculate_histogram:
            try:
                hist_data = self._calculate_histograms(processed_bg)
                self.signals.histogramUpdated.emit(hist_data)
            except Exception as e:
                print(f"Histogram calculation error: {e}")

        pil_bg = Image.fromarray((processed_bg * 255).astype(np.uint8))
        pix_bg = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_bg))

        # --- Part 2: Detail ROI ---
        pix_roi, roi_x, roi_y, roi_w, roi_h = QtGui.QPixmap(), 0, 0, 0, 0
        if is_zoomed_in:
            roi = self._view_ref.mapToScene(
                self._view_ref.viewport().rect()
            ).boundingRect()
            ix_min, iy_min = max(0, int(roi.left())), max(0, int(roi.top()))
            ix_max, iy_max = (
                min(full_w, int(roi.right())),
                min(full_h, int(roi.bottom())),
            )

            if (rw := ix_max - ix_min) > 10 and (rh := iy_max - iy_min) > 10:
                crop = self.base_img_full[iy_min:iy_max, ix_min:ix_max]
                processed_roi, _ = pynegative.apply_tone_map(crop, **tone_map_settings)
                pil_roi = Image.fromarray((processed_roi * 255).astype(np.uint8))

                if self.settings.get("sharpen_value", 0) > 0:
                    pil_roi = pynegative.sharpen_image(
                        pil_roi,
                        self.settings["sharpen_radius"],
                        self.settings["sharpen_percent"],
                        "High Quality",
                    )
                if self.settings.get("de_noise", 0) > 0:
                    pil_roi = pynegative.de_noise_image(
                        pil_roi, self.settings["de_noise"], "High Quality"
                    )

                pix_roi = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_roi))
                roi_x, roi_y, roi_w, roi_h = ix_min, iy_min, rw, rh

        return pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h

    def _calculate_histograms(self, img_float):
        """Calculate RGB and YUV histograms with high performance."""
        # Use 256 bins for the display
        bins = 256

        # Performance optimization 1: Downsample heavily for histogram.
        # 256px resolution is more than enough for an accurate histogram.
        h, w, _ = img_float.shape
        hist_scale = 256 / max(h, w)
        # Use INTER_NEAREST for maximum speed during histogram downsampling
        small_img = cv2.resize(
            img_float,
            (int(w * hist_scale), int(h * hist_scale)),
            interpolation=cv2.INTER_NEAREST,
        )

        # Performance optimization 2: Use cv2.calcHist on uint16 data.
        # This is significantly faster than np.histogram and preserves 16-bit precision.
        img_u16 = (small_img * 65535).astype(np.uint16)

        # RGB Histograms (Calculate on 16-bit to avoid sawtooth)
        hist_r = cv2.calcHist([img_u16], [0], None, [bins], [0, 65536]).flatten()
        hist_g = cv2.calcHist([img_u16], [1], None, [bins], [0, 65536]).flatten()
        hist_b = cv2.calcHist([img_u16], [2], None, [bins], [0, 65536]).flatten()

        # YUV Histograms (Use uint8 for speed as visual precision is less critical here)
        img_u8 = (small_img * 255).astype(np.uint8)
        img_yuv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV)
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
        self.base_img_preview = None
        self._processing_params = {}
        self._view_ref = None
        self.perf_start_time = 0
        self.histogram_enabled = False

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.histogramUpdated.connect(self.histogramUpdated.emit)
        self.signals.error.connect(self._on_worker_error)

    def set_image(self, img_array):
        self.base_img_full = img_array
        if img_array is not None:
            # Create a 2048px float32 preview once.
            # This is reused for background rendering and histograms, avoiding
            # expensive full-res resizes during slider updates.
            h, w, _ = img_array.shape
            scale = 2048 / max(h, w)
            target_h, target_w = int(h * scale), int(w * scale)
            self.base_img_preview = cv2.resize(
                img_array, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        else:
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
        self.render_timer.start(33)
        self.perf_start_time = time.perf_counter()

        worker = ImageProcessorWorker(
            self.signals,
            self._view_ref,
            self.base_img_full,
            self.base_img_preview,
            self.get_current_settings(),
            calculate_histogram=self.histogram_enabled,
        )
        self.thread_pool.start(worker)

    def _on_render_timer_timeout(self):
        self._is_rendering_locked = False
        if self._render_pending:
            self._process_pending_update()

    def _measure_and_emit_perf(self):
        elapsed_ms = (time.perf_counter() - self.perf_start_time) * 1000
        self.performanceMeasured.emit(elapsed_ms)

    @QtCore.Slot(QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int)
    def _on_worker_finished(
        self, pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
    ):
        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )
        self._measure_and_emit_perf()

    @QtCore.Slot(str)
    def _on_worker_error(self, error_message):
        print(f"Image processing error: {error_message}")
