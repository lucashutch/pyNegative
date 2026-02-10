import numpy as np
from PySide6 import QtCore, QtGui
import time
import cv2
from .pipeline.cache import PipelineCache
from .pipeline.worker import ImageProcessorSignals, ImageProcessorWorker


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
        self._unedited_img_full = None
        self._processing_params = {}
        self._last_heavy_adjusted = "de_haze"
        self._view_ref = None
        self.perf_start_time = 0
        self.histogram_enabled = False

        self._current_request_id = 0
        self._last_processed_id = -1
        self.cache = PipelineCache()

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.histogramUpdated.connect(self._on_histogram_updated)
        self.signals.error.connect(self._on_worker_error)

    def set_image(self, img_array):
        self.base_img_full = img_array
        # Use the same array reference, open_raw returns a fresh array anyway
        self._unedited_img_full = img_array
        self.cache.clear()
        self._processing_params = {}
        if img_array is not None:
            h, w, _ = img_array.shape
            self.base_img_half = cv2.resize(
                img_array, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR
            )
            # Chain resizes: Quarter from Half is much faster
            self.base_img_quarter = cv2.resize(
                self.base_img_half, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR
            )
            scale = 2048 / max(h, w)
            target_h, target_w = int(h * scale), int(w * scale)
            # Preview from whichever is closest and larger
            src_for_preview = img_array if scale > 0.5 else self.base_img_half
            self.base_img_preview = cv2.resize(
                src_for_preview, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            self.uneditedPixmapUpdated.emit(self.get_unedited_pixmap())
        else:
            self.base_img_half = None
            self.base_img_quarter = None
            self.base_img_preview = None

    def set_view_reference(self, view):
        self._view_ref = view

    def get_unedited_pixmap(self) -> QtGui.QPixmap:
        if self._unedited_img_full is None:
            return QtGui.QPixmap()
        try:
            # We know the pipeline uses float32 0-1 range by convention
            img_uint8 = (np.clip(self._unedited_img_full, 0, 1) * 255).astype(np.uint8)
            if img_uint8.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = img_uint8
            h, w, c = img_rgb.shape
            bytes_per_line = c * w
            qimage = QtGui.QImage(
                img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
            )
            # Must return a copy if we want to ensure the underlying buffer stays alive,
            # but QPixmap.fromImage already handles this.
            return QtGui.QPixmap.fromImage(qimage)
        except Exception:
            return QtGui.QPixmap()

    def set_histogram_enabled(self, enabled):
        self.histogram_enabled = enabled
        if enabled:
            self.request_update()

    def set_processing_params(self, **kwargs):
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

    def _on_render_timer_timeout(self):
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
        self, pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h, request_id
    ):
        self._is_rendering_locked = False
        if request_id < self._last_processed_id:
            if self._render_pending:
                self._process_pending_update()
            return
        self._last_processed_id = request_id
        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )
        self.editedPixmapUpdated.emit(pix_bg)
        self._measure_and_emit_perf()
        if self._render_pending:
            self._process_pending_update()

    @QtCore.Slot(dict, int)
    def _on_histogram_updated(self, hist_data, request_id):
        if request_id < self._last_processed_id:
            return
        self.histogramUpdated.emit(hist_data)

    @QtCore.Slot(str, int)
    def _on_worker_error(self, error_message, request_id):
        self._is_rendering_locked = False
        if self._render_pending:
            self._process_pending_update()
        if request_id < self._last_processed_id:
            return
        print(f"Image processing error (ID {request_id}): {error_message}")
