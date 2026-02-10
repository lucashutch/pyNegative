import numpy as np
from PySide6 import QtCore, QtGui
import time
import cv2
from .pipeline.cache import PipelineCache
from .pipeline.worker import (
    ImageProcessorSignals,
    ImageProcessorWorker,
    TierGeneratorWorker,
)


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
        self.signals.tiersGenerated.connect(self._on_tiers_generated)
        self.signals.uneditedPixmapGenerated.connect(self.uneditedPixmapUpdated.emit)
        self.signals.error.connect(self._on_worker_error)

        # Idle timer for expanded ROI (smooth panning)
        self.idle_timer = QtCore.QTimer()
        self.idle_timer.setSingleShot(True)
        self.idle_timer.timeout.connect(self._on_idle_timeout)
        self._last_roi_was_expanded = False
        self._last_roi_rect = QtCore.QRectF()

    def set_image(self, img_array):
        self.base_img_full = img_array
        self._unedited_img_full = img_array
        self.cache.clear()
        self._processing_params = {}

        # Reset tiers - they will be populated by the background worker
        self.base_img_half = None
        self.base_img_quarter = None
        self.base_img_preview = None

        if img_array is not None:
            # Trigger background tier generation
            worker = TierGeneratorWorker(self.signals, img_array)
            self.thread_pool.start(worker)
        else:
            self.uneditedPixmapUpdated.emit(QtGui.QPixmap())

    @QtCore.Slot(np.ndarray, np.ndarray, np.ndarray)
    def _on_tiers_generated(self, half, quarter, preview):
        self.base_img_half = half
        self.base_img_quarter = quarter
        self.base_img_preview = preview
        # Once tiers are ready, trigger a high-quality render if needed
        if self._render_pending or self._processing_params:
            self.request_update()

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
        # Settings changed, so last ROI is invalid
        self._last_roi_rect = QtCore.QRectF()

    def get_current_settings(self):
        return self._processing_params.copy()

    def request_update(self):
        if self.base_img_full is None:
            return

        # Check if we can skip this update because the current viewport is within the last ROI
        if self._view_ref is not None and not self._last_roi_rect.isEmpty():
            try:
                # Get current visible rect in scene coordinates
                viewport_rect = self._view_ref.mapToScene(
                    self._view_ref.viewport().rect()
                ).boundingRect()
                # If viewport is inside last ROI and no settings changed, skip update
                if (
                    self._last_roi_rect.contains(viewport_rect)
                    and not self._render_pending
                ):
                    # Still restart idle timer to eventually expand ROI if it wasn't
                    if not self._last_roi_was_expanded:
                        self.idle_timer.start(300)
                    return
            except (AttributeError, RuntimeError):
                pass

        self._render_pending = True
        # Reset idle timer on every user interaction
        self.idle_timer.stop()
        if not self._is_rendering_locked:
            self._process_pending_update()

    def _on_idle_timeout(self):
        """Triggered when user is idle, to render a larger padded ROI."""
        if (
            self.base_img_full is None
            or self._is_rendering_locked
            or self._render_pending
        ):
            return
        # Only expand if we haven't already
        if not self._last_roi_was_expanded:
            self._process_pending_update(expand_roi=True)

    def _on_render_timer_timeout(self):
        self._process_pending_update()

    def _process_pending_update(self, expand_roi=False):
        if (
            (not self._render_pending and not expand_roi)
            or self.base_img_full is None
            or self._view_ref is None
        ):
            return

        # If it's a regular update, clear the pending flag
        if not expand_roi:
            self._render_pending = False
            self._last_roi_was_expanded = False
        else:
            self._last_roi_was_expanded = True

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
            expand_roi=expand_roi,
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

        # Store the ROI rect in scene coordinates for panning checks
        if not pix_roi.isNull():
            self._last_roi_rect = QtCore.QRectF(roi_x, roi_y, roi_w, roi_h)
        else:
            self._last_roi_rect = QtCore.QRectF()

        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )
        self.editedPixmapUpdated.emit(pix_bg)
        self._measure_and_emit_perf()

        # After a successful render, start the idle timer to expand the ROI
        if not self._last_roi_was_expanded:
            self.idle_timer.start(300)  # 300ms idle threshold

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
