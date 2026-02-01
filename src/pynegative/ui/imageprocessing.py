import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui
import uuid
import time
from .. import core as pynegative

# --- Tiled Rendering Constants ---
BORDER_SIZE = 32  # For filters like sharpen/denoise

# --- Phase 1: Single Worker Classes ---


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the single image processing worker."""

    finished = QtCore.Signal(QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int)
    error = QtCore.Signal(str)


class ImageProcessorWorker(QtCore.QRunnable):
    """Worker to process a single large ROI in a background thread."""

    def __init__(self, signals, view_ref, base_img_full, settings):
        super().__init__()
        self.signals = signals
        self._view_ref = view_ref
        self.base_img_full = base_img_full
        self.settings = settings
        self._base_img_uint8 = None

    def run(self):
        try:
            result = self._update_preview()
            self.signals.finished.emit(*result)
        except Exception as e:
            self.signals.error.emit(str(e))

    def _update_preview(self):
        # This is the full, non-tiled processing logic from Phase 1
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
        base_img_uint8 = (self.base_img_full * 255).astype(np.uint8)
        scale = 1500 / max(full_h, full_w)
        target_h, target_w = int(full_h * scale), int(full_w * scale)
        temp_pil = Image.fromarray(base_img_uint8).resize(
            (target_w, target_h), Image.Resampling.BILINEAR
        )
        img_render_base = np.array(temp_pil).astype(np.float32) / 255.0

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


# --- Phase 2: Tiled Worker Classes ---


class TileSignals(QtCore.QObject):
    finished = QtCore.Signal(str, int, int, QtGui.QImage)
    error = QtCore.Signal(str)


class TileWorker(QtCore.QRunnable):
    def __init__(self, signals, job_id, tile_x, tile_y, image_crop, settings):
        super().__init__()
        (
            self.signals,
            self.job_id,
            self.tile_x,
            self.tile_y,
            self.image_crop,
            self.settings,
        ) = signals, job_id, tile_x, tile_y, image_crop, settings

    def run(self):
        try:
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
            processed_crop, _ = pynegative.apply_tone_map(
                self.image_crop, **tone_map_settings
            )
            pil_img = Image.fromarray((processed_crop * 255).astype(np.uint8))

            if self.settings.get("sharpen_value", 0) > 0:
                pil_img = pynegative.sharpen_image(
                    pil_img,
                    self.settings["sharpen_radius"],
                    self.settings["sharpen_percent"],
                    "High Quality",
                )
            if self.settings.get("de_noise", 0) > 0:
                pil_img = pynegative.de_noise_image(
                    pil_img, self.settings["de_noise"], "High Quality"
                )

            final_tile_pil = pil_img.crop(
                (
                    BORDER_SIZE,
                    BORDER_SIZE,
                    pil_img.width - BORDER_SIZE,
                    pil_img.height - BORDER_SIZE,
                )
            )
            self.signals.finished.emit(
                self.job_id, self.tile_x, self.tile_y, ImageQt.ImageQt(final_tile_pil)
            )
        except Exception as e:
            self.signals.error.emit(f"Error in tile {self.tile_x},{self.tile_y}: {e}")


class TileRenderJob(QtCore.QObject):
    jobFinished = QtCore.Signal(QtGui.QPixmap)

    def __init__(self, job_id, roi_w, roi_h, tiles_to_process, parent=None):
        super().__init__(parent)
        self.job_id, self.tiles_to_process, self.finished_tiles = (
            job_id,
            tiles_to_process,
            0,
        )
        self.output_image = QtGui.QImage(roi_w, roi_h, QtGui.QImage.Format_RGB888)
        self.output_image.fill(QtCore.Qt.black)

    def add_tile(self, tile_x, tile_y, image_data):
        painter = QtGui.QPainter(self.output_image)
        painter.drawImage(tile_x, tile_y, image_data)
        painter.end()
        self.finished_tiles += 1
        if self.finished_tiles == self.tiles_to_process:
            self.finish_job()

    def finish_job(self):
        self.jobFinished.emit(QtGui.QPixmap.fromImage(self.output_image))


# --- Main Pipeline ---


class ImageProcessingPipeline(QtCore.QObject):
    previewUpdated = QtCore.Signal(
        QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int
    )
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
        self._processing_params = {}
        self._view_ref = None
        self._current_job_id = None
        self._jobs = {}
        self.perf_start_time = 0

        # Configurable rendering mode
        self.use_tiled_rendering = True

        # Signals for both modes
        self.single_worker_signals = ImageProcessorSignals()
        self.single_worker_signals.finished.connect(self._on_single_worker_finished)
        self.single_worker_signals.error.connect(self._on_worker_error)
        self.tile_signals = TileSignals()
        self.tile_signals.finished.connect(self._on_tile_finished)
        self.tile_signals.error.connect(self._on_worker_error)

    def set_image(self, img_array):
        self.base_img_full = img_array

    def set_view_reference(self, view):
        self._view_ref = view

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
        self._current_job_id = str(uuid.uuid4())
        self.perf_start_time = time.perf_counter()

        if self.use_tiled_rendering:
            self._execute_tiled_update()
        else:
            self._execute_single_worker_update()

    def _execute_single_worker_update(self):
        worker = ImageProcessorWorker(
            self.single_worker_signals,
            self._view_ref,
            self.base_img_full,
            self.get_current_settings(),
        )
        self.thread_pool.start(worker)

    def _execute_tiled_update(self):
        full_h, full_w, _ = self.base_img_full.shape
        try:
            zoom_scale = self._view_ref.transform().m11()
            vw, vh = (
                self._view_ref.viewport().width(),
                self._view_ref.viewport().height(),
            )
            fit_scale = min(vw / full_w, vh / full_h)
            is_fitting = getattr(self._view_ref, "_is_fitting", False)
            is_zoomed_in = not is_fitting and (
                zoom_scale > fit_scale * 1.01 or zoom_scale > 0.99
            )
        except (AttributeError, RuntimeError):
            is_zoomed_in = False

        # Always emit a low-res background immediately
        scale = 1500 / max(full_h, full_w)
        target_h, target_w = int(full_h * scale), int(full_w * scale)
        base_img_uint8 = (self.base_img_full * 255).astype(np.uint8)
        temp_pil = Image.fromarray(base_img_uint8).resize(
            (target_w, target_h), Image.Resampling.BILINEAR
        )
        img_render_base = np.array(temp_pil).astype(np.float32) / 255.0
        tone_map_settings = {
            k: v
            for k, v in self.get_current_settings().items()
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
        pix_bg = QtGui.QPixmap.fromImage(
            ImageQt.ImageQt(Image.fromarray((processed_bg * 255).astype(np.uint8)))
        )

        if not is_zoomed_in:
            self.previewUpdated.emit(
                pix_bg, full_w, full_h, QtGui.QPixmap(), 0, 0, 0, 0
            )
            self._measure_and_emit_perf()
            return

        roi = self._view_ref.mapToScene(self._view_ref.viewport().rect()).boundingRect()
        roi_x, roi_y, roi_w, roi_h = (
            int(roi.left()),
            int(roi.top()),
            int(roi.width()),
            int(roi.height()),
        )

        if roi_w <= 0 or roi_h <= 0:
            self.previewUpdated.emit(
                pix_bg, full_w, full_h, QtGui.QPixmap(), 0, 0, 0, 0
            )
            self._measure_and_emit_perf()
            return

        job = TileRenderJob(self._current_job_id, roi_w, roi_h, 4)  # Always 4 tiles
        job.jobFinished.connect(
            lambda pix: self._on_job_finished(
                self._current_job_id,
                pix_bg,
                full_w,
                full_h,
                roi_x,
                roi_y,
                roi_w,
                roi_h,
                pix,
            )
        )
        self._jobs[self._current_job_id] = job

        tile_w = roi_w // 2
        tile_h = roi_h // 2

        # (x, y, w, h) in ROI coordinates
        tile_geometries = [
            (0, 0, tile_w, tile_h),
            (tile_w, 0, roi_w - tile_w, tile_h),
            (0, tile_h, tile_w, roi_h - tile_h),
            (tile_w, tile_h, roi_w - tile_w, roi_h - tile_h),
        ]

        for x, y, w, h in tile_geometries:
            if w <= 0 or h <= 0:
                continue

            img_x, img_y = roi_x + x, roi_y + y
            crop_x_start, crop_y_start = (
                max(0, img_x - BORDER_SIZE),
                max(0, img_y - BORDER_SIZE),
            )
            crop_x_end, crop_y_end = (
                min(full_w, img_x + w + BORDER_SIZE),
                min(full_h, img_y + h + BORDER_SIZE),
            )
            image_crop = self.base_img_full[
                crop_y_start:crop_y_end, crop_x_start:crop_x_end
            ]

            worker = TileWorker(
                self.tile_signals,
                self._current_job_id,
                x,
                y,
                image_crop,
                self.get_current_settings(),
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
    def _on_single_worker_finished(
        self, pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
    ):
        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )
        self._measure_and_emit_perf()

    @QtCore.Slot(str, int, int, QtGui.QImage)
    def _on_tile_finished(self, job_id, tile_x, tile_y, image_data):
        if job_id in self._jobs:
            self._jobs[job_id].add_tile(tile_x, tile_y, image_data.copy())

    def _on_job_finished(
        self, job_id, pix_bg, full_w, full_h, roi_x, roi_y, roi_w, roi_h, roi_pixmap
    ):
        if job_id == self._current_job_id:
            self.previewUpdated.emit(
                pix_bg, full_w, full_h, roi_pixmap, roi_x, roi_y, roi_w, roi_h
            )
            self._measure_and_emit_perf()
        if job_id in self._jobs:
            del self._jobs[job_id]

    @QtCore.Slot(str)
    def _on_worker_error(self, error_message):
        print(f"Image processing error: {error_message}")
