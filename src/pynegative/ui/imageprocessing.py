import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui
import uuid
from .. import core as pynegative

TILE_SIZE = 256
BORDER_SIZE = 32  # For filters like sharpen/denoise


class TileSignals(QtCore.QObject):
    """Signals for a single tile worker."""

    finished = QtCore.Signal(str, int, int, QtGui.QImage)
    error = QtCore.Signal(str)


class TileWorker(QtCore.QRunnable):
    """Worker to process a single image tile."""

    def __init__(self, signals, job_id, tile_x, tile_y, image_crop, settings):
        super().__init__()
        self.signals = signals
        self.job_id = job_id
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.image_crop = image_crop
        self.settings = settings

    def run(self):
        try:
            # 1. Process the entire crop (tile + border)
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
            processed_crop *= 255
            pil_img = Image.fromarray(processed_crop.astype(np.uint8))

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

            # 2. Crop out the border to get the final tile
            final_tile_pil = pil_img.crop(
                (
                    BORDER_SIZE,
                    BORDER_SIZE,
                    pil_img.width - BORDER_SIZE,
                    pil_img.height - BORDER_SIZE,
                )
            )

            # 3. Convert to QImage for signaling
            q_image = ImageQt.ImageQt(final_tile_pil)

            self.signals.finished.emit(self.job_id, self.tile_x, self.tile_y, q_image)

        except Exception as e:
            self.signals.error.emit(
                f"Error processing tile {self.tile_x},{self.tile_y}: {e}"
            )


class TileRenderJob(QtCore.QObject):
    """Manages the state of a tiled render job."""

    jobFinished = QtCore.Signal(QtGui.QPixmap)

    def __init__(self, job_id, roi_w, roi_h, tiles_to_process, parent=None):
        super().__init__(parent)
        self.job_id = job_id
        self.tiles_to_process = tiles_to_process
        self.finished_tiles = 0
        self.output_image = QtGui.QImage(roi_w, roi_h, QtGui.QImage.Format_RGB888)
        self.output_image.fill(QtCore.Qt.black)
        self.painter = QtGui.QPainter(self.output_image)

    def add_tile(self, tile_x, tile_y, image_data):
        self.painter.drawImage(tile_x, tile_y, image_data)
        self.finished_tiles += 1
        if self.finished_tiles == self.tiles_to_process:
            self.finish_job()

    def finish_job(self):
        self.painter.end()
        self.jobFinished.emit(QtGui.QPixmap.fromImage(self.output_image))


class ImageProcessingPipeline(QtCore.QObject):
    previewUpdated = QtCore.Signal(
        QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int
    )

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

        self.tile_signals = TileSignals()
        self.tile_signals.finished.connect(self._on_tile_finished)
        self.tile_signals.error.connect(self._on_worker_error)

    def set_image(self, img_array):
        self.base_img_full = img_array

    def set_view_reference(self, view):
        self._view_ref = view

    def set_processing_params(self, **kwargs):
        for key, value in kwargs.items():
            self._processing_params[key] = value

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

        # Cancel previous job
        self._current_job_id = str(uuid.uuid4())

        # --- Process Background (Low-res) ---
        full_h, full_w, _ = self.base_img_full.shape
        scale = 1500 / max(full_h, full_w)
        target_h, target_w = int(full_h * scale), int(full_w * scale)
        base_img_uint8 = (self.base_img_full * 255).astype(np.uint8)
        temp_pil = Image.fromarray(base_img_uint8)
        temp_pil = temp_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
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
        processed_bg *= 255
        pil_bg = Image.fromarray(processed_bg.astype(np.uint8))
        pix_bg = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_bg))

        # --- Process ROI (High-res, Tiled) ---
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

        if is_zoomed_in:
            roi = self._view_ref.mapToScene(
                self._view_ref.viewport().rect()
            ).boundingRect()
            roi_x, roi_y = int(roi.left()), int(roi.top())
            roi_w, roi_h = int(roi.width()), int(roi.height())

            if roi_w > 0 and roi_h > 0:
                job = TileRenderJob(self._current_job_id, roi_w, roi_h, 0)
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

                tiles = []
                for y in range(0, roi_h, TILE_SIZE):
                    for x in range(0, roi_w, TILE_SIZE):
                        # Real image coordinates for the tile crop
                        img_x = roi_x + x
                        img_y = roi_y + y

                        # Add border for processing
                        crop_x_start = max(0, img_x - BORDER_SIZE)
                        crop_y_start = max(0, img_y - BORDER_SIZE)
                        crop_x_end = min(full_w, img_x + TILE_SIZE + BORDER_SIZE)
                        crop_y_end = min(full_h, img_y + TILE_SIZE + BORDER_SIZE)

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
                        tiles.append(worker)

                job.tiles_to_process = len(tiles)
                for worker in tiles:
                    self.thread_pool.start(worker)
            else:
                self.previewUpdated.emit(
                    pix_bg, full_w, full_h, QtGui.QPixmap(), 0, 0, 0, 0
                )
        else:
            self.previewUpdated.emit(
                pix_bg, full_w, full_h, QtGui.QPixmap(), 0, 0, 0, 0
            )

    def _on_render_timer_timeout(self):
        self._is_rendering_locked = False
        if self._render_pending:
            self._process_pending_update()

    @QtCore.Slot(str, int, int, QtGui.QImage)
    def _on_tile_finished(self, job_id, tile_x, tile_y, image_data):
        if job_id in self._jobs:
            self._jobs[job_id].add_tile(tile_x, tile_y, image_data)

    def _on_job_finished(
        self, job_id, pix_bg, full_w, full_h, roi_x, roi_y, roi_w, roi_h, roi_pixmap
    ):
        if job_id == self._current_job_id:
            self.previewUpdated.emit(
                pix_bg, full_w, full_h, roi_pixmap, roi_x, roi_y, roi_w, roi_h
            )
        if job_id in self._jobs:
            del self._jobs[job_id]

    @QtCore.Slot(str)
    def _on_worker_error(self, error_message):
        print(f"Image processing error: {error_message}")
