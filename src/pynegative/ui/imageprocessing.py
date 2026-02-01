import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui
from .. import core as pynegative


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int)
    error = QtCore.Signal(str)


class ImageProcessorWorker(QtCore.QRunnable):
    """Worker to process image adjustments in a background thread."""

    def __init__(self, signals, view_ref, base_img_full, settings):
        super().__init__()
        self.signals = signals
        self._view_ref = view_ref
        self.base_img_full = base_img_full
        self.settings = settings
        self._base_img_uint8 = None  # Cached uint8 version for resizing

    def run(self):
        """Execute the image processing."""
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
            if self._view_ref is None:
                return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0
            zoom_scale = self._view_ref.transform().m11()
            viewport = self._view_ref.viewport()
            if viewport is None:
                return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0
            vw, vh = (
                viewport.width(),
                viewport.height(),
            )
            if vw <= 0 or vh <= 0:
                return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0
        except (AttributeError, RuntimeError):
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0

        fit_scale = 1.0
        if vw > 0 and vh > 0:
            fit_scale = min(vw / full_w, vh / full_h)

        if self._view_ref is None:
            return QtGui.QPixmap(), 0, 0, QtGui.QPixmap(), 0, 0, 0, 0
        is_fitting = getattr(self._view_ref, "_is_fitting", False)
        is_zoomed_in = not is_fitting and (
            zoom_scale > fit_scale * 1.01 or zoom_scale > 0.99
        )

        if self._base_img_uint8 is None:
            self._base_img_uint8 = (self.base_img_full * 255).astype(np.uint8)

        scale = 1500 / max(full_h, full_w)
        target_h, target_w = int(full_h * scale), int(full_w * scale)

        # This check needs to be adapted for the worker context or removed if cache is not shared
        # For now, we regenerate it each time in the worker.
        temp_pil = Image.fromarray(self._base_img_uint8)
        temp_pil = temp_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_render_base = np.array(temp_pil).astype(np.float32) / 255.0

        pix_bg = QtGui.QPixmap()
        try:
            # Create a new dictionary for tone map settings only
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
            processed_bg *= 255
            pil_bg = Image.fromarray(processed_bg.astype(np.uint8))
            qimg_bg = ImageQt.ImageQt(pil_bg)
            pix_bg = QtGui.QPixmap.fromImage(qimg_bg)
        except Exception:
            pass

        pix_roi, roi_x, roi_y, roi_w, roi_h = QtGui.QPixmap(), 0, 0, 0, 0
        if is_zoomed_in and self._view_ref is not None:
            try:
                viewport_rect = self._view_ref.viewport().rect()
                if viewport_rect is not None:
                    roi = self._view_ref.mapToScene(viewport_rect).boundingRect()
                    ix_min = max(0, int(roi.left()))
                    ix_max = min(full_w, int(roi.right()))
                    iy_min = max(0, int(roi.top()))
                    iy_max = min(full_h, int(roi.bottom()))

                    if ix_max > ix_min and iy_max > iy_min:
                        rw = ix_max - ix_min
                        rh = iy_max - iy_min

                        if rw > 10 and rh > 10:
                            MAX_REALTIME_PIXELS = 1_500_000
                            current_pixels = rw * rh

                            crop = self.base_img_full[iy_min:iy_max, ix_min:ix_max]

                            if current_pixels > MAX_REALTIME_PIXELS:
                                p_scale = (MAX_REALTIME_PIXELS / current_pixels) ** 0.5
                                p_w, p_h = int(rw * p_scale), int(rh * p_scale)

                                pil_crop = Image.fromarray(
                                    (crop * 255).astype(np.uint8)
                                )
                                pil_crop = pil_crop.resize(
                                    (p_w, p_h), Image.Resampling.BILINEAR
                                )
                                crop_to_proc = (
                                    np.array(pil_crop).astype(np.float32) / 255.0
                                )
                            else:
                                crop_to_proc = crop

                            processed_roi, _ = pynegative.apply_tone_map(
                                crop_to_proc, **tone_map_settings
                            )
                            processed_roi *= 255
                            pil_roi = Image.fromarray(processed_roi.astype(np.uint8))

                            if self.settings.get("sharpen_value", 0) > 0:
                                pil_roi = pynegative.sharpen_image(
                                    pil_roi,
                                    self.settings["sharpen_radius"],
                                    self.settings["sharpen_percent"],
                                    "High Quality",
                                )

                            if self.settings.get("de_noise", 0) > 0:
                                pil_roi = pynegative.de_noise_image(
                                    pil_roi,
                                    self.settings["de_noise"],
                                    "High Quality",
                                )

                            pix_roi = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_roi))
                            roi_x, roi_y = ix_min, iy_min
                            roi_w, roi_h = rw, rh
            except (AttributeError, RuntimeError):
                pass

        return pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h


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

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.error.connect(self._on_worker_error)

        self.base_img_full = None
        self._processing_params = {}
        self._view_ref = None

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
        if not self._render_pending or self.base_img_full is None:
            return

        self._render_pending = False
        self._is_rendering_locked = True
        self.render_timer.start(33)

        worker = ImageProcessorWorker(
            self.signals,
            self._view_ref,
            self.base_img_full,
            self.get_current_settings(),
        )
        self.thread_pool.start(worker)

    def _on_render_timer_timeout(self):
        self._is_rendering_locked = False
        if self._render_pending:
            self._process_pending_update()

    @QtCore.Slot(QtGui.QPixmap, int, int, QtGui.QPixmap, int, int, int, int)
    def _on_worker_finished(
        self, pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
    ):
        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )

    @QtCore.Slot(str)
    def _on_worker_error(self, error_message):
        # In a real app, you'd want to log this or show it to the user
        print(f"Image processing error: {error_message}")
