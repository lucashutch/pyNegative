import numpy as np
from PySide6 import QtCore, QtGui
import time
import logging
from .pipeline.worker import (
    ImageProcessorSignals,
    ImageProcessorWorker,
    TierGeneratorWorker,
)

logger = logging.getLogger(__name__)


class ImageProcessingPipeline(QtCore.QObject):
    previewUpdated = QtCore.Signal(
        QtGui.QPixmap, int, int, float, object, object, object, bool
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
        self.lens_info = None

        self._current_request_id = 0
        self._last_processed_id = -1
        self._last_zoom_scale = 1.0
        self._last_requested_zoom = 1.0

        self._tile_cache = {}
        self._current_render_state_id = 0
        self._last_rendered_state_id = 0
        self._last_settings = None

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.histogramUpdated.connect(self._on_histogram_updated)
        self.signals.tierGenerated.connect(self._on_tier_generated)
        self.signals.uneditedPixmapGenerated.connect(self.uneditedPixmapUpdated.emit)
        self.signals.error.connect(self._on_worker_error)

    def set_image(self, img_array):
        self.base_img_full = img_array
        self._unedited_img_full = img_array
        self._processing_params = {}

        # Dictionary of scales (0.5, 0.25, 0.125, 0.0625)
        self.tiers = {}

        if img_array is not None:
            # Asynchronous Pyramid Generation
            # Start background worker to generate tiers so UI isn't blocked
            worker = TierGeneratorWorker(self.signals, img_array)
            self.thread_pool.start(worker)
        else:
            self.uneditedPixmapUpdated.emit(QtGui.QPixmap())

    @QtCore.Slot(float, object)
    def _on_tier_generated(self, scale, array):
        self.tiers[scale] = array
        # Once we have the 1:4 tier (0.25) or better, we can request a high-quality preview update
        if scale == 0.25:
            # We don't emit uneditedPixmapUpdated here because TierGeneratorWorker
            # already emitted a fast 1:16 unedited pixmap.
            # We just request a full update so the background renderer uses the new better tier.
            self.request_update()

    def set_view_reference(self, view):
        self._view_ref = view

    def get_unedited_pixmap(self, max_width: int = 0) -> QtGui.QPixmap:
        if self._unedited_img_full is None:
            return QtGui.QPixmap()
        try:
            # OPTIMIZATION: pick the best tier if max_width is specified, or cap if 0
            if max_width <= 0:
                max_width = 2048  # Default cap for unedited preview

            source = self._unedited_img_full
            available_scales = sorted(self.tiers.keys())
            for s in available_scales:
                tier = self.tiers[s]
                if tier.shape[1] >= max_width:
                    source = tier
                    break

            # We know the pipeline uses float32 0-1 range by convention
            img_uint8 = (np.clip(source, 0, 1) * 255).astype(np.uint8)
            img_rgb = img_uint8
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

    def set_lens_info(self, info):
        if self.lens_info != info:
            logger.debug("Lens info changed")
            self.lens_info = info

    def set_processing_params(self, **kwargs):
        heavy_keys = {"de_haze", "denoise_luma", "denoise_chroma", "sharpen_value"}

        changed = False
        for k, v in kwargs.items():
            if self._processing_params.get(k) != v:
                self._processing_params[k] = v
                changed = True
                if k in heavy_keys:
                    self._last_heavy_adjusted = k

        if changed:
            self.request_update()

    def get_current_settings(self):
        return self._processing_params.copy()

    def request_update(self):
        if self.base_img_full is None:
            return

        self._render_pending = True

        if not self._is_rendering_locked:
            # Rate limit/Debounce: wait a few ms to catch rapid slider movements
            # 16ms = ~60fps, 33ms = ~30fps.
            if not self.render_timer.isActive():
                self.render_timer.start(20)

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

        # Capture viewport state in UI thread
        zoom_scale = self._view_ref.transform().m11()
        self._last_requested_zoom = zoom_scale

        viewport_size = self._view_ref.viewport().size()
        v_w = viewport_size.width()
        is_fitting = getattr(self._view_ref, "_is_fitting", False)

        is_cropping = self._view_ref._crop_item.isVisible()
        target_w = self.base_img_full.shape[1] * zoom_scale if not is_fitting else v_w

        current_settings = self.get_current_settings()
        settings_changed = self._last_settings != current_settings
        zoom_changed = abs(zoom_scale - self._last_requested_zoom) > 0.05

        if settings_changed or zoom_changed or is_fitting or is_cropping:
            self._current_render_state_id += 1
            self._tile_cache.clear()
            self._last_settings = current_settings
            self._last_requested_zoom = zoom_scale

        self._current_request_id += 1

        if is_fitting or is_cropping:
            worker = ImageProcessorWorker(
                self.signals,
                self.base_img_full,
                self.tiers,
                current_settings,
                self._current_request_id,
                zoom_scale=zoom_scale,
                is_fitting=is_fitting,
                calculate_histogram=self.histogram_enabled,
                last_heavy_adjusted=self._last_heavy_adjusted,
                lens_info=self.lens_info,
                target_on_screen_width=target_w,
                visible_scene_rect=None,
                tile_key=None,
                render_state_id=self._current_render_state_id,
                calculate_lowres=True,
            )
            self.thread_pool.start(worker)
            return

        viewport_rect = self._view_ref.viewport().rect()
        visible_poly = self._view_ref.mapToScene(viewport_rect)
        visible_rect = visible_poly.boundingRect()

        # Slight overscan for panning buffer (~5%)
        buf_w = visible_rect.width() * 0.05
        buf_h = visible_rect.height() * 0.05
        visible_rect.adjust(-buf_w, -buf_h, buf_w, buf_h)

        scene_rect = self._view_ref.sceneRect()
        visible_rect = visible_rect.intersected(scene_rect)

        TILE_SIZE = 256
        tx_min = int(visible_rect.x() // TILE_SIZE)
        ty_min = int(visible_rect.y() // TILE_SIZE)
        tx_max = int((visible_rect.x() + visible_rect.width()) // TILE_SIZE)
        ty_max = int((visible_rect.y() + visible_rect.height()) // TILE_SIZE)

        sw = int(scene_rect.width())
        sh = int(scene_rect.height())

        tx_max = min(tx_max, sw // TILE_SIZE)
        ty_max = min(ty_max, sh // TILE_SIZE)

        needs_lowres = True

        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                tile_key = (tx, ty)
                if tile_key in self._tile_cache:
                    continue

                self._tile_cache[tile_key] = "pending"

                vx = tx * TILE_SIZE
                vy = ty * TILE_SIZE
                vw = TILE_SIZE
                vh = TILE_SIZE

                vw = min(vw, sw - vx)
                vh = min(vh, sh - vy)

                if vw <= 0 or vh <= 0:
                    continue

                worker = ImageProcessorWorker(
                    self.signals,
                    self.base_img_full,
                    self.tiers,
                    current_settings,
                    self._current_request_id,
                    zoom_scale=zoom_scale,
                    is_fitting=is_fitting,
                    calculate_histogram=self.histogram_enabled and needs_lowres,
                    last_heavy_adjusted=self._last_heavy_adjusted,
                    lens_info=self.lens_info,
                    target_on_screen_width=target_w,
                    visible_scene_rect=(int(vx), int(vy), int(vw), int(vh)),
                    tile_key=tile_key,
                    render_state_id=self._current_render_state_id,
                    calculate_lowres=needs_lowres,
                )
                needs_lowres = False
                self.thread_pool.start(worker)

    def _measure_and_emit_perf(self):
        elapsed_ms = (time.perf_counter() - self.perf_start_time) * 1000
        self.performanceMeasured.emit(elapsed_ms)

    @QtCore.Slot(QtGui.QPixmap, int, int, float, object, object, int, object, int)
    def _on_worker_finished(
        self,
        pix_bg,
        full_w,
        full_h,
        rotation,
        visible_scene_rect,
        bg_lowres_pix,
        request_id,
        tile_key,
        render_state_id,
    ):
        self._is_rendering_locked = False

        if render_state_id < self._current_render_state_id:
            # Optionally check if there are pending updates for rapid slider moves
            if self._render_pending:
                self._process_pending_update()
            return

        clear_tiles = False
        if render_state_id > self._last_rendered_state_id:
            clear_tiles = True
            self._last_rendered_state_id = render_state_id

        self._last_processed_id = request_id
        self._last_zoom_scale = self._last_requested_zoom

        self.previewUpdated.emit(
            pix_bg,
            full_w,
            full_h,
            rotation,
            visible_scene_rect,
            bg_lowres_pix,
            tile_key,
            clear_tiles,
        )
        if tile_key is not None:
            self._tile_cache[tile_key] = "done"

        self.editedPixmapUpdated.emit(pix_bg)
        self._measure_and_emit_perf()

        if self._render_pending:
            # If there's another update pending, don't start it IMMEDIATELY.
            # Give the UI thread a tiny slice of time to handle input events.
            self.render_timer.start(5)

    @QtCore.Slot(dict, int)
    def _on_histogram_updated(self, hist_data, request_id):
        # Drop stale results
        if request_id < self._current_request_id:
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
