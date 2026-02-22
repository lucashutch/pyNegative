import logging
import time

import numpy as np
from PySide6 import QtCore, QtGui

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
        self._current_settings_state_id = 0
        self._last_rendered_settings_state_id = 0
        self._last_settings = None

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.histogramUpdated.connect(self._on_histogram_updated)
        self.signals.tierGenerated.connect(self._on_tier_generated)
        self.signals.uneditedPixmapGenerated.connect(self._on_unedited_pixmap_generated)
        self.signals.error.connect(self._on_worker_error)
        self._current_image_id = 0

    def set_image(self, img_array):
        self.base_img_full = img_array
        self._unedited_img_full = img_array
        self._processing_params = {}

        # Dictionary of scales (0.5, 0.25, 0.125, 0.0625)
        self.tiers = {}

        self._tile_cache.clear()
        self._current_settings_state_id += 1
        self._current_render_state_id += 1

        if img_array is not None:
            h, w = img_array.shape[:2]
            # Create a unique ID for this image file so async ghosts drop properly
            self._current_image_id += 1
            # Asynchronous Pyramid Generation
            # Start background worker to generate tiers so UI isn't blocked
            worker = TierGeneratorWorker(
                self.signals, img_array, self._current_image_id
            )
            self.thread_pool.start(worker)
        else:
            self._current_image_id += 1
            self.previewUpdated.emit(QtGui.QPixmap(), 0, 0, 0.0, None, None, None, True)
            self.uneditedPixmapUpdated.emit(QtGui.QPixmap())

    @QtCore.Slot(QtGui.QPixmap, int)
    def _on_unedited_pixmap_generated(self, pixmap, image_id):
        if image_id == self._current_image_id:
            self.uneditedPixmapUpdated.emit(pixmap)

    @QtCore.Slot(float, object, int)
    def _on_tier_generated(self, scale, array, image_id):
        if image_id != self._current_image_id:
            return
        self.tiers[scale] = array
        # Once we have the first reliable downscaled tier, or the requested specific tier natively triggers it, we request a high-quality preview update
        if scale == 0.25 or scale == 0.1667 or len(self.tiers) == 1:
            # We don't emit uneditedPixmapUpdated here because TierGeneratorWorker
            # already emitted a fast 1:16 unedited pixmap.
            # We just request a full update so the background renderer uses the new better tier.
            self._tile_cache.clear()
            self._current_render_state_id += 1
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

        viewport_size = self._view_ref.viewport().size()
        v_w = viewport_size.width()
        is_fitting = getattr(self._view_ref, "_is_fitting", False)

        is_cropping = self._view_ref._crop_item.isVisible()
        target_w = self.base_img_full.shape[1] * zoom_scale if not is_fitting else v_w

        current_settings = self.get_current_settings()
        settings_changed = self._last_settings != current_settings
        zoom_changed = abs(zoom_scale - self._last_requested_zoom) > 0.05

        fit_crop_changed = (is_fitting != getattr(self, "_last_is_fitting", False)) or (
            is_cropping != getattr(self, "_last_is_cropping", False)
        )
        self._last_is_fitting = is_fitting
        self._last_is_cropping = is_cropping

        if settings_changed or zoom_changed or fit_crop_changed:
            self._current_render_state_id += 1
            self._tile_cache.clear()
            self._last_settings = current_settings
            self._last_requested_zoom = zoom_scale

        if settings_changed or fit_crop_changed:
            self._current_settings_state_id += 1

        self._current_request_id += 1

        viewport_rect = self._view_ref.viewport().rect()
        visible_poly = self._view_ref.mapToScene(viewport_rect)
        visible_rect = visible_poly.boundingRect()

        # Slight overscan for panning buffer (~5%)
        buf_w = visible_rect.width() * 0.05
        buf_h = visible_rect.height() * 0.05
        visible_rect.adjust(-buf_w, -buf_h, buf_w, buf_h)

        scene_rect = self._view_ref.sceneRect()
        sw = int(scene_rect.width())
        sh = int(scene_rect.height())

        # Bootstrap scene rect on first load if missing
        if sw <= 0 or sh <= 0:
            sw = self.base_img_full.shape[1]
            sh = self.base_img_full.shape[0]

        visible_rect = visible_rect.intersected(QtCore.QRectF(0, 0, sw, sh))

        # Dynamically calculate scene tile size so that chunks represent roughly 256x256 on the screen
        ideal_scale = 1.0
        if zoom_scale < 1.0:
            for scale in [0.0625, 0.125, 0.25, 0.5]:
                if scale >= zoom_scale:
                    ideal_scale = scale
                    break

        TILE_SIZE_SCENE = int(256 / ideal_scale)
        tx_min = int(visible_rect.x() // TILE_SIZE_SCENE)
        ty_min = int(visible_rect.y() // TILE_SIZE_SCENE)
        tx_max = int((visible_rect.x() + visible_rect.width()) // TILE_SIZE_SCENE)
        ty_max = int((visible_rect.y() + visible_rect.height()) // TILE_SIZE_SCENE)

        # Use sw/sh defined in bootstrap instead of pulling from scene again
        tx_max = min(tx_max, sw // TILE_SIZE_SCENE)
        ty_max = min(ty_max, sh // TILE_SIZE_SCENE)

        # Deduce the selected tier matching what workers will use
        available_scales = sorted(self.tiers.keys()) + [1.0]
        selected_scale = 1.0
        if zoom_scale >= 1.0:
            selected_scale = 1.0
        else:
            for scale in available_scales:
                if scale >= zoom_scale:
                    selected_scale = scale
                    break

        # Prevent 100% full-resolution processing queues when zoomed out drastically during initial file load
        # Wait for any async scaled tiers to become available before processing grid chunks.
        if zoom_scale <= 0.5 and selected_scale == 1.0 and len(self.tiers) == 0:
            logger.debug(
                f"Delaying worker threads: awaiting async downscale tiers for zoom ({zoom_scale:.4f})."
            )
            self._is_rendering_locked = False
            return

        needs_lowres = True
        workers_queued = 0
        workers_pending = 0

        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                tile_key = (tx, ty, TILE_SIZE_SCENE)

                # If a tile is already processing or done specifically for this render_state, skip it.
                # If the state changed (e.g. slider dragged), we want to submit a new worker
                # for this tile_key, overwriting the old geometry string pending.
                current_state = self._tile_cache.get(tile_key)
                if current_state == f"pending_{self._current_render_state_id}":
                    workers_pending += 1
                    continue
                elif current_state == f"done_{self._current_render_state_id}":
                    continue

                self._tile_cache[tile_key] = f"pending_{self._current_render_state_id}"

                vx = tx * TILE_SIZE_SCENE
                vy = ty * TILE_SIZE_SCENE
                vw = TILE_SIZE_SCENE
                vh = TILE_SIZE_SCENE

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
                    settings_state_id=self._current_settings_state_id,
                )
                needs_lowres = False
                workers_queued += 1
                self.thread_pool.start(worker)

        v_w = self._view_ref.viewport().rect().width()
        v_h = self._view_ref.viewport().rect().height()

        logger.info(
            f"Viewport: {v_w}x{v_h} | Zoom Scale: {zoom_scale:.4f} | Tier: {selected_scale} | Queued {workers_queued} chunks (256x256)"
        )

        if workers_queued == 0 and workers_pending == 0:
            self._is_rendering_locked = False

    def _measure_and_emit_perf(self):
        elapsed_ms = (time.perf_counter() - self.perf_start_time) * 1000
        self.performanceMeasured.emit(elapsed_ms)

    @QtCore.Slot(QtGui.QPixmap, int, int, float, object, object, int, object, int, int)
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
        settings_state_id,
    ):
        self._is_rendering_locked = False

        if render_state_id < self._current_render_state_id:
            # Optionally check if there are pending updates for rapid slider moves
            if self._render_pending:
                self._process_pending_update()
            return

        if render_state_id > self._last_rendered_state_id:
            self._last_rendered_state_id = render_state_id

        clear_tiles = False
        if settings_state_id > self._last_rendered_settings_state_id:
            # We explicitly do NOT clear_tiles here anymore.
            # Doing so obliterates the high-res view and causes a blurry 0.25x flash.
            # The old tiles will simply remain on screen until the workers replace them.
            self._last_rendered_settings_state_id = settings_state_id

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
            self._tile_cache[tile_key] = f"done_{render_state_id}"

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
