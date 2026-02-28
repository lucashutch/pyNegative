import logging
import time

import numpy as np
from PySide6 import QtCore, QtGui

import cv2

from .pipeline.worker import (
    ImageProcessorSignals,
    ImageProcessorWorker,
    TierGeneratorWorker,
)
from .pipeline.stages import (
    process_denoise_stage,
    process_heavy_stage,
    get_fused_geometry,
    resolve_vignette_params,
    apply_fused_remap,
)
from .. import core as pynegative

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
        self._lowres_idle_timer = QtCore.QTimer()
        self._lowres_idle_timer.setSingleShot(True)
        self._lowres_idle_timer.timeout.connect(self._on_lowres_idle_timeout)
        self._lowres_idle_ms = 500
        self._render_pending = False
        self._active_workers = 0
        self._shutting_down = False
        self.base_img_full = None
        self.base_img_half = None
        self.base_img_quarter = None
        self.base_img_preview = None
        self._unedited_img_full = None
        self._processing_params = {}
        self._last_heavy_adjusted = "de_haze"
        self._view_ref = None
        self.histogram_enabled = False
        self.lens_info = None

        self._current_request_id = 0
        self._last_processed_id = -1
        self._last_zoom_scale = 1.0
        self._last_requested_zoom = 1.0

        self._current_render_state_id = 0
        self._last_rendered_state_id = 0
        self._current_settings_state_id = 0
        self._last_rendered_settings_state_id = 0
        self._last_settings = None
        self._last_roi_signature = None

        self.signals = ImageProcessorSignals()
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.histogramUpdated.connect(self._on_histogram_updated)
        self.signals.tierGenerated.connect(self._on_tier_generated)
        self.signals.uneditedPixmapGenerated.connect(self._on_unedited_pixmap_generated)
        self.signals.error.connect(self._on_worker_error)
        self._current_image_id = 0

        # Pre-computed dehaze atmospheric light (shared across updates)
        self._dehaze_atmos_light = None
        self._dehaze_atmos_settings_id = -1

        # ROI render cache: list of recent renders (newest first, max _CACHE_SIZE).
        # On cache hit, crop the stored pixmap instead of launching a worker.
        _CACHE_SIZE = 16
        self._render_cache: list = []
        self._cache_size = _CACHE_SIZE
        self._last_queued_scale = 1.0
        self._defer_lowres_until_idle = False

        # Per-request perf start times so concurrent renders don't overwrite each other.
        self._perf_start_times: dict[int, float] = {}

    def set_image(self, img_array):
        self.base_img_full = img_array
        self._unedited_img_full = img_array
        self._processing_params = {}

        # Dictionary of scales (0.5, 0.25, 0.125, 0.0625)
        self.tiers = {}

        self._current_settings_state_id += 1
        self._current_render_state_id += 1
        self._dehaze_atmos_light = None
        self._dehaze_atmos_settings_id = -1
        self._last_roi_signature = None
        self._render_cache = []
        self._perf_start_times = {}
        self._defer_lowres_until_idle = False
        self._lowres_idle_timer.stop()

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
            self._current_render_state_id += 1
            self._last_roi_signature = None
            self._render_cache = []
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

    def render_snapshot_pixmap(
        self, settings: dict, max_width: int = 1024
    ) -> QtGui.QPixmap:
        """Render arbitrary settings on the current image synchronously.

        Uses the smallest tier that meets *max_width* and runs the full
        processing pipeline (preprocess -> denoise -> geometry -> heavy ->
        tonemap -> defringe) in the calling thread.  Returns a QPixmap
        suitable for the comparison overlay.
        """
        if self.base_img_full is None:
            return QtGui.QPixmap()

        try:
            full_h, full_w = self.base_img_full.shape[:2]

            # Pick the smallest tier whose width >= max_width
            source = self.base_img_full
            selected_scale = 1.0
            for s in sorted(self.tiers.keys()):
                tier = self.tiers.get(s)
                if tier is not None and tier.shape[1] >= max_width:
                    source = tier
                    selected_scale = s
                    break

            h_src, w_src = source.shape[:2]
            res_key = f"tier_{selected_scale}"

            # --- Extract settings ---
            rotate_val = settings.get("rotation", 0.0)
            flip_h = settings.get("flip_h", False)
            flip_v = settings.get("flip_v", False)
            crop_val = settings.get("crop", None)

            heavy_params = {
                "de_haze": settings.get("de_haze", 0) / 50.0,
                "denoise_luma": settings.get("denoise_luma", 0),
                "denoise_chroma": settings.get("denoise_chroma", 0) * 2,
                "denoise_method": settings.get("denoise_method", "Bilateral"),
                "sharpen_value": settings.get("sharpen_value", 0),
                "sharpen_radius": settings.get("sharpen_radius", 0.5),
                "sharpen_percent": settings.get("sharpen_percent", 0.0),
            }

            preprocess_settings = {
                "temperature": settings.get("temperature", 0.0),
                "tint": settings.get("tint", 0.0),
                "exposure": settings.get("exposure", 0.0),
            }

            tone_map_settings = {
                "contrast": settings.get("contrast", 0.0),
                "blacks": settings.get("blacks", 0.0),
                "whites": settings.get("whites", 0.0),
                "shadows": settings.get("shadows", 0.0),
                "highlights": settings.get("highlights", 0.0),
                "saturation": settings.get("saturation", 0.0),
            }

            # --- Vignette ---
            vig_params = resolve_vignette_params(
                settings, self.lens_info, roi_offset=None, full_size=(w_src, h_src)
            )
            vig_k1, vig_k2, vig_k3, vig_cx, vig_cy, vig_fw, vig_fh = vig_params

            # --- Preprocess ---
            preprocessed = pynegative.apply_preprocess(
                source,
                temperature=preprocess_settings["temperature"],
                tint=preprocess_settings["tint"],
                exposure=preprocess_settings["exposure"],
                vignette_k1=vig_k1,
                vignette_k2=vig_k2,
                vignette_k3=vig_k3,
                vignette_cx=vig_cx,
                vignette_cy=vig_cy,
                full_width=vig_fw,
                full_height=vig_fh,
            )

            # --- Denoise ---
            denoised = process_denoise_stage(
                preprocessed, res_key, heavy_params, selected_scale
            )

            # --- Geometry ---
            fused_maps, o_w, o_h, _ = get_fused_geometry(
                settings,
                self.lens_info,
                w_src,
                h_src,
                rotate_val,
                crop_val,
                flip_h,
                flip_v,
                ts_roi=selected_scale,
            )

            img_dest = apply_fused_remap(
                denoised, fused_maps, o_w, o_h, cv2.INTER_LINEAR
            )

            # --- Heavy stage ---
            dehaze_val = settings.get("de_haze", 0)
            atmos_light = None
            if dehaze_val and float(dehaze_val) > 0:
                atmos_light = pynegative.estimate_atmospheric_light(source)

            processed = process_heavy_stage(
                img_dest,
                res_key,
                heavy_params,
                selected_scale,
                "de_haze",
                atmos_light,
            )

            # --- Tonemap + defringe ---
            output, _ = pynegative.apply_tone_map(
                processed, **tone_map_settings, calculate_stats=False
            )
            output = pynegative.apply_defringe(output, settings)

            # --- Convert to QPixmap ---
            img_uint8 = pynegative.float32_to_uint8(output)
            h_out, w_out = img_uint8.shape[:2]
            qimage = QtGui.QImage(
                img_uint8.data, w_out, h_out, 3 * w_out, QtGui.QImage.Format_RGB888
            )
            return QtGui.QPixmap.fromImage(qimage)
        except Exception:
            logger.error("Failed to render snapshot pixmap", exc_info=True)
            return QtGui.QPixmap()

    def set_histogram_enabled(self, enabled):
        self.histogram_enabled = enabled
        if enabled:
            # Force a fresh pass so at least one worker recomputes histogram
            # even when all current tiles are already marked done.
            self._current_render_state_id += 1
            self._last_roi_signature = None
            self._render_cache = []
            self.request_update()

    def set_lens_info(self, info):
        if self.lens_info != info:
            logger.debug("Lens info changed")
            self.lens_info = info

    def set_processing_params(self, **kwargs):
        heavy_keys = {"de_haze", "denoise_luma", "denoise_chroma", "sharpen_value"}
        geometry_keys = {
            "rotation",
            "crop",
            "flip_h",
            "flip_v",
            "lens_distortion",
            "lens_ca",
            "lens_autocrop",
            "lens_enabled",
        }

        changed = False
        geometry_changed = False
        for k, v in kwargs.items():
            if self._processing_params.get(k) != v:
                self._processing_params[k] = v
                changed = True
                if k in heavy_keys:
                    self._last_heavy_adjusted = k
                if k in geometry_keys:
                    geometry_changed = True

        if changed:
            # Keep render state stable for pure settings drags so intermediate
            # worker results can still be displayed while sliders move.
            # Settings state still advances to invalidate ROI cache entries.
            if geometry_changed:
                self._current_render_state_id += 1
            self._current_settings_state_id += 1
            self._last_roi_signature = None
            self._render_cache = []
            self._defer_lowres_until_idle = True
            self._lowres_idle_timer.start(self._lowres_idle_ms)
            self.request_update()

    def get_current_settings(self):
        return self._processing_params.copy()

    def request_update(self):
        if self.base_img_full is None or self._shutting_down:
            return

        self._render_pending = True

        # Always ensure a timer is running when a render is pending.
        # If workers are active, use a slightly longer delay so their
        # stale results have time to drop before we queue the next pass.
        if not self.render_timer.isActive():
            delay = 20 if self._active_workers == 0 else 50
            self.render_timer.start(delay)

    def shutdown(self):
        """Stop all pending work gracefully before app close."""
        self._shutting_down = True
        self.render_timer.stop()
        self._lowres_idle_timer.stop()
        self._render_pending = False

    def _on_lowres_idle_timeout(self):
        if self._shutting_down or self.base_img_full is None:
            return
        self._defer_lowres_until_idle = False
        self.request_update()

    def _on_render_timer_timeout(self):
        self._process_pending_update()

    def _process_pending_update(self):
        if (
            not self._render_pending
            or self.base_img_full is None
            or self._view_ref is None
            or self._shutting_down
        ):
            return

        self._render_pending = False
        perf_queue_start = time.perf_counter()

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
            if zoom_changed or fit_crop_changed:
                self._current_render_state_id += 1
            self._last_settings = current_settings
            self._last_requested_zoom = zoom_scale
            self._last_roi_signature = None

            # Only drop the cache when tier changes or settings/geometry change.
            # A zoom change within the same tier leaves cached pixmaps valid
            # because the pixmap-to-scene-coord scale is tier-based, not zoom-based.
            if settings_changed or fit_crop_changed:
                self._render_cache = []
            elif zoom_changed:
                new_tier = 1.0
                if zoom_scale < 1.0:
                    for s in sorted(self.tiers.keys()) + [1.0]:
                        if s >= zoom_scale:
                            new_tier = s
                            break
                if new_tier != self._last_queued_scale:
                    self._render_cache = []

        if settings_changed or fit_crop_changed:
            self._current_settings_state_id += 1

        self._current_request_id += 1

        viewport_rect = self._view_ref.viewport().rect()
        visible_poly = self._view_ref.mapToScene(viewport_rect)
        visible_rect = visible_poly.boundingRect()

        # Overscan for panning buffer (~10%)
        buf_w = visible_rect.width() * 0.10
        buf_h = visible_rect.height() * 0.10
        visible_rect.adjust(-buf_w, -buf_h, buf_w, buf_h)

        scene_rect = self._view_ref.sceneRect()
        sw = int(scene_rect.width())
        sh = int(scene_rect.height())

        # Bootstrap scene rect on first load if missing
        if sw <= 0 or sh <= 0:
            sw = self.base_img_full.shape[1]
            sh = self.base_img_full.shape[0]

        visible_rect = visible_rect.intersected(QtCore.QRectF(0, 0, sw, sh))

        roi_rect = QtCore.QRect(
            int(np.floor(visible_rect.x())),
            int(np.floor(visible_rect.y())),
            max(1, int(np.ceil(visible_rect.width()))),
            max(1, int(np.ceil(visible_rect.height()))),
        )
        roi_rect = roi_rect.intersected(QtCore.QRect(0, 0, sw, sh))
        if roi_rect.width() <= 0 or roi_rect.height() <= 0:
            return

        # ROI cache: iterate recent renders (newest first) and serve a crop if the
        # new viewport falls within a previously rendered region at the same settings.
        rx, ry, rw, rh = (
            roi_rect.x(),
            roi_rect.y(),
            roi_rect.width(),
            roi_rect.height(),
        )
        for entry in self._render_cache:
            if entry["settings_id"] != self._current_settings_state_id:
                continue
            cx, cy, cw, ch = entry["scene_rect"]
            # Cache hits must be fully covered by the cached scene rect.
            if rx >= cx and ry >= cy and rx + rw <= cx + cw and ry + rh <= cy + ch:
                pix = entry["pixmap"]
                pix_w = pix.width()
                pix_h = pix.height()
                if pix_w <= 0 or pix_h <= 0 or cw <= 0 or ch <= 0:
                    continue

                # Map scene-space ROI to pixmap-space via rect ratios.
                src_x = int(round((rx - cx) * pix_w / cw))
                src_y = int(round((ry - cy) * pix_h / ch))
                src_w = int(round(rw * pix_w / cw))
                src_h = int(round(rh * pix_h / ch))

                src_x = max(0, min(pix_w - 1, src_x))
                src_y = max(0, min(pix_h - 1, src_y))
                src_w = max(1, min(pix_w - src_x, src_w))
                src_h = max(1, min(pix_h - src_y, src_h))

                cropped = pix.copy(src_x, src_y, src_w, src_h)
                logger.debug(f"🗃️  Cache hit | ROI {rw}x{rh} within cached {cw}x{ch}")
                self._last_roi_signature = (
                    rx,
                    ry,
                    rw,
                    rh,
                    round(zoom_scale, 3),
                    int(is_fitting),
                    int(is_cropping),
                    self._current_settings_state_id,
                )
                self.previewUpdated.emit(
                    cropped,
                    entry["full_w"],
                    entry["full_h"],
                    entry["rotation"],
                    (rx, ry, rw, rh),
                    None,
                    None,
                    False,
                )
                return

        # Never queue a second worker while one is already in-flight.
        # The completion handler will re-schedule via _render_pending if needed.
        if self._active_workers > 0:
            self._render_pending = True
            return

        roi_signature = (
            roi_rect.x(),
            roi_rect.y(),
            roi_rect.width(),
            roi_rect.height(),
            round(zoom_scale, 3),
            int(is_fitting),
            int(is_cropping),
            self._current_settings_state_id,
        )
        if roi_signature == self._last_roi_signature:
            return
        self._last_roi_signature = roi_signature

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

        self._last_queued_scale = selected_scale

        # Prevent 100% full-resolution processing queues when zoomed out drastically during initial file load
        # Wait for any async scaled tiers to become available before processing grid chunks.
        if zoom_scale <= 0.5 and selected_scale == 1.0 and len(self.tiers) == 0:
            logger.debug(
                f"Delaying worker threads: awaiting async downscale tiers for zoom ({zoom_scale:.4f})."
            )
            return

        # Pre-compute dehaze atmospheric light from smallest available tier
        # so every tile uses the same value (prevents per-tile colour tints).
        dehaze_val = current_settings.get("de_haze", 0)
        atmos_light = None
        if dehaze_val and float(dehaze_val) > 0:
            if self._dehaze_atmos_settings_id != self._current_settings_state_id:
                smallest_tier = None
                for s in sorted(self.tiers.keys()):
                    if self.tiers[s] is not None:
                        smallest_tier = self.tiers[s]
                        break
                if smallest_tier is None:
                    smallest_tier = self.base_img_full
                self._dehaze_atmos_light = pynegative.estimate_atmospheric_light(
                    smallest_tier
                )
                self._dehaze_atmos_settings_id = self._current_settings_state_id
                logger.debug(
                    f"Dehaze atmospheric light computed: {self._dehaze_atmos_light}"
                )
            atmos_light = self._dehaze_atmos_light

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
            visible_scene_rect=(
                int(roi_rect.x()),
                int(roi_rect.y()),
                int(roi_rect.width()),
                int(roi_rect.height()),
            ),
            tile_key=None,
            render_state_id=self._current_render_state_id,
            calculate_lowres=not self._defer_lowres_until_idle,
            settings_state_id=self._current_settings_state_id,
            dehaze_atmospheric_light=atmos_light,
        )
        self._active_workers += 1
        self._perf_start_times[self._current_request_id] = perf_queue_start
        self.thread_pool.start(worker)

        v_w = self._view_ref.viewport().rect().width()
        v_h = self._view_ref.viewport().rect().height()

        perf_schedule_time = (time.perf_counter() - perf_queue_start) * 1000
        roi_pixels = roi_rect.width() * roi_rect.height()
        # Compare ROI area against the expected visible scene area (viewport mapped through zoom).
        # At zoom > 1 the scene area is smaller than screen pixels, so we compare in scene-space.
        scene_viewport_px = (v_w * v_h) / max(zoom_scale * zoom_scale, 1e-9)
        overscan_pct = (
            (roi_pixels - scene_viewport_px) / scene_viewport_px * 100
            if scene_viewport_px > 0
            else 0
        )

        logger.info(
            f"🎨 Render Start | Viewport: {v_w}x{v_h} | "
            f"ROI: {roi_rect.width()}x{roi_rect.height()} (+{overscan_pct:.0f}% overscan) | "
            f"Zoom: {zoom_scale:.3f} | Tier: {selected_scale} | Schedule: {perf_schedule_time:.1f}ms"
        )

    def _measure_and_emit_perf(self, request_id: int):
        start = self._perf_start_times.pop(request_id, None)
        if start is None:
            return
        # Evict start times for older requests that were dropped (stale).
        stale = [k for k in self._perf_start_times if k < request_id]
        for k in stale:
            del self._perf_start_times[k]
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"✅ Render Complete | Total: {elapsed_ms:.1f}ms")
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
        self._active_workers = max(0, self._active_workers - 1)

        if self._shutting_down:
            return

        if render_state_id < self._current_render_state_id:
            # Stale result — ensure any pending render gets scheduled.
            # If all workers are done we can fire immediately; otherwise
            # start a timer so the pending render isn't lost.
            if self._render_pending and not self.render_timer.isActive():
                self.render_timer.start(5 if self._active_workers == 0 else 30)
            return

        if render_state_id > self._last_rendered_state_id:
            self._last_rendered_state_id = render_state_id

        clear_tiles = False
        if settings_state_id > self._last_rendered_settings_state_id:
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
            None,
            clear_tiles,
        )

        if visible_scene_rect is not None:
            entry = {
                "pixmap": pix_bg,
                "scene_rect": visible_scene_rect,
                "scale": self._last_queued_scale,
                "settings_id": settings_state_id,
                "full_w": full_w,
                "full_h": full_h,
                "rotation": rotation,
            }
            self._render_cache.insert(0, entry)
            del self._render_cache[self._cache_size :]

        self.editedPixmapUpdated.emit(pix_bg)
        self._measure_and_emit_perf(request_id)

        if self._render_pending and not self.render_timer.isActive():
            self.render_timer.start(5 if self._active_workers == 0 else 30)

    @QtCore.Slot(dict, int)
    def _on_histogram_updated(self, hist_data, request_id):
        # Drop only results older than the latest rendered request.
        # Using _current_request_id is too aggressive during rapid queueing
        # and can discard valid histogram payloads before display.
        if request_id < self._last_processed_id:
            return
        self.histogramUpdated.emit(hist_data)

    @QtCore.Slot(str, int)
    def _on_worker_error(self, error_message, request_id):
        self._active_workers = max(0, self._active_workers - 1)
        if self._shutting_down:
            return
        if self._active_workers == 0 and self._render_pending:
            self.render_timer.start(5)
        if request_id < self._last_processed_id:
            return
        print(f"Image processing error (ID {request_id}): {error_message}")
