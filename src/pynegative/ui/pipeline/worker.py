import logging
import time

import cv2
import numpy as np
from PySide6 import QtCore, QtGui

from ... import core as pynegative
from ...processing.geometry import GeometryResolver

from .stages import (
    process_denoise_stage,
    process_heavy_stage,
    get_fused_geometry,
    resolve_vignette_params,
    apply_fused_remap,
    calculate_histograms,
)

logger = logging.getLogger(__name__)


class ImageProcessorSignals(QtCore.QObject):
    """Signals for the image processing worker."""

    finished = QtCore.Signal(
        QtGui.QPixmap, int, int, float, object, object, int, object, int, int
    )
    histogramUpdated = QtCore.Signal(dict, int)
    tierGenerated = QtCore.Signal(float, object, int)  # scale, array, image_id
    uneditedPixmapGenerated = QtCore.Signal(QtGui.QPixmap, int)
    error = QtCore.Signal(str, int)


class TierGeneratorWorker(QtCore.QRunnable):
    """Worker to generate scaled image tiers in the background."""

    def __init__(self, signals, img_array, image_id):
        super().__init__()
        self.signals = signals
        self.img_array = img_array
        self.image_id = image_id

    def run(self):
        if self.img_array is None:
            return

        try:
            h, w = self.img_array.shape[:2]

            # 1. Immediate Fast Feedback
            total_pixels = h * w
            scale_fast = 0.0625 if total_pixels >= 40_000_000 else 0.125
            fast_w, fast_h = int(w * scale_fast), int(h * scale_fast)
            preview_fast = cv2.resize(
                self.img_array, (fast_w, fast_h), interpolation=cv2.INTER_LINEAR
            )

            img_uint8 = (np.clip(preview_fast, 0, 1) * 255).astype(np.uint8)
            h_p, w_p, c_p = img_uint8.shape
            qimage = QtGui.QImage(
                img_uint8.data, w_p, h_p, c_p * w_p, QtGui.QImage.Format_RGB888
            )
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.signals.uneditedPixmapGenerated.emit(pixmap, self.image_id)

            # Emit fast tier
            self.signals.tierGenerated.emit(scale_fast, preview_fast, self.image_id)

            # 2. High Quality Pyramid Chain
            scales = [
                round(1.0 / 1.5, 4),
                0.5,
                round(1.0 / 3.0, 4),
                0.25,
                round(1.0 / 6.0, 4),
                0.125,
            ]
            if scale_fast <= 0.0625:
                scales.append(0.0625)

            for scale in scales:
                tw, th = int(w * scale), int(h * scale)
                next_img = cv2.resize(
                    self.img_array, (tw, th), interpolation=cv2.INTER_AREA
                )
                self.signals.tierGenerated.emit(scale, next_img, self.image_id)

        except Exception as e:
            logger.error(f"Tier generation error: {e}")


class ImageProcessorWorker(QtCore.QRunnable):
    """Worker to process a single large ROI in a background thread."""

    def __init__(
        self,
        signals,
        base_img_full,
        tiers,
        settings,
        request_id,
        zoom_scale=1.0,
        is_fitting=True,
        calculate_histogram=False,
        last_heavy_adjusted="de_haze",
        lens_info=None,
        target_on_screen_width=None,
        visible_scene_rect=None,
        tile_key=None,
        render_state_id=0,
        calculate_lowres=True,
        settings_state_id=0,
        dehaze_atmospheric_light=None,
    ):
        super().__init__()
        self.signals = signals
        self.base_img_full = base_img_full
        self.tiers = tiers
        self.settings = settings
        self.request_id = request_id
        self.zoom_scale = zoom_scale
        self.is_fitting = is_fitting
        self.calculate_histogram = calculate_histogram
        self.last_heavy_adjusted = last_heavy_adjusted
        self.lens_info = lens_info
        self.target_on_screen_width = target_on_screen_width
        self.visible_scene_rect = visible_scene_rect
        self.tile_key = tile_key
        self.render_state_id = render_state_id
        self.calculate_lowres = calculate_lowres
        self.settings_state_id = settings_state_id
        self.dehaze_atmospheric_light = dehaze_atmospheric_light

    def run(self):
        try:
            *result, rotate_val, visible_scene_rect_out, bg_lowres_pix = (
                self._update_preview()
            )
            try:
                self.signals.finished.emit(
                    *result,
                    rotate_val,
                    visible_scene_rect_out,
                    bg_lowres_pix,
                    self.request_id,
                    self.tile_key,
                    self.render_state_id,
                    self.settings_state_id,
                )
            except RuntimeError:
                pass  # Signal source deleted (app closing)
        except Exception as e:
            logger.error(f"Image processing worker failed: {e}", exc_info=True)
            try:
                self.signals.error.emit(str(e), self.request_id)
            except RuntimeError:
                pass  # Signal source deleted (app closing)

    def _extract_settings(self):
        """Extract settings dictionaries for the pipeline."""
        rotate_val = self.settings.get("rotation", 0.0)
        flip_h = self.settings.get("flip_h", False)
        flip_v = self.settings.get("flip_v", False)
        crop_val = self.settings.get("crop", None)

        heavy_params = {
            "de_haze": self.settings.get("de_haze", 0) / 50.0,
            "denoise_luma": self.settings.get("denoise_luma", 0),
            "denoise_chroma": self.settings.get("denoise_chroma", 0) * 2,
            "denoise_method": self.settings.get("denoise_method", "Bilateral"),
            "sharpen_value": self.settings.get("sharpen_value", 0),
            "sharpen_radius": self.settings.get("sharpen_radius", 0.5),
            "sharpen_percent": self.settings.get("sharpen_percent", 0.0),
        }

        preprocess_settings = {
            "temperature": self.settings.get("temperature", 0.0),
            "tint": self.settings.get("tint", 0.0),
            "exposure": self.settings.get("exposure", 0.0),
        }

        tone_map_settings = {
            "contrast": self.settings.get("contrast", 0.0),
            "blacks": self.settings.get("blacks", 0.0),
            "whites": self.settings.get("whites", 0.0),
            "shadows": self.settings.get("shadows", 0.0),
            "highlights": self.settings.get("highlights", 0.0),
            "saturation": self.settings.get("saturation", 0.0),
        }

        return (
            rotate_val,
            flip_h,
            flip_v,
            crop_val,
            heavy_params,
            preprocess_settings,
            tone_map_settings,
        )

    def _select_resolution_tier(self, full_w):
        """Select the appropriate resolution tier based on zoom and screen bounds."""
        target_on_screen_width = (
            self.target_on_screen_width
            if self.target_on_screen_width is not None
            else full_w
        )
        available_scales = sorted(self.tiers.keys()) + [1.0]

        selected_scale = 1.0
        selected_img = self.base_img_full

        if self.visible_scene_rect is not None:
            if self.zoom_scale >= 1.0:
                selected_scale = 1.0
                selected_img = self.base_img_full
            else:
                for scale in available_scales:
                    tier_img = (
                        self.tiers.get(scale) if scale < 1.0 else self.base_img_full
                    )
                    if tier_img is not None:
                        selected_scale = scale
                        selected_img = tier_img
                        if scale >= self.zoom_scale:
                            break
        else:
            for scale in available_scales:
                tier_img = self.tiers.get(scale) if scale < 1.0 else self.base_img_full
                if tier_img is not None and tier_img.shape[1] >= target_on_screen_width:
                    selected_scale = scale
                    break

        return f"tier_{selected_scale}", selected_scale, selected_img

    def _update_preview(self):
        if self.base_img_full is None:
            return QtGui.QPixmap(), 0, 0, 0.0

        worker_start = time.perf_counter()
        full_h, full_w = self.base_img_full.shape[:2]

        # --- STEP 0: SETTINGS EXTRACTION ---
        t0 = time.perf_counter()
        (
            rotate_val,
            flip_h,
            flip_v,
            crop_val,
            heavy_params,
            preprocess_settings,
            tone_map_settings,
        ) = self._extract_settings()

        full_resolver = GeometryResolver(full_w, full_h)
        full_resolver.resolve(
            rotate=rotate_val,
            crop=crop_val,
            flip_h=flip_h,
            flip_v=flip_v,
            expand=True,
        )
        new_full_w, new_full_h = full_resolver.get_output_size()

        # --- STEP 1: RESOLUTION SELECTION ---
        t1 = time.perf_counter()
        res_key, selected_scale, selected_img = self._select_resolution_tier(full_w)
        h_src, w_src = selected_img.shape[:2]

        crop_x, crop_y, crop_w, crop_h = 0, 0, w_src, h_src
        out_vx, out_vy, out_vw, out_vh = 0, 0, new_full_w, new_full_h

        is_viewport_only = self.visible_scene_rect is not None
        if is_viewport_only:
            out_vx, out_vy, out_vw, out_vh = self.visible_scene_rect

            # Map viewport back to source image to find raw crop bounds
            corners_processed = np.array(
                [
                    [out_vx, out_vy],
                    [out_vx + out_vw, out_vy],
                    [out_vx + out_vw, out_vy + out_vh],
                    [out_vx, out_vy + out_vh],
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            M_full = full_resolver.get_matrix_2x3()
            hom_M = np.vstack([M_full, [0, 0, 1]])
            M_inv = np.linalg.inv(hom_M)[:2, :]

            corners_raw = cv2.transform(corners_processed, M_inv).reshape(-1, 2)
            rx_min, ry_min = corners_raw.min(axis=0)
            rx_max, ry_max = corners_raw.max(axis=0)

            crop_x = int(np.floor(rx_min * selected_scale))
            crop_y = int(np.floor(ry_min * selected_scale))
            crop_w = int(np.ceil((rx_max - rx_min) * selected_scale))
            crop_h = int(np.ceil((ry_max - ry_min) * selected_scale))

            crop_x = max(0, crop_x)
            crop_y = max(0, crop_y)
            crop_w = min(w_src - crop_x, crop_w)
            crop_h = min(h_src - crop_y, crop_h)

            if crop_w <= 0 or crop_h <= 0:
                is_viewport_only = False
                crop_x, crop_y, crop_w, crop_h = 0, 0, w_src, h_src

        vig_params = resolve_vignette_params(
            self.settings,
            self.lens_info,
            roi_offset=(crop_x, crop_y) if is_viewport_only else None,
            full_size=(w_src, h_src),
        )
        vig_k1, vig_k2, vig_k3, vig_cx, vig_cy, vig_fw, vig_fh = vig_params

        # --- STEP 2: PROCESSING ---
        t2 = time.perf_counter()
        source_for_preprocess = (
            selected_img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            if is_viewport_only
            else selected_img
        )

        preprocessed_bg = pynegative.apply_preprocess(
            source_for_preprocess,
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

        # Denoise only what is seen!
        t_denoise = time.perf_counter()
        denoised_bg = process_denoise_stage(
            preprocessed_bg, res_key, heavy_params, self.zoom_scale
        )
        denoise_time = (time.perf_counter() - t_denoise) * 1000

        t_geometry = time.perf_counter()
        fused_maps, o_w, o_h, _ = get_fused_geometry(
            self.settings,
            self.lens_info,
            w_src,
            h_src,
            rotate_val,
            crop_val,
            flip_h,
            flip_v,
            ts_roi=selected_scale,
            roi_offset=(0, 0),
            full_size=(w_src, h_src),
        )
        geometry_time = (time.perf_counter() - t_geometry) * 1000

        if is_viewport_only:
            out_vx_s = int(np.floor(out_vx * selected_scale))
            out_vy_s = int(np.floor(out_vy * selected_scale))
            out_vw_s = int(np.ceil(out_vw * selected_scale))
            out_vh_s = int(np.ceil(out_vh * selected_scale))

            out_vw_s = max(1, min(out_vw_s, o_w - out_vx_s))
            out_vh_s = max(1, min(out_vh_s, o_h - out_vy_s))
            dest_roi_scaled = (out_vx_s, out_vy_s, out_vw_s, out_vh_s)
        else:
            dest_roi_scaled = None

        t_remap = time.perf_counter()
        img_dest = apply_fused_remap(
            denoised_bg,
            fused_maps,
            o_w,
            o_h,
            cv2.INTER_CUBIC,
            crop_offset=(crop_x, crop_y) if is_viewport_only else (0, 0),
            dest_roi=dest_roi_scaled,
        )
        remap_time = (time.perf_counter() - t_remap) * 1000

        t_heavy = time.perf_counter()
        processed_bg = process_heavy_stage(
            img_dest,
            res_key,
            heavy_params,
            self.zoom_scale,
            self.last_heavy_adjusted,
            self.dehaze_atmospheric_light,
        )
        heavy_time = (time.perf_counter() - t_heavy) * 1000

        t_tonemap = time.perf_counter()
        bg_output, _ = pynegative.apply_tone_map(
            processed_bg, **tone_map_settings, calculate_stats=False
        )
        bg_output = pynegative.apply_defringe(bg_output, self.settings)
        tonemap_time = (time.perf_counter() - t_tonemap) * 1000

        img_uint8 = pynegative.float32_to_uint8(bg_output)

        # -- STEP 3: Handle LowRes Pan Background & Histograms --
        t3 = time.perf_counter()
        bg_lowres_pix = None
        if self.calculate_lowres:
            if is_viewport_only:
                # Pick the smallest available tier for lowres background / histogram.
                # The 0.0625 tier only exists for >=40MP images; fall back gracefully.
                low_scale = None
                low_img = None
                for s in sorted(self.tiers.keys()):
                    if self.tiers[s] is not None:
                        low_scale = s
                        low_img = self.tiers[s]
                        break
                if low_img is not None:
                    low_h, low_w = low_img.shape[:2]
                    low_vig_params = resolve_vignette_params(
                        self.settings, self.lens_info, None, (low_w, low_h)
                    )
                    low_pre = pynegative.apply_preprocess(
                        low_img,
                        temperature=preprocess_settings["temperature"],
                        tint=preprocess_settings["tint"],
                        exposure=preprocess_settings["exposure"],
                        vignette_k1=low_vig_params[0],
                        vignette_k2=low_vig_params[1],
                        vignette_k3=low_vig_params[2],
                        vignette_cx=low_vig_params[3],
                        vignette_cy=low_vig_params[4],
                        full_width=low_vig_params[5],
                        full_height=low_vig_params[6],
                    )

                    low_maps, low_o_w, low_o_h, _ = get_fused_geometry(
                        self.settings,
                        self.lens_info,
                        low_w,
                        low_h,
                        rotate_val,
                        crop_val,
                        flip_h,
                        flip_v,
                        ts_roi=low_scale,
                    )

                    low_dest = apply_fused_remap(
                        low_pre, low_maps, low_o_w, low_o_h, cv2.INTER_LINEAR
                    )

                    low_heavy = process_heavy_stage(
                        low_dest,
                        f"tier_{low_scale}",
                        heavy_params,
                        low_scale,
                        self.last_heavy_adjusted,
                        self.dehaze_atmospheric_light,
                    )

                    low_out, _ = pynegative.apply_tone_map(
                        low_heavy, **tone_map_settings, calculate_stats=False
                    )
                    low_out = pynegative.apply_defringe(low_out, self.settings)

                    low_uint8 = pynegative.float32_to_uint8(low_out)

                    if self.calculate_histogram:
                        hist_data = calculate_histograms(low_uint8)
                        self.signals.histogramUpdated.emit(hist_data, self.request_id)

                    lh, lw = low_uint8.shape[:2]
                    qimg_low = QtGui.QImage(
                        low_uint8.data, lw, lh, 3 * lw, QtGui.QImage.Format_RGB888
                    )
                    bg_lowres_pix = QtGui.QPixmap.fromImage(qimg_low)
                else:
                    # No lowres tier available yet — compute histogram from
                    # the processed tile so the widget doesn't stay empty.
                    if self.calculate_histogram:
                        hist_data = calculate_histograms(img_uint8)
                        self.signals.histogramUpdated.emit(hist_data, self.request_id)
            else:
                if self.calculate_histogram:
                    hist_data = calculate_histograms(img_uint8)
                    self.signals.histogramUpdated.emit(hist_data, self.request_id)

        h_bg, w_bg = img_uint8.shape[:2]
        qimg_bg = QtGui.QImage(
            img_uint8.data, w_bg, h_bg, 3 * w_bg, QtGui.QImage.Format_RGB888
        )
        pix_bg = QtGui.QPixmap.fromImage(qimg_bg)

        visible_scene_rect_out = (
            (out_vx, out_vy, out_vw, out_vh) if is_viewport_only else None
        )

        # Timing breakdown (at end to capture all work)
        lowres_time = (time.perf_counter() - t3) * 1000
        setup_time = (t1 - t0) * 1000
        roi_time = (t2 - t1) * 1000
        preprocess_time = (t_denoise - t2) * 1000
        worker_total = (time.perf_counter() - worker_start) * 1000

        logger.debug(
            f"⚙️  Worker Pipeline | Setup: {setup_time:.1f}ms | ROI: {roi_time:.1f}ms | "
            f"Prepro: {preprocess_time:.1f}ms | Denoise: {denoise_time:.1f}ms | "
            f"Geo: {geometry_time:.1f}ms | Remap: {remap_time:.1f}ms | Heavy: {heavy_time:.1f}ms | "
            f"Tonemap: {tonemap_time:.1f}ms | Lowres+Hist: {lowres_time:.1f}ms | Total: {worker_total:.1f}ms"
        )

        return (
            pix_bg,
            new_full_w,
            new_full_h,
            rotate_val,
            visible_scene_rect_out,
            bg_lowres_pix,
        )


def _select_resolution_tier(
    tiers, base_img_full, zoom_scale, target_on_screen_width, visible_scene_rect, full_w
):
    """Select the appropriate resolution tier based on zoom and screen bounds."""
    target_on_screen_width = (
        target_on_screen_width if target_on_screen_width is not None else full_w
    )
    available_scales = sorted(tiers.keys()) + [1.0]

    selected_scale = 1.0
    selected_img = base_img_full

    if visible_scene_rect is not None:
        if zoom_scale >= 1.0:
            pass  # selected_scale = 1.0; selected_img = base_img_full
        else:
            for scale in available_scales:
                tier_img = tiers.get(scale) if scale < 1.0 else base_img_full
                if tier_img is not None:
                    selected_scale = scale
                    selected_img = tier_img
                    if scale >= zoom_scale:
                        break
    else:
        for scale in available_scales:
            tier_img = tiers.get(scale) if scale < 1.0 else base_img_full
            if tier_img is not None and tier_img.shape[1] >= target_on_screen_width:
                selected_scale = scale
                selected_img = tier_img
                break

    return f"tier_{selected_scale}", selected_scale, selected_img
