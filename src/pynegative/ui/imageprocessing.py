import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui
from .. import core as pynegative


class ImageProcessingPipeline(QtCore.QObject):
    previewUpdated = QtCore.Signal(
        QtGui.QPixmap,  # Background pixmap
        int,
        int,  # Full image dimensions (width, height)
        QtGui.QPixmap,  # ROI pixmap (None if no ROI)
        int,
        int,
        int,
        int,  # ROI coordinates and dimensions
    )

    def __init__(self, parent=None):
        super().__init__(parent)

        # Throttled Render Timer (30 FPS)
        self.render_timer = QtCore.QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._on_render_timer_timeout)
        self._render_pending = False
        self._is_rendering_locked = False

        # Image data
        self.base_img_full = None  # The High-Res Proxy (e.g. 4000px)
        self._base_img_uint8 = None  # Cached uint8 version for resizing

        # Processing parameters
        self.val_exposure = 0.0
        self.val_contrast = 1.0
        self.val_whites = 1.0
        self.val_blacks = 0.0
        self.val_highlights = 0.0
        self.val_shadows = 0.0
        self.val_saturation = 1.0
        self.val_sharpen = 0.0
        self.val_radius = 0.5
        self.val_percent = 0.0
        self.val_denoise = 0

        # View reference for processing (set from editor)
        self._view_ref = None

    def set_image(self, img_array):
        """Set the base image for processing."""
        self.base_img_full = img_array

        # Clear caches for the new image
        self._base_img_uint8 = None
        if hasattr(self, "_img_render_base"):
            del self._img_render_base

    def set_view_reference(self, view):
        """Set reference to the view for processing."""
        self._view_ref = view

    def set_processing_params(self, **kwargs):
        """Update processing parameters."""
        for key, value in kwargs.items():
            if hasattr(self, f"val_{key}"):
                setattr(self, f"val_{key}", value)

    def request_update(self):
        """
        Requests an image redraw. Uses a throttled timer to maintain 30 FPS.
        """
        if self.base_img_full is None:
            return

        self._render_pending = True

        if not self._is_rendering_locked:
            self._process_pending_update()

    def _process_pending_update(self):
        """Processes the actual redraw and manages the throttle lockout."""
        if not self._render_pending or self.base_img_full is None:
            return

        # Perform the actual update
        self.update_preview()
        self._render_pending = False

        # Lock rendering for 33ms (30 FPS)
        self._is_rendering_locked = True
        self.render_timer.start(33)  # 33ms interval for throttle

    def _on_render_timer_timeout(self):
        """Called when the 33ms throttle window expires."""
        self._is_rendering_locked = False
        # If another update was requested during the lockout, process it now
        if self._render_pending:
            self._process_pending_update()

    def update_preview(self):
        if self.base_img_full is None or self._view_ref is None:
            return

        # Strategy: Single-Layer Dynamic ROI
        # 1. Determine the viewport size
        # 2. If zoomed out (Fit), process a 1500px global image.
        # 3. If zoomed in, process a crop matching the viewport size from base_img_full.

        full_h, full_w, _ = self.base_img_full.shape

        # Safe access to view properties
        try:
            if self._view_ref is None:
                return
            zoom_scale = self._view_ref.transform().m11()
            viewport = self._view_ref.viewport()
            if viewport is None:
                return
            vw, vh = (
                viewport.width(),
                viewport.height(),
            )
            if vw <= 0 or vh <= 0:
                return
        except (AttributeError, RuntimeError):
            return

        # Calculate fit scale
        fit_scale = 1.0
        if vw > 0 and vh > 0:
            fit_scale = min(vw / full_w, vh / full_h)

        # Safe access to view attributes
        if self._view_ref is None:
            return
        is_fitting = getattr(self._view_ref, "_is_fitting", False)
        is_zoomed_in = not is_fitting and (
            zoom_scale > fit_scale * 1.01 or zoom_scale > 0.99
        )

        # --- Part 1: Global Background ---
        # Cache the 1500px base
        if self._base_img_uint8 is None:
            self._base_img_uint8 = (self.base_img_full * 255).astype(np.uint8)

        scale = 1500 / max(full_h, full_w)
        target_h, target_w = int(full_h * scale), int(full_w * scale)
        if (
            not hasattr(self, "_img_render_base")
            or self._img_render_base.shape[0] != target_h
        ):
            temp_pil = Image.fromarray(self._base_img_uint8)
            temp_pil = temp_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
            self._img_render_base = np.array(temp_pil).astype(np.float32) / 255.0

        # Process Background
        pix_bg = QtGui.QPixmap()
        try:
            processed_bg, _ = pynegative.apply_tone_map(
                self._img_render_base,
                exposure=self.val_exposure,
                contrast=self.val_contrast,
                blacks=self.val_blacks,
                whites=self.val_whites,
                shadows=self.val_shadows,
                highlights=self.val_highlights,
                saturation=self.val_saturation,
            )
            processed_bg *= 255
            pil_bg = Image.fromarray(processed_bg.astype(np.uint8))
            qimg_bg = ImageQt.ImageQt(pil_bg)
            pix_bg = QtGui.QPixmap.fromImage(qimg_bg)
        except Exception:
            pass

        # --- Part 2: Detail ROI (Only if zoomed in) ---
        pix_roi, roi_x, roi_y, roi_w, roi_h = QtGui.QPixmap(), 0, 0, 0, 0
        if is_zoomed_in and self._view_ref is not None:
            try:
                viewport_rect = self._view_ref.viewport().rect()
                if viewport_rect is None:
                    pass
                else:
                    roi = self._view_ref.mapToScene(viewport_rect).boundingRect()
                    ix_min = max(0, int(roi.left()))
                    ix_max = min(full_w, int(roi.right()))
                    iy_min = max(0, int(roi.top()))
                    iy_max = min(full_h, int(roi.bottom()))

                    if ix_max > ix_min and iy_max > iy_min:
                        rw = ix_max - ix_min
                        rh = iy_max - iy_min

                        if rw > 10 and rh > 10:
                            # Performance Cap: Max 1.5 million pixels for real-time ROI
                            MAX_REALTIME_PIXELS = 1_500_000
                            current_pixels = rw * rh

                            crop = self.base_img_full[iy_min:iy_max, ix_min:ix_max]

                            if current_pixels > MAX_REALTIME_PIXELS:
                                # Scale down for processing, then GPU upscale later
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
                                crop_to_proc,
                                exposure=self.val_exposure,
                                contrast=self.val_contrast,
                                blacks=self.val_blacks,
                                whites=self.val_whites,
                                shadows=self.val_shadows,
                                highlights=self.val_highlights,
                                saturation=self.val_saturation,
                            )
                            processed_roi *= 255
                            pil_roi = Image.fromarray(processed_roi.astype(np.uint8))

                            if self.val_sharpen > 0:
                                pil_roi = pynegative.sharpen_image(
                                    pil_roi,
                                    self.val_radius,
                                    self.val_percent,
                                    "High Quality",
                                )

                            if self.val_denoise > 0:
                                pil_roi = pynegative.de_noise_image(
                                    pil_roi, self.val_denoise, "High Quality"
                                )

                            pix_roi = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_roi))
                            roi_x, roi_y = ix_min, iy_min
                            roi_w, roi_h = rw, rh
            except (AttributeError, RuntimeError):
                # If we can't get ROI, just skip ROI processing
                pass

        # --- Final Update ---
        self.previewUpdated.emit(
            pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h
        )

    def get_current_settings(self):
        """Get current processing settings as a dictionary."""
        return {
            "exposure": self.val_exposure,
            "contrast": self.val_contrast,
            "whites": self.val_whites,
            "blacks": self.val_blacks,
            "highlights": self.val_highlights,
            "shadows": self.val_shadows,
            "saturation": self.val_saturation,
            "sharpen_method": "High Quality",
            "sharpen_radius": self.val_radius,
            "sharpen_percent": self.val_percent,
            "sharpen_value": self.val_sharpen,
            "denoise_method": "High Quality",
            "de_noise": self.val_denoise,
        }
