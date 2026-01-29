from pathlib import Path
import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt

from .. import core as pynegative
from .loaders import ThumbnailLoader, RawLoader
from .widgets import HorizontalListWidget, CollapsibleSection, ResetableSlider, ZoomableGraphicsView, ZoomControls

class EditorWidget(QtWidgets.QWidget):
    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.raw_path = None
        self.base_img_full = None  # The High-Res Proxy (e.g. 4000px)
        self._base_img_uint8 = None # Cached uint8 version for resizing
        self.current_qpixmap = None

        # Auto-save timer
        self.save_timer = QtCore.QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._auto_save_sidecar)

        # Throttled Render Timer (30 FPS)
        self.render_timer = QtCore.QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._on_render_timer_timeout)
        self._render_pending = False
        self._is_rendering_locked = False

        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # --- Left Panel (Controls) ---
        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("EditorPanel")
        self.panel.setFixedWidth(280)
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(10, 10, 10, 10)
        self.panel_layout.setSpacing(2)
        main_layout.addWidget(self.panel)

        # --- Canvas (Right Side) ---
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setObjectName("CanvasFrame")
        self.canvas_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas_frame)

        # View replaced with ZoomableGraphicsView
        self.view = ZoomableGraphicsView()

        # Layout for canvas + zoom controls
        self.canvas_container = QtWidgets.QGridLayout(self.canvas_frame)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.canvas_container.addWidget(self.view, 0, 0)

        # Zoom Controls (Bottom Right overlay)
        self.zoom_ctrl = ZoomControls()
        self.zoom_ctrl.zoomChanged.connect(lambda z: self.view.set_zoom(z, manual=True))
        self.view.zoomChanged.connect(self.zoom_ctrl.update_zoom)

        # Trigger ROI re-render when zoom or pan changes
        self.view.zoomChanged.connect(self.request_update)

        self.canvas_container.addWidget(self.zoom_ctrl, 0, 0, Qt.AlignBottom | Qt.AlignRight)
        self.zoom_ctrl.setContentsMargins(0, 0, 20, 20)

        # Carousel (Bottom)
        self.carousel = HorizontalListWidget()
        self.carousel.setObjectName("Carousel")
        self.carousel.setViewMode(QtWidgets.QListView.IconMode)
        self.carousel.setFlow(QtWidgets.QListView.LeftToRight) # Horizontal
        self.carousel.setWrapping(False)
        self.carousel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.carousel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.carousel.setFixedHeight(145)
        self.carousel.setIconSize(QtCore.QSize(100, 100))
        self.carousel.setSpacing(5)
        self.carousel.itemClicked.connect(self._on_carousel_item_clicked)
        self.canvas_container.addWidget(self.carousel, 1, 0)

        self._setup_controls()

    def resizeEvent(self, event):
        # Auto-fit image if in fitting mode
        # Using hasattr check to be absolutely safe against racing AttributeErrors
        if hasattr(self, 'base_img_full') and self.base_img_full is not None:
            if getattr(self.view, '_is_fitting', False):
                 self.view.reset_zoom()
        super().resizeEvent(event)

    def _setup_controls(self):
        # Wrap everything in a scroll area just in case
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.panel_layout.addWidget(scroll)

        container = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(container)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(5)
        scroll.setWidget(container)

        self.lbl_info = QtWidgets.QLabel("No file loaded")
        self.lbl_info.setObjectName("InfoLabel")
        self.lbl_info.setWordWrap(True)
        self.controls_layout.addWidget(self.lbl_info)

        # --- Tone Section ---
        self.tone_section = CollapsibleSection("TONE", expanded=True)
        self.controls_layout.addWidget(self.tone_section)

        self.val_exposure = 0.0
        self.val_contrast = 1.0
        self.val_whites = 1.0
        self.val_blacks = 0.0
        self.val_highlights = 0.0
        self.val_shadows = 0.0

        self._add_slider("Exposure", -4.0, 4.0, self.val_exposure, "val_exposure", 0.01, self.tone_section)
        self._add_slider("Contrast", 0.5, 2.0, self.val_contrast, "val_contrast", 0.01, self.tone_section)
        self._add_slider("Highlights", -1.0, 1.0, self.val_highlights, "val_highlights", 0.01, self.tone_section)
        self._add_slider("Shadows", -1.0, 1.0, self.val_shadows, "val_shadows", 0.01, self.tone_section)
        self._add_slider("Whites", 0.5, 1.5, self.val_whites, "val_whites", 0.01, self.tone_section, flipped=True)
        self._add_slider("Blacks", -0.2, 0.2, self.val_blacks, "val_blacks", 0.001, self.tone_section)

        # --- Color Section ---
        self.color_section = CollapsibleSection("COLOR", expanded=False)
        self.controls_layout.addWidget(self.color_section)

        self.val_saturation = 1.0
        self._add_slider("Saturation", 0.0, 2.0, self.val_saturation, "val_saturation", 0.01, self.color_section)

        # --- Details Section ---
        self.details_section = CollapsibleSection("DETAILS", expanded=False)
        self.controls_layout.addWidget(self.details_section)

        self.var_sharpen_enabled = False
        self.sharpen_checkbox = QtWidgets.QCheckBox("Enable Processing")
        self.sharpen_checkbox.toggled.connect(self._update_sharpen_state)
        self.details_section.add_widget(self.sharpen_checkbox)

        self.val_radius = 2.0
        self.val_percent = 150
        self.val_denoise = 0
        self._add_slider("Sharpen Radius", 0.5, 5.0, self.val_radius, "val_radius", 0.01, self.details_section)
        self._add_slider("Sharpen Amount", 0, 300, self.val_percent, "val_percent", 1, self.details_section)
        self._add_slider("De-noise", 0, 5, self.val_denoise, "val_denoise", 1, self.details_section)

        # Save Button
        self.controls_layout.addSpacing(10)
        self.btn_save = QtWidgets.QPushButton("Save Result")
        self.btn_save.setObjectName("SaveButton")
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setEnabled(False)
        self.controls_layout.addWidget(self.btn_save)

        self.controls_layout.addStretch()

    def _add_slider(self, label_text, min_val, max_val, default, var_name, step_size, section=None, flipped=False):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(0)

        # Store flip state for programmatic updates
        setattr(self, f"{var_name}_flipped", flipped)

        # Top row: Label and Value
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label_text)
        val_lbl = QtWidgets.QLabel(f"{default:.2f}")
        val_lbl.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(lbl)
        row.addWidget(val_lbl)
        layout.addLayout(row)

        slider = ResetableSlider(QtCore.Qt.Horizontal)
        multiplier = 1000
        slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
        # Default value on initial setup
        slider.default_slider_value = int(default * multiplier)
        slider.setValue(int(default * multiplier))

        def on_change(val):
            actual = val / multiplier
            if flipped:
                # Map slider min..max to max..min
                # Formula: actual = s_max + s_min - actual
                actual = max_val + min_val - actual

            val_lbl.setText(f"{actual:.2f}")
            setattr(self, var_name, actual)
            self.request_update()

            # Trigger auto-save
            self.save_timer.start(1000) # Save after 1 second of inactivity

        slider.valueChanged.connect(on_change)

        # Store refs
        setattr(self, f"{var_name}_slider", slider)
        setattr(self, f"{var_name}_label", val_lbl) # Store label for updates

        layout.addWidget(slider)
        if section:
            section.add_widget(frame)
        else:
            self.panel_layout.addWidget(frame)

    def _set_slider_value(self, var_name, value):
        slider = getattr(self, f"{var_name}_slider", None)
        label = getattr(self, f"{var_name}_label", None)
        flipped = getattr(self, f"{var_name}_flipped", False)

        if slider:
            multiplier = 1000
            if flipped:
                # Need to find the min/max to calculate the flipped slider position
                s_min = slider.minimum() / multiplier
                s_max = slider.maximum() / multiplier
                val_to_set = (s_max + s_min) - value
                slider.setValue(int(val_to_set * multiplier))
            else:
                slider.setValue(int(value * multiplier))

            # Since this is a programmatic update (likely from load),
            # we also update the reset value.
            slider.default_slider_value = slider.value()

        if label:
            label.setText(f"{value:.2f}")
        setattr(self, var_name, value)


    def _update_sharpen_state(self, checked):
        self.var_sharpen_enabled = checked
        self.request_update()
        self.save_timer.start(500)

    def _auto_save_sidecar(self):
        if not self.raw_path:
            return

        settings = {
            "exposure": self.val_exposure,
            "contrast": self.val_contrast,
            "whites": self.val_whites,
            "blacks": self.val_blacks,
            "highlights": self.val_highlights,
            "shadows": self.val_shadows,
            "saturation": self.val_saturation,
            "sharpen_enabled": self.var_sharpen_enabled,
            "sharpen_radius": self.val_radius,
            "sharpen_percent": self.val_percent,
            "de_noise": self.val_denoise
        }
        pynegative.save_sidecar(self.raw_path, settings)

    def load_image(self, path):
        path = Path(path)
        self.lbl_info.setText(f"Loading: {path.name}")
        self.raw_path = path

        QtWidgets.QApplication.processEvents()

        self.btn_save.setEnabled(False) # Disable save until full load

        loader = RawLoader(path)
        loader.signals.finished.connect(self._on_raw_loaded)
        self.thread_pool.start(loader)

    def _on_raw_loaded(self, path, img_arr, settings):
        if Path(path) != self.raw_path:
            return # User switched images already

        if img_arr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load image")
            return

        # Apply Auto-Expose Settings
        if settings:
            self._set_slider_value("val_exposure", settings.get("exposure", 0.0))
            self._set_slider_value("val_contrast", settings.get("contrast", 1.0))
            self._set_slider_value("val_whites", settings.get("whites", 1.0))
            self._set_slider_value("val_blacks", settings.get("blacks", 0.0))
            self._set_slider_value("val_saturation", settings.get("saturation", 1.0))
            self._set_slider_value("val_highlights", settings.get("highlights", 0.0))
            self._set_slider_value("val_shadows", settings.get("shadows", 0.0))

            # Sharpening state
            sharpen_on = settings.get("sharpen_enabled", False)
            self.sharpen_checkbox.setChecked(sharpen_on)
            self.var_sharpen_enabled = sharpen_on
            self._set_slider_value("val_radius", settings.get("sharpen_radius", 2.0))
            self._set_slider_value("val_percent", settings.get("sharpen_percent", 150))
            self._set_slider_value("val_denoise", settings.get("de_noise", 0))

        self.base_img_full = img_arr # The half-res proxy

        # Clear caches for the new image
        self._base_img_uint8 = None
        if hasattr(self, '_img_render_base'):
            del self._img_render_base

        self.btn_save.setEnabled(True)
        self.request_update()

        # Request a fit once the UI settles
        QtCore.QTimer.singleShot(50, self.view.reset_zoom)
        QtCore.QTimer.singleShot(200, self.view.reset_zoom)

        self.lbl_info.setText(f"Loaded: {self.raw_path.name}")

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
        self.render_timer.start(33) # 33ms interval for throttle

    def _on_render_timer_timeout(self):
        """Called when the 33ms throttle window expires."""
        self._is_rendering_locked = False
        # If another update was requested during the lockout, process it now
        if self._render_pending:
            self._process_pending_update()

    def update_preview(self):
        if self.base_img_full is None: return

        # Strategy: Single-Layer Dynamic ROI
        # 1. Determine the viewport size
        # 2. If zoomed out (Fit), process a 1500px global image.
        # 3. If zoomed in, process a crop matching the viewport size from base_img_full.

        full_h, full_w, _ = self.base_img_full.shape
        zoom_scale = self.view.transform().m11()
        vw, vh = self.view.viewport().width(), self.view.viewport().height()

        # Calculate fit scale
        fit_scale = 1.0
        if vw > 0 and vh > 0:
            fit_scale = min(vw / full_w, vh / full_h)

        is_zoomed_in = not self.view._is_fitting and (zoom_scale > fit_scale * 1.01 or zoom_scale > 0.99)

        # --- Part 1: Global Background ---
        # Cache the 1500px base
        if self._base_img_uint8 is None:
            self._base_img_uint8 = (self.base_img_full * 255).astype(np.uint8)

        scale = 1500 / max(full_h, full_w)
        target_h, target_w = int(full_h * scale), int(full_w * scale)
        if not hasattr(self, '_img_render_base') or self._img_render_base.shape[0] != target_h:
            temp_pil = Image.fromarray(self._base_img_uint8)
            temp_pil = temp_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
            self._img_render_base = np.array(temp_pil).astype(np.float32) / 255.0

        # Process Background
        processed_bg, _ = pynegative.apply_tone_map(
            self._img_render_base,
            exposure=self.val_exposure, contrast=self.val_contrast,
            blacks=self.val_blacks, whites=self.val_whites,
            shadows=self.val_shadows, highlights=self.val_highlights,
            saturation=self.val_saturation
        )
        processed_bg *= 255
        pil_bg = Image.fromarray(processed_bg.astype(np.uint8))
        pix_bg = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_bg))

        # --- Part 2: Detail ROI (Only if zoomed in) ---
        pix_roi, roi_x, roi_y, roi_w, roi_h = None, 0, 0, 0, 0
        if is_zoomed_in:
            roi = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
            ix_min = max(0, int(roi.left()))
            ix_max = min(full_w, int(roi.right()))
            iy_min = max(0, int(roi.top()))
            iy_max = min(full_h, int(roi.bottom()))

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

                    pil_crop = Image.fromarray((crop * 255).astype(np.uint8))
                    pil_crop = pil_crop.resize((p_w, p_h), Image.Resampling.BILINEAR)
                    crop_to_proc = np.array(pil_crop).astype(np.float32) / 255.0
                else:
                    crop_to_proc = crop

                processed_roi, _ = pynegative.apply_tone_map(
                    crop_to_proc,
                    exposure=self.val_exposure, contrast=self.val_contrast,
                    blacks=self.val_blacks, whites=self.val_whites,
                    shadows=self.val_shadows, highlights=self.val_highlights,
                    saturation=self.val_saturation
                )
                processed_roi *= 255
                pil_roi = Image.fromarray(processed_roi.astype(np.uint8))

                if self.var_sharpen_enabled:
                    pil_roi = pynegative.sharpen_image(pil_roi, self.val_radius, self.val_percent)
                    if self.val_denoise > 0:
                        pil_roi = pynegative.de_noise_image(pil_roi, self.val_denoise)

                pix_roi = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_roi))
                roi_x, roi_y = ix_min, iy_min
                roi_w, roi_h = rw, rh

        # --- Final Update ---
        self.view.set_pixmaps(pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h)

    def save_file(self):
        if self.base_img_full is None: return

        input_dir = self.raw_path.parent
        default_name = self.raw_path.with_suffix(".jpg").name
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save", str(input_dir / default_name), "JPEG (*.jpg);;HEIF (*.heic)")

        if path:
            path = Path(path)
            try:
                # RELOAD FULL RESOLUTION FOR SAVING
                self.lbl_info.setText("Processing Full Res...")
                QtWidgets.QApplication.processEvents()

                # Explicitly request full size (half_size=False)
                full_img = pynegative.open_raw(self.raw_path, half_size=False)

                # Process full resolution with current settings
                img, _ = pynegative.apply_tone_map(
                    full_img,
                    exposure=self.val_exposure,
                    contrast=self.val_contrast,
                    blacks=self.val_blacks,
                    whites=self.val_whites,
                    shadows=self.val_shadows,
                    highlights=self.val_highlights,
                    saturation=self.val_saturation
                )
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                if self.var_sharpen_enabled:
                    pil_img = pynegative.sharpen_image(pil_img, self.val_radius, self.val_percent)
                    if self.val_denoise > 0:
                        pil_img = pynegative.de_noise_image(pil_img, self.val_denoise)

                pynegative.save_image(pil_img, path)
                QtWidgets.QMessageBox.information(self, "Saved", f"Saved full resolution to {path}")
                self.lbl_info.setText(f"Saved: {path.name}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
                self.lbl_info.setText("Error saving")

    def load_carousel_folder(self, folder):
        self.current_folder = Path(folder)
        self.carousel.clear()

        files = sorted([f for f in self.current_folder.iterdir() if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS])

        for path in files:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.carousel.addItem(item)

            # Async load thumbnail
            loader = ThumbnailLoader(path, size=100)
            loader.signals.finished.connect(self._on_carousel_thumbnail_loaded)
            self.thread_pool.start(loader)

    def _on_carousel_thumbnail_loaded(self, path, pixmap):
        for i in range(self.carousel.count()):
            item = self.carousel.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))
                break

    def _on_carousel_item_clicked(self, item):
        path = item.data(QtCore.Qt.UserRole)
        # Avoid reloading if same image
        if Path(path) != self.raw_path:
            self.load_image(path)
