from PySide6 import QtWidgets, QtCore
from .widgets import (
    CollapsibleSection,
    StarRatingWidget,
    HistogramWidget,
)
from .controls.tone_controls import ToneControls
from .controls.color_controls import ColorControls
from .controls.detail_controls import DetailControls
from .controls.geometry_controls import GeometryControls
from .controls.lens_controls import LensControls


class EditingControls(QtWidgets.QWidget):
    DENOISE_METHODS = [
        "High Quality",
        "NLMeans (Numba Hybrid YUV)",
        "NLMeans (Numba Fast+ YUV)",
    ]

    # Signals for changes
    settingChanged = QtCore.Signal(str, object)  # setting_name, value
    ratingChanged = QtCore.Signal(int)
    presetApplied = QtCore.Signal(str)
    autoWbRequested = QtCore.Signal()
    histogramModeChanged = QtCore.Signal(str)
    cropToggled = QtCore.Signal(bool)
    aspectRatioChanged = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.val_denoise_method = "NLMeans (Numba Fast+)"
        self._init_ui()

    def _init_ui(self):
        # Wrap everything in a scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        container = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(container)
        self.controls_layout.setContentsMargins(8, 0, 8, 0)
        self.controls_layout.setSpacing(5)
        scroll.setWidget(container)

        # Dynamic margin adjustment for scrollbar
        def _update_scroll_margins():
            try:
                vbar = scroll.verticalScrollBar()
                is_visible = vbar.isVisible()
                right_margin = 24 if is_visible else 8
                self.controls_layout.setContentsMargins(8, 0, right_margin, 0)
            except Exception:
                pass

        scroll.verticalScrollBar().rangeChanged.connect(_update_scroll_margins)
        scroll.verticalScrollBar().valueChanged.connect(_update_scroll_margins)
        QtCore.QTimer.singleShot(100, _update_scroll_margins)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        # --- Histogram Section ---
        self.histogram_section = CollapsibleSection("HISTOGRAM", expanded=False)
        self.controls_layout.addWidget(self.histogram_section)
        self.histogram_widget = HistogramWidget()
        self.histogram_section.add_widget(self.histogram_widget)

        self.hist_mode_combo = QtWidgets.QComboBox()
        self.hist_mode_combo.addItems(["Auto", "Luminance", "RGB", "YUV"])
        self.hist_mode_combo.currentTextChanged.connect(self._on_hist_mode_changed)
        self.histogram_section.add_widget(self.hist_mode_combo)

        # --- Rating Section ---
        self.rating_section = CollapsibleSection("RATING", expanded=True)
        self.controls_layout.addWidget(self.rating_section)
        self.star_rating_widget_internal = StarRatingWidget()
        self.star_rating_widget_internal.ratingChanged.connect(self.ratingChanged.emit)
        self.rating_section.add_widget(self.star_rating_widget_internal)

        # --- Control Sub-widgets ---
        self.tone_controls = ToneControls()
        self.color_controls = ColorControls()
        self.detail_controls = DetailControls()
        self.geometry_controls = GeometryControls()
        self.lens_controls = LensControls()

        self.controls_layout.addWidget(self.tone_controls)
        self.controls_layout.addWidget(self.color_controls)
        self.controls_layout.addWidget(self.detail_controls)
        self.controls_layout.addWidget(self.geometry_controls)
        self.controls_layout.addWidget(self.lens_controls)

        # Connect signals
        self.tone_controls.settingChanged.connect(self.settingChanged.emit)
        self.color_controls.settingChanged.connect(self.settingChanged.emit)
        self.color_controls.autoWbRequested.connect(self.autoWbRequested.emit)
        self.detail_controls.settingChanged.connect(self.settingChanged.emit)
        self.detail_controls.presetApplied.connect(self.presetApplied.emit)
        self.geometry_controls.settingChanged.connect(self.settingChanged.emit)
        self.geometry_controls.cropToggled.connect(self.cropToggled.emit)
        self.geometry_controls.aspectRatioChanged.connect(self.aspectRatioChanged.emit)
        self.lens_controls.settingChanged.connect(self.settingChanged.emit)

        self.controls_layout.addStretch()

    @property
    def val_temperature(self):
        return self.color_controls.get_value("val_temperature")

    @property
    def val_tint(self):
        return self.color_controls.get_value("val_tint")

    @property
    def val_exposure(self):
        return self.tone_controls.get_value("val_exposure")

    @property
    def val_contrast(self):
        return self.tone_controls.get_value("val_contrast")

    @property
    def val_whites(self):
        return self.tone_controls.get_value("val_whites")

    @property
    def val_blacks(self):
        return self.tone_controls.get_value("val_blacks")

    @property
    def val_highlights(self):
        return self.tone_controls.get_value("val_highlights")

    @property
    def val_shadows(self):
        return self.tone_controls.get_value("val_shadows")

    @property
    def val_saturation(self):
        return self.color_controls.get_value("val_saturation")

    @property
    def val_sharpen_value(self):
        return self.detail_controls.get_value("val_sharpen_value")

    @property
    def val_sharpen_radius(self):
        return self.detail_controls.val_sharpen_radius

    @property
    def val_sharpen_percent(self):
        return self.detail_controls.val_sharpen_percent

    @property
    def val_denoise_luma(self):
        return self.detail_controls.get_value("val_denoise_luma")

    @property
    def val_denoise_chroma(self):
        return self.detail_controls.get_value("val_denoise_chroma")

    @property
    def val_de_haze(self):
        return self.detail_controls.get_value("val_de_haze")

    @property
    def val_flip_h(self):
        return self.geometry_controls.val_flip_h

    @property
    def val_flip_v(self):
        return self.geometry_controls.val_flip_v

    @property
    def rotation(self):
        return self.geometry_controls.get_value("rotation")

    @property
    def val_lens_distortion(self):
        return self.lens_controls.get_value("lens_distortion")

    @property
    def val_lens_vignette(self):
        return self.lens_controls.get_value("lens_vignette")

    @property
    def val_lens_ca(self):
        return self.lens_controls.get_value("lens_ca")

    # Compatibility for tests
    @property
    def val_sharpen_value_slider(self):
        return self.detail_controls.sliders.get("val_sharpen_value")

    @property
    def details_section(self):
        return self.detail_controls.details_section

    @property
    def tone_section(self):
        return self.tone_controls.tone_section

    @property
    def color_section(self):
        return self.color_controls.color_section

    @property
    def geometry_section(self):
        return self.geometry_controls.geometry_section

    @property
    def star_rating_widget(self):
        return self.star_rating_widget_internal

    def set_slider_value(self, var_name, value, silent=False):
        if var_name in [
            "val_exposure",
            "val_contrast",
            "val_highlights",
            "val_shadows",
            "val_whites",
            "val_blacks",
        ]:
            self.tone_controls.set_slider_value(var_name, value, silent)
        elif var_name in ["val_temperature", "val_tint", "val_saturation"]:
            self.color_controls.set_slider_value(var_name, value, silent)
        elif var_name in [
            "val_sharpen_value",
            "val_denoise_luma",
            "val_denoise_chroma",
            "val_de_haze",
        ]:
            self.detail_controls.set_slider_value(var_name, value, silent)
        elif var_name == "rotation":
            self.geometry_controls.set_slider_value(var_name, value, silent)

    def set_crop_checked(self, checked):
        self.geometry_controls.crop_btn.setChecked(checked)

    def reset_sliders(self, silent=False):
        self.tone_controls.reset_section()
        self.color_controls.reset_section()
        self.detail_controls.reset_section()
        self.geometry_controls.reset_section()
        self.lens_controls.reset_section()

    def cycle_denoise_method(self):
        try:
            current_idx = self.DENOISE_METHODS.index(self.val_denoise_method)
        except ValueError:
            current_idx = 0

        next_idx = (current_idx + 1) % len(self.DENOISE_METHODS)
        self.val_denoise_method = self.DENOISE_METHODS[next_idx]
        self.settingChanged.emit("denoise_method", self.val_denoise_method)
        return self.val_denoise_method

    def set_rating(self, rating):
        self.star_rating_widget.set_rating(rating)

    def _on_hist_mode_changed(self, mode):
        self.histogram_widget.set_mode(mode)
        self.histogramModeChanged.emit(mode)

    def get_all_settings(self):
        settings = {
            "version": 2,
            "temperature": self.val_temperature,
            "tint": self.val_tint,
            "exposure": self.val_exposure,
            "contrast": self.val_contrast,
            "whites": self.val_whites,
            "blacks": self.val_blacks,
            "highlights": self.val_highlights,
            "shadows": self.val_shadows,
            "saturation": self.val_saturation,
            "sharpen_method": "High Quality",
            "sharpen_radius": self.val_sharpen_radius,
            "sharpen_percent": self.val_sharpen_percent,
            "sharpen_value": self.val_sharpen_value,
            "denoise_method": self.val_denoise_method,
            "denoise_luma": self.val_denoise_luma,
            "denoise_chroma": self.val_denoise_chroma,
            "de_haze": self.val_de_haze,
            "rotation": self.rotation,
            "flip_h": self.val_flip_h,
            "flip_v": self.val_flip_v,
            "lens_distortion": self.val_lens_distortion,
            "lens_vignette": self.val_lens_vignette,
            "lens_ca": self.val_lens_ca,
            "lens_camera_override": self.lens_controls.camera_combo.currentText()
            if self.lens_controls.camera_combo.currentIndex() > 0
            else None,
            "lens_name_override": self.lens_controls.lens_combo.currentText()
            if self.lens_controls.lens_combo.currentIndex() > 0
            else None,
            "lens_auto_detect": self.lens_controls.camera_combo.currentIndex() == 0
            and self.lens_controls.lens_combo.currentIndex() == 0,
            "lens_enabled": self.lens_controls.enable_check.isChecked(),
            "lens_autocrop": self.lens_controls.autocrop_check.isChecked(),
        }

        # Collect raw slider values
        for ctrl in [
            self.tone_controls,
            self.color_controls,
            self.detail_controls,
            self.geometry_controls,
            self.lens_controls,
        ]:
            for var_name, slider in ctrl.sliders.items():
                settings[f"raw_{var_name}"] = slider.value()

        return settings

    def apply_settings(self, settings):
        if settings is None:
            return

        version = settings.get("version", 1)

        # 1. Geometry
        self.geometry_controls.btn_flip_h.blockSignals(True)
        self.geometry_controls.btn_flip_h.setChecked(settings.get("flip_h", False))
        self.geometry_controls.btn_flip_h.blockSignals(False)
        self.geometry_controls.btn_flip_v.blockSignals(True)
        self.geometry_controls.btn_flip_v.setChecked(settings.get("flip_v", False))
        self.geometry_controls.btn_flip_v.blockSignals(False)
        self.geometry_controls.val_flip_h = settings.get("flip_h", False)
        self.geometry_controls.val_flip_v = settings.get("flip_v", False)

        # 2. Main Processing Sliders
        slider_vars = [
            "val_temperature",
            "val_tint",
            "val_exposure",
            "val_contrast",
            "val_whites",
            "val_blacks",
            "val_highlights",
            "val_shadows",
            "val_saturation",
            "val_sharpen_value",
            "val_denoise_luma",
            "val_denoise_chroma",
            "val_de_haze",
            "rotation",
            "lens_distortion",
            "lens_vignette",
            "lens_ca",
        ]

        controls_map = {
            "val_exposure": self.tone_controls,
            "val_contrast": self.tone_controls,
            "val_highlights": self.tone_controls,
            "val_shadows": self.tone_controls,
            "val_whites": self.tone_controls,
            "val_blacks": self.tone_controls,
            "val_temperature": self.color_controls,
            "val_tint": self.color_controls,
            "val_saturation": self.color_controls,
            "val_sharpen_value": self.detail_controls,
            "val_denoise_luma": self.detail_controls,
            "val_denoise_chroma": self.detail_controls,
            "val_de_haze": self.detail_controls,
            "rotation": self.geometry_controls,
            "lens_distortion": self.lens_controls,
            "lens_vignette": self.lens_controls,
            "lens_ca": self.lens_controls,
        }

        for var in slider_vars:
            ctrl = controls_map.get(var)
            if not ctrl:
                continue

            raw_key = f"raw_{var}"
            if version >= 2 and raw_key in settings:
                slider = ctrl.sliders.get(var)
                if slider:
                    slider.blockSignals(True)
                    slider.setValue(int(settings[raw_key]))
                    slider.blockSignals(False)

                    logical_val = ctrl.get_value(var)
                    label = ctrl.labels.get(var)
                    if label:
                        label.setText(f"{logical_val:.2f}")
            else:
                key = var.replace("val_", "")
                default_val = (
                    1.0 if key in ["contrast", "saturation", "whites"] else 0.0
                )
                val = settings.get(key, default_val)
                if var == "val_sharpen_value":
                    val = min(50.0, val)
                if var in ["val_denoise_luma", "val_denoise_chroma"]:
                    val = settings.get(key, settings.get("de_noise", 0.0))
                    val = min(50.0, val)
                if var == "val_de_haze":
                    val = min(50.0, val)

                # Special handling for lens settings which don't have val_ prefix in some contexts
                if var in ["lens_distortion", "lens_vignette", "lens_ca"]:
                    val = settings.get(var, 0.0)

                ctrl.set_slider_value(var, val, silent=True)

        if "lens_enabled" in settings:
            self.lens_controls.enable_check.blockSignals(True)
            self.lens_controls.enable_check.setChecked(settings["lens_enabled"])
            self.lens_controls.enable_check.blockSignals(False)

        if "lens_autocrop" in settings:
            self.lens_controls.autocrop_check.blockSignals(True)
            self.lens_controls.autocrop_check.setChecked(settings["lens_autocrop"])
            self.lens_controls.autocrop_check.blockSignals(False)

        # 3. Handle lens overrides
        self.lens_controls.camera_combo.blockSignals(True)
        self.lens_controls.lens_combo.blockSignals(True)

        cam_ovr = settings.get("lens_camera_override")
        if cam_ovr and cam_ovr != "Auto":
            self.lens_controls.camera_combo.setCurrentText(cam_ovr)
        else:
            self.lens_controls.camera_combo.setCurrentIndex(0)

        lens_ovr = settings.get("lens_name_override")
        if lens_ovr and lens_ovr != "Auto":
            self.lens_controls.lens_combo.setCurrentText(lens_ovr)
        else:
            self.lens_controls.lens_combo.setCurrentIndex(0)

        self.lens_controls.camera_combo.blockSignals(False)
        self.lens_controls.lens_combo.blockSignals(False)

        # 4. Special derived parameters
        s_val = self.val_sharpen_value
        self.detail_controls.val_sharpen_radius = 0.5 + (s_val / 100.0) * 2.5
        self.detail_controls.val_sharpen_percent = (s_val / 100.0) * 300.0

        self.val_denoise_method = settings.get(
            "denoise_method", "NLMeans (Numba Fast+)"
        )
