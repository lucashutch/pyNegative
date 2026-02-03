from PySide6 import QtWidgets, QtCore
from .widgets import (
    CollapsibleSection,
    ResetableSlider,
    StarRatingWidget,
    HistogramWidget,
)


class EditingControls(QtWidgets.QWidget):
    # Signals for changes
    settingChanged = QtCore.Signal(str, object)  # setting_name, value
    ratingChanged = QtCore.Signal(int)
    presetApplied = QtCore.Signal(str)
    autoWbRequested = QtCore.Signal()
    histogramModeChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Processing parameter values
        self.val_temperature = 0.0
        self.val_tint = 0.0
        self.val_exposure = 0.0
        self.val_contrast = 1.0
        self.val_whites = 1.0
        self.val_blacks = 0.0
        self.val_highlights = 0.0
        self.val_shadows = 0.0
        self.val_saturation = 1.0
        self.val_sharpen_value = 0.0
        self.val_sharpen_radius = 0.5
        self.val_sharpen_percent = 0.0
        self.val_de_noise = 0

        self._init_ui()

    def _init_ui(self):
        # Wrap everything in a scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

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
                pass  # Ignore errors during initialization

        # Connect to scrollbar visibility change
        scroll.verticalScrollBar().rangeChanged.connect(_update_scroll_margins)
        scroll.verticalScrollBar().valueChanged.connect(_update_scroll_margins)

        # Force initial check after UI is built
        QtCore.QTimer.singleShot(100, _update_scroll_margins)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        # --- Histogram Section (At Top) ---
        self.histogram_section = CollapsibleSection("HISTOGRAM", expanded=False)
        self.controls_layout.addWidget(self.histogram_section)

        self.histogram_widget = HistogramWidget()
        self.histogram_section.add_widget(self.histogram_widget)

        # Histogram Mode Selector
        self.hist_mode_combo = QtWidgets.QComboBox()
        self.hist_mode_combo.addItems(["Auto", "Luminance", "RGB", "YUV"])
        self.hist_mode_combo.currentTextChanged.connect(self._on_hist_mode_changed)
        self.histogram_section.add_widget(self.hist_mode_combo)

        # --- Rating Section ---
        self.rating_section = CollapsibleSection("RATING", expanded=True)
        self.controls_layout.addWidget(self.rating_section)
        self.star_rating_widget = StarRatingWidget()
        self.star_rating_widget.ratingChanged.connect(self._on_rating_changed)
        self.rating_section.add_widget(self.star_rating_widget)

        # --- Tone Section ---
        self.tone_section = CollapsibleSection("TONE", expanded=True)
        self.controls_layout.addWidget(self.tone_section)

        self._add_slider(
            "Exposure",
            -4.0,
            4.0,
            self.val_exposure,
            "val_exposure",
            0.01,
            self.tone_section,
        )
        self._add_slider(
            "Contrast",
            0.5,
            2.0,
            self.val_contrast,
            "val_contrast",
            0.01,
            self.tone_section,
        )
        self._add_slider(
            "Highlights",
            -1.0,
            1.0,
            self.val_highlights,
            "val_highlights",
            0.01,
            self.tone_section,
        )
        self._add_slider(
            "Shadows",
            -1.0,
            1.0,
            self.val_shadows,
            "val_shadows",
            0.01,
            self.tone_section,
        )
        self._add_slider(
            "Whites",
            0.5,
            1.5,
            self.val_whites,
            "val_whites",
            0.01,
            self.tone_section,
            flipped=True,
        )
        self._add_slider(
            "Blacks",
            -0.2,
            0.2,
            self.val_blacks,
            "val_blacks",
            0.001,
            self.tone_section,
            flipped=True,
        )

        # --- Color Section ---
        self.color_section = CollapsibleSection("COLOR", expanded=True)
        self.controls_layout.addWidget(self.color_section)

        # WB Buttons
        wb_btn_widget = QtWidgets.QWidget()
        wb_btn_layout = QtWidgets.QHBoxLayout(wb_btn_widget)
        wb_btn_layout.setContentsMargins(0, 0, 0, 5)
        wb_btn_layout.setSpacing(4)

        self.btn_auto_wb = QtWidgets.QPushButton("Auto")
        self.btn_auto_wb.setStyleSheet("""
            QPushButton {
                min-height: 18px;
                max-height: 20px;
                padding: 2px 8px;
                font-size: 11px;
            }
        """)
        self.btn_auto_wb.setFixedWidth(60)
        self.btn_auto_wb.clicked.connect(self.autoWbRequested.emit)

        self.btn_as_shot = QtWidgets.QPushButton("As Shot")
        self.btn_as_shot.setStyleSheet("""
            QPushButton {
                min-height: 18px;
                max-height: 20px;
                padding: 2px 8px;
                font-size: 11px;
            }
        """)
        self.btn_as_shot.setFixedWidth(70)
        self.btn_as_shot.clicked.connect(self._reset_wb)

        wb_btn_layout.addWidget(self.btn_auto_wb)
        wb_btn_layout.addWidget(self.btn_as_shot)
        wb_btn_layout.addStretch()
        self.color_section.add_widget(wb_btn_widget)

        self._add_slider(
            "Temperature",
            -1.0,
            1.0,
            self.val_temperature,
            "val_temperature",
            0.01,
            self.color_section,
        )
        self._add_slider(
            "Tint",
            -1.0,
            1.0,
            self.val_tint,
            "val_tint",
            0.01,
            self.color_section,
        )
        self._add_slider(
            "Saturation",
            0.0,
            2.0,
            self.val_saturation,
            "val_saturation",
            0.01,
            self.color_section,
        )

        # --- Details Section ---
        self.details_section = CollapsibleSection("DETAILS", expanded=False)
        self.controls_layout.addWidget(self.details_section)

        # Preset Buttons at the top of Details
        preset_widget = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 5, 0, 5)
        preset_layout.setSpacing(4)  # Tighten spacing
        self.btn_low = QtWidgets.QPushButton("Low")
        self.btn_low.setProperty("label", "Low")
        self.btn_low.clicked.connect(lambda: self._apply_preset("low"))
        self.btn_medium = QtWidgets.QPushButton("Medium")
        self.btn_medium.setProperty("label", "Medium")
        self.btn_medium.clicked.connect(lambda: self._apply_preset("medium"))
        self.btn_high = QtWidgets.QPushButton("High")
        self.btn_high.setProperty("label", "High")
        self.btn_high.clicked.connect(lambda: self._apply_preset("high"))

        preset_layout.addWidget(self.btn_low)
        preset_layout.addWidget(self.btn_medium)
        preset_layout.addWidget(self.btn_high)
        self.details_section.add_widget(preset_widget)

        # Mapping function for combined sharpening
        def update_sharpen_params(val):
            # val is 0..100
            self.val_sharpen_radius = 0.5 + (val / 100.0) * 0.75
            self.val_sharpen_percent = (val / 100.0) * 150.0
            self.val_sharpen_value = val
            self.settingChanged.emit("sharpen_value", val)
            self.settingChanged.emit("sharpen_radius", self.val_sharpen_radius)
            self.settingChanged.emit("sharpen_percent", self.val_sharpen_percent)

        self._add_slider(
            "Sharpening",
            0,
            100,
            self.val_sharpen_value,
            "val_sharpen_value",
            1,
            self.details_section,
            custom_callback=update_sharpen_params,
        )

        self._add_slider(
            "De-noise",
            0,
            20,
            self.val_de_noise,
            "val_de_noise",
            1,
            self.details_section,
        )

        self.controls_layout.addStretch()

    def _add_slider(
        self,
        label_text,
        min_val,
        max_val,
        default,
        var_name,
        step_size,
        section=None,
        flipped=False,
        custom_callback=None,
    ):
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

            if custom_callback:
                custom_callback(actual)
            else:
                # Extract setting name from var_name
                setting_name = var_name.replace("val_", "")
                self.settingChanged.emit(setting_name, actual)

        slider.valueChanged.connect(on_change)

        # Store refs
        setattr(self, f"{var_name}_slider", slider)
        setattr(self, f"{var_name}_label", val_lbl)  # Store label for updates

        layout.addWidget(slider)
        if section:
            section.add_widget(frame)
        else:
            self.controls_layout.addWidget(frame)

    def set_slider_value(self, var_name, value):
        """Set slider value programmatically."""
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

    def reset_sliders(self):
        """Reset all sliders to their default values."""
        for attr_name in dir(self):
            if attr_name.endswith("_slider"):
                slider = getattr(self, attr_name)
                if hasattr(slider, "default_slider_value"):
                    slider.setValue(slider.default_slider_value)

    def set_rating(self, rating):
        """Set the star rating."""
        self.star_rating_widget.set_rating(rating)

    def set_save_enabled(self, enabled):
        """No-op as save button is removed."""
        pass

    def _on_rating_changed(self, rating):
        """Handle rating change."""
        self.ratingChanged.emit(rating)

    def _on_hist_mode_changed(self, mode):
        """Handle histogram mode change."""
        self.histogram_widget.set_mode(mode)
        self.histogramModeChanged.emit(mode)

    def _reset_wb(self):
        """Reset WB sliders to 0.0."""
        self.set_slider_value("val_temperature", 0.0)
        self.set_slider_value("val_tint", 0.0)
        self.settingChanged.emit("temperature", 0.0)
        self.settingChanged.emit("tint", 0.0)

    def _apply_preset(self, preset_type):
        """Apply preset values for sharpening and denoising."""
        if preset_type == "low":
            self.set_slider_value("val_sharpen_value", 30.0)
            self.set_slider_value("val_de_noise", 5.0)
        elif preset_type == "medium":
            self.set_slider_value("val_sharpen_value", 60.0)
            self.set_slider_value("val_de_noise", 15.0)
        elif preset_type == "high":
            self.set_slider_value("val_sharpen_value", 100.0)
            self.set_slider_value("val_de_noise", 25.0)

        self.presetApplied.emit(preset_type)

    def get_all_settings(self):
        """Get all current settings as a dictionary."""
        return {
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
            "denoise_method": "High Quality",
            "de_noise": self.val_de_noise,
        }

    def apply_settings(self, settings):
        """Apply settings from a dictionary."""
        self.set_slider_value("val_temperature", settings.get("temperature", 0.0))
        self.set_slider_value("val_tint", settings.get("tint", 0.0))
        self.set_slider_value("val_exposure", settings.get("exposure", 0.0))
        self.set_slider_value("val_contrast", settings.get("contrast", 1.0))
        self.set_slider_value("val_whites", settings.get("whites", 1.0))
        self.set_slider_value("val_blacks", settings.get("blacks", 0.0))
        self.set_slider_value("val_highlights", settings.get("highlights", 0.0))
        self.set_slider_value("val_shadows", settings.get("shadows", 0.0))
        self.set_slider_value("val_saturation", settings.get("saturation", 1.0))

        sharpen_val = settings.get("sharpen_value", 0.0)
        if sharpen_val is not None:
            self.set_slider_value("val_sharpen_value", sharpen_val)

        self.set_slider_value("val_de_noise", settings.get("de_noise", 0))
