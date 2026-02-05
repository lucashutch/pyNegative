from PySide6 import QtWidgets, QtCore, QtGui
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
    cropToggled = QtCore.Signal(bool)
    aspectRatioChanged = QtCore.Signal(float)

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
        self.val_de_haze = 0.0
        self.val_denoise_method = "High Quality"
        self.val_flip_h = False
        self.val_flip_v = False
        self.rotation = 0.0

        # Throttling for rotation slider updates
        self._rotation_slider_throttle_timer = QtCore.QTimer()
        self._rotation_slider_throttle_timer.setSingleShot(True)
        self._rotation_slider_throttle_timer.setInterval(33)  # ~30fps
        self._pending_rotation_value = None
        self._rotation_slider_throttle_timer.timeout.connect(
            self._emit_throttled_rotation
        )

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
        self.tone_section.resetClicked.connect(lambda: self._reset_section("tone"))
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
        self.color_section.resetClicked.connect(lambda: self._reset_section("color"))
        self.controls_layout.addWidget(self.color_section)

        # WB Buttons
        wb_btn_widget = QtWidgets.QWidget()
        wb_btn_layout = QtWidgets.QHBoxLayout(wb_btn_widget)
        wb_btn_layout.setContentsMargins(0, 0, 0, 5)
        wb_btn_layout.setSpacing(8)  # Increase spacing slightly

        wb_btn_layout.addStretch()  # Spacer left

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
        wb_btn_layout.addWidget(self.btn_auto_wb)

        self.btn_as_shot = QtWidgets.QPushButton("As Shot")
        self.btn_as_shot.setStyleSheet("""
            QPushButton {
                min-height: 18px;
                max-height: 20px;
                padding: 2px 8px;
                font-size: 11px;
            }
        """)
        self.btn_as_shot.setFixedWidth(60)  # Consistent width
        self.btn_as_shot.clicked.connect(self._reset_wb)
        wb_btn_layout.addWidget(self.btn_as_shot)

        wb_btn_layout.addStretch()  # Spacer right
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
        self.details_section.resetClicked.connect(
            lambda: self._reset_section("details")
        )
        self.controls_layout.addWidget(self.details_section)

        # Preset Buttons at the top of Details
        preset_widget = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 0, 0, 5)
        preset_layout.setSpacing(8)  # Tighten spacing

        preset_layout.addStretch()  # Spacer left

        btn_style = """
            QPushButton {
                min-height: 18px;
                max-height: 20px;
                padding: 2px 8px;
                font-size: 11px;
            }
        """

        self.btn_low = QtWidgets.QPushButton("Low")
        self.btn_low.setStyleSheet(btn_style)
        self.btn_low.setFixedWidth(60)
        self.btn_low.clicked.connect(lambda: self._apply_preset("low"))

        self.btn_medium = QtWidgets.QPushButton("Medium")
        self.btn_medium.setStyleSheet(btn_style)
        self.btn_medium.setFixedWidth(60)
        self.btn_medium.clicked.connect(lambda: self._apply_preset("medium"))

        self.btn_high = QtWidgets.QPushButton("High")
        self.btn_high.setStyleSheet(btn_style)
        self.btn_high.setFixedWidth(60)
        self.btn_high.clicked.connect(lambda: self._apply_preset("high"))

        preset_layout.addWidget(self.btn_low)
        preset_layout.addWidget(self.btn_medium)
        preset_layout.addWidget(self.btn_high)
        preset_layout.addStretch()  # Spacer right
        self.details_section.add_widget(preset_widget)

        # Mapping function for combined sharpening
        def update_sharpen_params(val):
            # val is 0..50 (reduced from 100)
            # radius: 0.5 to 1.75 (at val=50)
            self.val_sharpen_radius = 0.5 + (val / 100.0) * 2.5
            # percent: 0 to 150 (at val=50)
            self.val_sharpen_percent = (val / 100.0) * 300.0
            self.val_sharpen_value = val
            self.settingChanged.emit("sharpen_value", val)
            self.settingChanged.emit("sharpen_radius", self.val_sharpen_radius)
            self.settingChanged.emit("sharpen_percent", self.val_sharpen_percent)

        self._add_slider(
            "Sharpening",
            0,
            50,
            self.val_sharpen_value,
            "val_sharpen_value",
            1,
            self.details_section,
            custom_callback=update_sharpen_params,
        )

        self._add_slider(
            "De-noise",
            0,
            50,
            self.val_de_noise,
            "val_de_noise",
            1,
            self.details_section,
        )

        self._add_slider(
            "De-haze",
            0,
            50,
            self.val_de_haze,
            "val_de_haze",
            1,
            self.details_section,
        )

        # Denoise Method Selector
        denoise_method_layout = QtWidgets.QHBoxLayout()
        denoise_method_label = QtWidgets.QLabel("Method:")
        denoise_method_label.setStyleSheet("font-size: 11px; color: #aaa;")
        self.denoise_method_combo = QtWidgets.QComboBox()
        self.denoise_method_combo.addItems(["High Quality", "NLMeans"])
        self.denoise_method_combo.setCurrentText(self.val_denoise_method)
        self.denoise_method_combo.setStyleSheet("""
            QComboBox {
                min-height: 18px;
                max-height: 20px;
                font-size: 11px;
                padding: 0px 5px;
            }
        """)
        self.denoise_method_combo.currentTextChanged.connect(
            self._on_denoise_method_changed
        )
        denoise_method_layout.addWidget(denoise_method_label)
        denoise_method_layout.addWidget(self.denoise_method_combo)
        self.details_section.add_layout(denoise_method_layout)

        # 6. Geometry
        self.geometry_section = CollapsibleSection("Geometry")
        self.geometry_section.resetClicked.connect(
            lambda: self._reset_section("geometry")
        )
        self.controls_layout.addWidget(self.geometry_section)

        # Crop controls layout
        crop_widget = QtWidgets.QWidget()
        crop_layout = QtWidgets.QHBoxLayout(crop_widget)
        crop_layout.setContentsMargins(0, 0, 0, 0)
        crop_layout.setSpacing(5)

        # Crop Button
        self.crop_btn = QtWidgets.QPushButton("Crop Tool")
        self.crop_btn.setCheckable(True)
        self.crop_btn.setFixedWidth(80)
        self.crop_btn.setStyleSheet("""
             QPushButton {
                 min-height: 18px;
                 max-height: 20px;
                 padding: 2px 8px;
                 font-size: 11px;
             }
             QPushButton:checked {
                 background-color: #9C27B0;
                 color: white;
                 border: 1px solid #7B1FA2;
             }
        """)

        # Aspect Ratio Selector
        self.aspect_ratio_combo = QtWidgets.QComboBox()
        self.aspect_ratio_combo.setEditable(True)
        self.aspect_ratio_combo.lineEdit().setReadOnly(True)
        self.aspect_ratio_combo.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.aspect_ratio_combo.addItems(["Unlocked", "1:1", "4:3", "3:2", "16:9"])
        for i in range(self.aspect_ratio_combo.count()):
            self.aspect_ratio_combo.setItemData(
                i, QtCore.Qt.AlignCenter, QtCore.Qt.TextAlignmentRole
            )

        self.aspect_ratio_combo.setToolTip("Lock aspect ratio")
        self.aspect_ratio_combo.setFixedWidth(85)
        self.aspect_ratio_combo.setStyleSheet("""
            QComboBox {
                min-height: 18px;
                max-height: 20px;
                font-size: 11px;
                padding: 0px;
            }
            QComboBox QLineEdit {
                background: transparent;
                border: none;
                color: #ccc;
                font-size: 11px;
                text-align: center;
            }
        """)
        self.aspect_ratio_combo.currentIndexChanged.connect(
            self._on_aspect_ratio_changed
        )

        # Flip Buttons (created early to be added to crop_layout)
        self.btn_flip_h = QtWidgets.QPushButton()
        self.btn_flip_v = QtWidgets.QPushButton()

        for btn, name, is_h in [
            (self.btn_flip_h, "Horizontal", True),
            (self.btn_flip_v, "Vertical", False),
        ]:
            btn.setCheckable(True)
            btn.setFixedSize(26, 18)
            btn.setToolTip(f"Flip {name}")
            # Create Icon
            pixmap = QtGui.QPixmap(32, 32)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor("#ccc"), 2)
            painter.setPen(pen)

            # Draw Mirroring triangles icon
            if is_h:
                # Horizontal Flip (Mirror across vertical axis)
                painter.drawLine(16, 6, 16, 26)  # Axis
                # Left triangle
                tri_left = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(14, 10),
                        QtCore.QPointF(4, 16),
                        QtCore.QPointF(14, 22),
                    ]
                )
                # Right triangle
                tri_right = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(18, 10),
                        QtCore.QPointF(28, 16),
                        QtCore.QPointF(18, 22),
                    ]
                )

                painter.setBrush(QtGui.QColor("#ccc"))
                painter.drawPolygon(tri_left)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawPolygon(tri_right)
            else:
                # Vertical Flip (Mirror across horizontal axis)
                painter.drawLine(6, 16, 26, 16)  # Axis
                # Top triangle
                tri_top = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(10, 14),
                        QtCore.QPointF(16, 4),
                        QtCore.QPointF(22, 14),
                    ]
                )
                # Bottom triangle
                tri_bottom = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(10, 18),
                        QtCore.QPointF(16, 28),
                        QtCore.QPointF(22, 18),
                    ]
                )

                painter.setBrush(QtGui.QColor("#ccc"))
                painter.drawPolygon(tri_top)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawPolygon(tri_bottom)

            painter.end()
            btn.setIcon(QtGui.QIcon(pixmap))
            btn.setIconSize(QtCore.QSize(14, 14))

            btn.setStyleSheet("""
                QPushButton {
                    background-color: #333;
                    border: 1px solid #444;
                    padding: 0px;
                    min-height: 18px;
                    max-height: 18px;
                }
                QPushButton:checked {
                    background-color: #6366f1;
                    border-color: #8b5cf6;
                }
                QPushButton:hover {
                    background-color: #444;
                }
            """)

        def on_flip_h_toggled(checked):
            self.val_flip_h = checked
            self.settingChanged.emit("flip_h", checked)

        def on_flip_v_toggled(checked):
            self.val_flip_v = checked
            self.settingChanged.emit("flip_v", checked)

        self.btn_flip_h.toggled.connect(on_flip_h_toggled)
        self.btn_flip_v.toggled.connect(on_flip_v_toggled)

        def on_crop_toggled(checked):
            if checked:
                self.crop_btn.setText("Done")
                self.aspect_ratio_combo.show()
                self.btn_flip_h.show()
                self.btn_flip_v.show()
                if hasattr(self, "rotation_frame"):
                    self.rotation_frame.show()
            else:
                self.crop_btn.setText("Crop Tool")
                self.aspect_ratio_combo.hide()
                self.btn_flip_h.hide()
                self.btn_flip_v.hide()
                if hasattr(self, "rotation_frame"):
                    self.rotation_frame.hide()
            self.cropToggled.emit(checked)

        self.crop_btn.toggled.connect(on_crop_toggled)

        crop_layout.addWidget(self.crop_btn)
        crop_layout.addStretch()
        crop_layout.addWidget(self.btn_flip_h)
        crop_layout.addWidget(self.btn_flip_v)
        crop_layout.addWidget(self.aspect_ratio_combo)

        # Hide elements initially
        self.aspect_ratio_combo.hide()
        self.btn_flip_h.hide()
        self.btn_flip_v.hide()

        self.geometry_section.add_widget(crop_widget)

        # Rotation Slider
        self._add_slider(
            "Rotation",
            -45.0,
            45.0,
            0.0,
            "rotation",
            0.1,
            section=self.geometry_section,
            unit="deg",
        )
        # Hide initially per user request
        if hasattr(self, "rotation_frame"):
            self.rotation_frame.hide()

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
        unit="",
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

        # Editable Value (DoubleSpinBox style or LineEdit)
        # Using a QLineEdit that validates input
        val_input = (
            QtWidgets.QLineEdit()
        )  # No QDoubleSpinBox style for now to keep it minimal
        val_input.setText(f"{default:.2f}")
        val_input.setAlignment(QtCore.Qt.AlignRight)
        val_input.setFixedWidth(60)

        # Unit Label
        unit_lbl = None
        if unit:
            unit_lbl = QtWidgets.QLabel(unit)
            unit_lbl.setStyleSheet("color: #888; font-size: 11px;")

        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(val_input)
        if unit_lbl:
            row.addWidget(unit_lbl)

        layout.addLayout(row)

        slider = ResetableSlider(QtCore.Qt.Horizontal)
        multiplier = 1000
        slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
        # Default value on initial setup
        slider.default_slider_value = int(default * multiplier)
        slider.setValue(int(default * multiplier))

        def on_slider_change(val):
            actual = val / multiplier
            if flipped:
                # Map slider min..max to max..min
                # Formula: actual = s_max + s_min - actual
                actual = max_val + min_val - actual

            # Update input without triggering signal loop if possible
            val_input.blockSignals(True)
            val_input.setText(f"{actual:.2f}")
            val_input.blockSignals(False)

            setattr(self, var_name, actual)

            if custom_callback:
                custom_callback(actual)
            else:
                # Extract setting name from var_name
                setting_name = var_name.replace("val_", "")

                # Throttle rotation updates to 30fps
                if setting_name == "rotation":
                    self._pending_rotation_value = actual
                    if not self._rotation_slider_throttle_timer.isActive():
                        self._rotation_slider_throttle_timer.start()
                else:
                    self.settingChanged.emit(setting_name, actual)

        def on_text_changed():
            try:
                text = val_input.text()
                val = float(text)

                # Clamp value
                val = max(min_val, min(max_val, val))

                # Update slider
                slider.blockSignals(True)
                if flipped:
                    # val = s_max + s_min - slider_val (unscaled)
                    # slider_val = s_max + s_min - val
                    # BUT slider is int scaled
                    slider_val = (max_val + min_val - val) * multiplier
                    slider.setValue(int(slider_val))
                else:
                    slider.setValue(int(val * multiplier))
                slider.blockSignals(False)

                setattr(self, var_name, val)

                if custom_callback:
                    custom_callback(val)
                else:
                    setting_name = var_name.replace("val_", "")
                    self.settingChanged.emit(setting_name, val)

            except ValueError:
                pass  # Ignore invalid float

        slider.valueChanged.connect(on_slider_change)
        val_input.editingFinished.connect(on_text_changed)

        # Store refs
        setattr(self, f"{var_name}_slider", slider)
        setattr(self, f"{var_name}_label", val_input)  # Store input for updates

        # Rotation Specific: Add +/- buttons if requested (detected by var_name="rotation")
        # Or generalize if needed. User asked specifically for rotation.
        if var_name == "rotation":
            # Add buttons row
            btn_row = QtWidgets.QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)

            btn_minus = QtWidgets.QPushButton("-")
            btn_plus = QtWidgets.QPushButton("+")
            btn_reset = QtWidgets.QPushButton("Reset")

            for b in [btn_minus, btn_plus, btn_reset]:
                b.setFixedSize(22, 14)
                b.setStyleSheet("""
                    QPushButton {
                        padding: 0px;
                        margin: 0px;
                        font-size: 10px;
                        border: 1px solid #444;
                        background-color: #333;
                        color: #ccc;
                        min-height: 0px;
                        max-height: 14px;
                    }
                    QPushButton:hover {
                        background-color: #444;
                        border: 1px solid #555;
                    }
                """)

            btn_reset.setFixedWidth(34)
            btn_row.setSpacing(2)

            def adjust_rot(delta):
                new_val = getattr(self, var_name, 0.0) + delta
                new_val = max(min_val, min(max_val, new_val))

                # Update slider -> triggers everything else
                slider.blockSignals(True)
                slider.setValue(int(new_val * multiplier))
                slider.blockSignals(False)
                on_slider_change(int(new_val * multiplier))  # Force update

            btn_minus.clicked.connect(lambda: adjust_rot(-0.1))
            btn_plus.clicked.connect(lambda: adjust_rot(0.1))
            btn_reset.clicked.connect(
                lambda: adjust_rot(-getattr(self, var_name, 0.0))
            )  # Reset to 0

            btn_row.addWidget(slider)
            btn_row.addWidget(btn_minus)
            btn_row.addWidget(btn_plus)
            btn_row.addWidget(btn_reset)
            layout.addLayout(btn_row)

        else:
            layout.addWidget(slider)

        # Store frame ref
        setattr(self, f"{var_name}_frame", frame)

        if section:
            section.add_widget(frame)
        else:
            self.controls_layout.addWidget(frame)

    def set_slider_value(self, var_name, value, silent=False):
        """Set slider value programmatically, optionally without triggering signals."""
        slider = getattr(self, f"{var_name}_slider", None)
        label = getattr(self, f"{var_name}_label", None)
        flipped = getattr(self, f"{var_name}_flipped", False)

        if silent and slider:
            slider.blockSignals(True)
        if silent and label:
            label.blockSignals(True)

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
            if not silent:
                slider.default_slider_value = slider.value()

        if label:
            label.setText(f"{value:.2f}")
        setattr(self, var_name, value)

        if silent and slider:
            slider.blockSignals(False)
        if silent and label:
            label.blockSignals(False)

    def set_crop_checked(self, checked):
        if self.crop_btn:
            self.crop_btn.setChecked(checked)

    def reset_sliders(self):
        """Reset all sliders to their default values."""
        for attr_name in dir(self):
            if attr_name.endswith("_slider"):
                slider = getattr(self, attr_name)
                if hasattr(slider, "default_slider_value"):
                    slider.setValue(slider.default_slider_value)

    def _reset_section(self, section_name):
        """Reset all parameters within a specific section."""
        params_to_reset = []
        if section_name == "tone":
            params_to_reset = [
                ("val_exposure", 0.0, "exposure"),
                ("val_contrast", 1.0, "contrast"),
                ("val_highlights", 0.0, "highlights"),
                ("val_shadows", 0.0, "shadows"),
                ("val_whites", 1.0, "whites"),
                ("val_blacks", 0.0, "blacks"),
            ]
        elif section_name == "color":
            params_to_reset = [
                ("val_temperature", 0.0, "temperature"),
                ("val_tint", 0.0, "tint"),
                ("val_saturation", 1.0, "saturation"),
            ]
        elif section_name == "details":
            params_to_reset = [
                ("val_sharpen_value", 20.0, "sharpen_value"),
                ("val_de_noise", 0.0, "de_noise"),
                ("val_de_haze", 0.0, "de_haze"),
            ]
        elif section_name == "geometry":
            params_to_reset = [
                ("rotation", 0.0, "rotation"),
            ]
            self.btn_flip_h.setChecked(False)
            self.btn_flip_v.setChecked(False)
            self.val_flip_h = False
            self.val_flip_v = False
            self.settingChanged.emit("flip_h", False)
            self.settingChanged.emit("flip_v", False)

            # Special case for crop: reset to full image
            self.settingChanged.emit("crop", None)

        for var_name, default, setting_name in params_to_reset:
            self.set_slider_value(var_name, default)
            self.settingChanged.emit(setting_name, default)

    def set_rating(self, rating):
        """Set the star rating."""
        self.star_rating_widget.set_rating(rating)

    def set_save_enabled(self, enabled):
        """No-op as save button is removed."""
        pass

    def _on_rating_changed(self, rating):
        """Handle rating change."""
        self.ratingChanged.emit(rating)

    def _emit_throttled_rotation(self):
        """Emit the pending rotation value (throttled to 30fps)."""
        if self._pending_rotation_value is not None:
            self.settingChanged.emit("rotation", self._pending_rotation_value)
            self._pending_rotation_value = None

    def _on_hist_mode_changed(self, mode):
        """Handle histogram mode change."""
        self.histogram_widget.set_mode(mode)
        self.histogramModeChanged.emit(mode)

    def _on_denoise_method_changed(self, method):
        """Handle denoise method change."""
        self.val_denoise_method = method
        self.settingChanged.emit("denoise_method", method)

    def _on_aspect_ratio_changed(self, index):
        """Handle aspect ratio selection change."""
        text = self.aspect_ratio_combo.currentText()
        ratio = 0.0
        if text == "1:1":
            ratio = 1.0
        elif text == "4:3":
            ratio = 4.0 / 3.0
        elif text == "3:2":
            ratio = 3.0 / 2.0
        elif text == "16:9":
            ratio = 16.0 / 9.0

        self.aspectRatioChanged.emit(ratio)

    def _reset_wb(self):
        """Reset WB sliders to 0.0."""
        self.set_slider_value("val_temperature", 0.0)
        self.set_slider_value("val_tint", 0.0)
        self.settingChanged.emit("temperature", 0.0)
        self.settingChanged.emit("tint", 0.0)

    def _apply_preset(self, preset_type):
        """Apply preset values for sharpening and denoising."""
        if preset_type == "low":
            self.set_slider_value("val_sharpen_value", 15.0)
            self.set_slider_value("val_de_noise", 2.0)
        elif preset_type == "medium":
            self.set_slider_value("val_sharpen_value", 30.0)
            self.set_slider_value("val_de_noise", 7.0)
        elif preset_type == "high":
            self.set_slider_value("val_sharpen_value", 50.0)
            self.set_slider_value("val_de_noise", 12.0)

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
            "denoise_method": self.val_denoise_method,
            "de_noise": self.val_de_noise,
            "de_haze": self.val_de_haze,
            "rotation": getattr(self, "rotation", 0.0),
            "flip_h": self.val_flip_h,
            "flip_v": self.val_flip_v,
        }

    def apply_settings(self, settings):
        """Apply settings from a dictionary."""
        self.set_slider_value(
            "val_temperature", settings.get("temperature", 0.0), silent=True
        )
        self.set_slider_value("val_tint", settings.get("tint", 0.0), silent=True)
        self.set_slider_value(
            "val_exposure", settings.get("exposure", 0.0), silent=True
        )
        self.set_slider_value(
            "val_contrast", settings.get("contrast", 1.0), silent=True
        )
        self.set_slider_value("val_whites", settings.get("whites", 1.0), silent=True)
        self.set_slider_value("val_blacks", settings.get("blacks", 0.0), silent=True)
        self.set_slider_value(
            "val_highlights", settings.get("highlights", 0.0), silent=True
        )
        self.set_slider_value("val_shadows", settings.get("shadows", 0.0), silent=True)
        self.set_slider_value(
            "val_saturation", settings.get("saturation", 1.0), silent=True
        )

        # Geometry
        self.set_slider_value("rotation", settings.get("rotation", 0.0), silent=True)

        self.btn_flip_h.blockSignals(True)
        self.btn_flip_h.setChecked(settings.get("flip_h", False))
        self.btn_flip_h.blockSignals(False)

        self.btn_flip_v.blockSignals(True)
        self.btn_flip_v.setChecked(settings.get("flip_v", False))
        self.btn_flip_v.blockSignals(False)

        self.val_flip_h = settings.get("flip_h", False)
        self.val_flip_v = settings.get("flip_v", False)

        sharpen_val = settings.get("sharpen_value", 0.0)
        if sharpen_val is not None:
            # Clamp to new max of 50
            sharpen_val = min(50.0, sharpen_val)
            self.set_slider_value("val_sharpen_value", sharpen_val, silent=True)
            # Update derived sharpening parameters using the scale factor of 100 for compatibility
            self.val_sharpen_radius = 0.5 + (sharpen_val / 100.0) * 2.5
            self.val_sharpen_percent = (sharpen_val / 100.0) * 300.0

        denoise_val = settings.get("de_noise", 0)
        self.set_slider_value("val_de_noise", min(50.0, denoise_val), silent=True)

        de_haze_val = settings.get("de_haze", 0)
        self.set_slider_value("val_de_haze", min(50.0, de_haze_val), silent=True)

        denoise_method = settings.get("denoise_method", "High Quality")
        self.denoise_method_combo.blockSignals(True)
        self.denoise_method_combo.setCurrentText(denoise_method)
        self.denoise_method_combo.blockSignals(False)
        self.val_denoise_method = denoise_method
