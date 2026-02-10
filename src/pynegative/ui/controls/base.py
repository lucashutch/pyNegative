from PySide6 import QtWidgets, QtCore
from ..widgets import ResetableSlider


class BaseControlWidget(QtWidgets.QWidget):
    settingChanged = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sliders = {}
        self.labels = {}
        self.flipped_states = {}

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

        self.flipped_states[var_name] = flipped

        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label_text)

        val_input = QtWidgets.QLineEdit()
        val_input.setText(f"{default:.2f}")
        val_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        val_input.setFixedWidth(60)

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
        slider.default_slider_value = int(default * multiplier)
        slider.setValue(int(default * multiplier))

        def on_slider_change(val):
            actual = val / multiplier
            if flipped:
                actual = max_val + min_val - actual

            val_input.blockSignals(True)
            val_input.setText(f"{actual:.2f}")
            val_input.blockSignals(False)

            if custom_callback:
                custom_callback(actual)
            else:
                setting_name = var_name.replace("val_", "")
                self.settingChanged.emit(setting_name, actual)

        def on_text_changed():
            try:
                text = val_input.text()
                val = float(text)
                val = max(min_val, min(max_val, val))

                slider.blockSignals(True)
                if flipped:
                    slider_val = (max_val + min_val - val) * multiplier
                    slider.setValue(int(slider_val))
                else:
                    slider.setValue(int(val * multiplier))
                slider.blockSignals(False)

                if custom_callback:
                    custom_callback(val)
                else:
                    setting_name = var_name.replace("val_", "")
                    self.settingChanged.emit(setting_name, val)

            except ValueError:
                pass

        slider.valueChanged.connect(on_slider_change)
        val_input.editingFinished.connect(on_text_changed)

        self.sliders[var_name] = slider
        self.labels[var_name] = val_input

        if var_name == "rotation":
            btn_row = QtWidgets.QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)
            btn_row.setSpacing(2)

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

            def adjust_rot(delta):
                current_val = self.get_value(var_name)
                new_val = max(min_val, min(max_val, current_val + delta))
                self.set_slider_value(var_name, new_val)
                # set_slider_value might not trigger signals if silent, but here we want it to.
                # Actually, set_slider_value in EditingControls didn't trigger signals.
                # But adjust_rot in EditingControls called on_slider_change.
                on_slider_change(int(new_val * multiplier))

            btn_minus.clicked.connect(lambda: adjust_rot(-0.1))
            btn_plus.clicked.connect(lambda: adjust_rot(0.1))
            btn_reset.clicked.connect(lambda: adjust_rot(-self.get_value(var_name)))

            btn_row.addWidget(slider)
            btn_row.addWidget(btn_minus)
            btn_row.addWidget(btn_plus)
            btn_row.addWidget(btn_reset)
            layout.addLayout(btn_row)
        else:
            layout.addWidget(slider)

        if section:
            section.add_widget(frame)
        else:
            if self.layout():
                self.layout().addWidget(frame)

        return frame

    def set_slider_value(self, var_name, value, silent=False):
        slider = self.sliders.get(var_name)
        label = self.labels.get(var_name)
        flipped = self.flipped_states.get(var_name, False)

        if silent:
            if slider:
                slider.blockSignals(True)
            if label:
                label.blockSignals(True)

        if slider:
            multiplier = 1000
            if flipped:
                s_min = slider.minimum() / multiplier
                s_max = slider.maximum() / multiplier
                val_to_set = (s_max + s_min) - value
                slider.setValue(int(val_to_set * multiplier))
            else:
                slider.setValue(int(value * multiplier))

            if not silent:
                slider.default_slider_value = slider.value()

        if label:
            label.setText(f"{value:.2f}")

        if silent:
            if slider:
                slider.blockSignals(False)
            if label:
                label.blockSignals(False)

    def get_value(self, var_name):
        slider = self.sliders.get(var_name)
        if not slider:
            return 0.0

        multiplier = 1000.0
        val = slider.value() / multiplier
        if self.flipped_states.get(var_name, False):
            s_min = slider.minimum() / multiplier
            s_max = slider.maximum() / multiplier
            val = s_max + s_min - val
        return val
