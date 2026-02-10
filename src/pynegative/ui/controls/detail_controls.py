from PySide6 import QtWidgets, QtCore
from ..widgets import CollapsibleSection
from .base import BaseControlWidget


class DetailControls(BaseControlWidget):
    presetApplied = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.val_sharpen_radius = 0.5
        self.val_sharpen_percent = 0.0
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.details_section = CollapsibleSection("DETAILS", expanded=False)
        self.details_section.resetClicked.connect(self.reset_section)
        layout.addWidget(self.details_section)

        # Preset Buttons
        preset_widget = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 0, 0, 5)
        preset_layout.setSpacing(8)
        preset_layout.addStretch()

        btn_style = """
            QPushButton {
                min-height: 18px;
                max-height: 20px;
                padding: 2px 8px;
                font-size: 11px;
            }
        """

        for name in ["Low", "Medium", "High"]:
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet(btn_style)
            btn.setFixedWidth(60)
            btn.clicked.connect(
                lambda checked=False, n=name.lower(): self._apply_preset(n)
            )
            preset_layout.addWidget(btn)

        preset_layout.addStretch()
        self.details_section.add_widget(preset_widget)

        def update_sharpen_params(val):
            self.val_sharpen_radius = 0.5 + (val / 100.0) * 2.5
            self.val_sharpen_percent = (val / 100.0) * 300.0
            self.settingChanged.emit("sharpen_value", val)
            self.settingChanged.emit("sharpen_radius", self.val_sharpen_radius)
            self.settingChanged.emit("sharpen_percent", self.val_sharpen_percent)

        self._add_slider(
            "Sharpening",
            0,
            50,
            0.0,
            "val_sharpen_value",
            1,
            self.details_section,
            custom_callback=update_sharpen_params,
        )
        self._add_slider("De-haze", 0, 50, 0.0, "val_de_haze", 1, self.details_section)
        self._add_slider(
            "Luma Denoise", 0, 50, 0.0, "val_denoise_luma", 1, self.details_section
        )
        self._add_slider(
            "Chroma Denoise", 0, 50, 0.0, "val_denoise_chroma", 1, self.details_section
        )

    def reset_section(self):
        params = [
            ("val_sharpen_value", 0.0, "sharpen_value"),
            ("val_denoise_luma", 0.0, "denoise_luma"),
            ("val_denoise_chroma", 0.0, "denoise_chroma"),
            ("val_de_haze", 0.0, "de_haze"),
        ]
        for var_name, default, setting_name in params:
            self.set_slider_value(var_name, default)
            self.settingChanged.emit(setting_name, default)

    def _apply_preset(self, preset_type):
        if preset_type == "low":
            self.set_slider_value("val_sharpen_value", 15.0)
            self.set_slider_value("val_denoise_luma", 2.0)
            self.set_slider_value("val_denoise_chroma", 2.0)
            self.settingChanged.emit("denoise_luma", 2.0)
            self.settingChanged.emit("denoise_chroma", 2.0)
        elif preset_type == "medium":
            self.set_slider_value("val_sharpen_value", 30.0)
            self.set_slider_value("val_denoise_luma", 7.0)
            self.set_slider_value("val_denoise_chroma", 7.0)
            self.settingChanged.emit("denoise_luma", 7.0)
            self.settingChanged.emit("denoise_chroma", 7.0)
        elif preset_type == "high":
            self.set_slider_value("val_sharpen_value", 50.0)
            self.set_slider_value("val_denoise_luma", 12.0)
            self.set_slider_value("val_denoise_chroma", 12.0)
            self.settingChanged.emit("denoise_luma", 12.0)
            self.settingChanged.emit("denoise_chroma", 12.0)

        self.presetApplied.emit(preset_type)
