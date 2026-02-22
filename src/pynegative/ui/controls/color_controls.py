from PySide6 import QtWidgets, QtCore
from ..widgets import CollapsibleSection
from .base import BaseControlWidget


class ColorControls(BaseControlWidget):
    autoWbRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.color_section = CollapsibleSection("COLOR", expanded=True)
        self.color_section.resetClicked.connect(self.reset_section)
        layout.addWidget(self.color_section)

        # WB Buttons
        wb_btn_widget = QtWidgets.QWidget()
        wb_btn_layout = QtWidgets.QHBoxLayout(wb_btn_widget)
        wb_btn_layout.setContentsMargins(0, 0, 0, 5)
        wb_btn_layout.setSpacing(8)

        wb_btn_layout.addStretch()

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
        self.btn_as_shot.setFixedWidth(60)
        self.btn_as_shot.clicked.connect(self.reset_wb)
        wb_btn_layout.addWidget(self.btn_as_shot)

        wb_btn_layout.addStretch()
        self.color_section.add_widget(wb_btn_widget)

        self._add_slider(
            "Temperature", -1.0, 1.0, 0.0, "val_temperature", 0.01, self.color_section
        )
        self._add_slider("Tint", -1.0, 1.0, 0.0, "val_tint", 0.01, self.color_section)
        self._add_slider(
            "Saturation", -1.0, 1.0, 0.0, "val_saturation", 0.01, self.color_section
        )

    def reset_section(self):
        params = [
            ("val_temperature", 0.0, "temperature"),
            ("val_tint", 0.0, "tint"),
            ("val_saturation", 0.0, "saturation"),
        ]
        for var_name, default, setting_name in params:
            self.set_slider_value(var_name, default)
            self.settingChanged.emit(setting_name, default)

    def reset_wb(self):
        self.set_slider_value("val_temperature", 0.0)
        self.set_slider_value("val_tint", 0.0)
        self.settingChanged.emit("temperature", 0.0)
        self.settingChanged.emit("tint", 0.0)
