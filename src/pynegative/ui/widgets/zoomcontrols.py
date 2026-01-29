from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt, Signal

class ZoomControls(QtWidgets.QFrame):
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ZoomControls")
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(10, 5, 10, 5)
        self.layout.setSpacing(10)

        # Slider (50 to 400)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(50, 400)
        self.slider.setValue(100)
        self.slider.setFixedWidth(120)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.layout.addWidget(self.slider)

        # Percentage Box
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(50, 400)
        self.spin.setValue(100)
        self.spin.setSuffix("%")
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin.setAlignment(Qt.AlignCenter)
        self.spin.setFixedWidth(60)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.layout.addWidget(self.spin)

        self.setStyleSheet("""
            QFrame#ZoomControls {
                background-color: rgba(36, 36, 36, 0.8);
                border-radius: 8px;
                border: 1px solid #404040;
            }
            QSpinBox {
                background-color: #1a1a1a;
                border: 1px solid #303030;
                border-radius: 4px;
                color: #e5e5e5;
            }
        """)

    def _on_slider_changed(self, val):
        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self.zoomChanged.emit(val / 100.0)

    def _on_spin_changed(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)
        self.zoomChanged.emit(val / 100.0)

    def update_zoom(self, scale):
        val = int(scale * 100)
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(val)
        self.spin.setValue(val)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
