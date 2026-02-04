from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal


class ZoomControls(QtWidgets.QFrame):
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ZoomControls")
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(4, 0, 4, 0)
        self.layout.setSpacing(4)

        # Slider (50 to 400)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(1, 400)
        self.slider.setValue(100)
        self.slider.setFixedWidth(120)
        self.slider.setFixedHeight(20)  # Tighten the slider itself
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.layout.addWidget(self.slider)

        # Percentage Box
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(1, 400)
        self.spin.setValue(100)
        self.spin.setSuffix("%")
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin.setAlignment(Qt.AlignCenter)
        self.spin.setFixedWidth(45)
        self.spin.setFixedHeight(16)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.layout.addWidget(self.spin)

        self.setStyleSheet("""
            QFrame#ZoomControls {
                background-color: rgba(30, 30, 30, 0.85);
                border-radius: 4px;
                border: 1px solid #444;
            }
            QSlider {
                min-height: 0px;
                height: 18px;
            }
            QSpinBox {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 2px;
                color: #ccc;
                font-size: 10px;
                padding: 0px;
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
