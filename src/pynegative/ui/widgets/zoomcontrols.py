from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal


class ZoomControls(QtWidgets.QFrame):
    zoomChanged = Signal(float)
    zoomInClicked = Signal()
    zoomOutClicked = Signal()
    fitClicked = Signal()
    zoomPresetSelected = Signal(float)

    _BTN_H = 30

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ZoomControls")
        self.setFixedHeight(40)
        self.setFixedWidth(280)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(6, 0, 6, 0)
        main_layout.setSpacing(4)

        # [-] Zoom Out
        self.btn_out = QtWidgets.QPushButton("\u2212")
        self.btn_out.setObjectName("ZoomBtn")
        self.btn_out.setFixedSize(34, self._BTN_H)
        self.btn_out.clicked.connect(self.zoomOutClicked.emit)
        main_layout.addWidget(self.btn_out, 0, Qt.AlignVCenter)

        # Slider
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(1, 400)
        self.slider.setValue(100)
        self.slider.setFixedHeight(23)
        self.slider.valueChanged.connect(self._on_slider_changed)
        main_layout.addWidget(self.slider, 1, Qt.AlignVCenter)

        # [+] Zoom In
        self.btn_in = QtWidgets.QPushButton("+")
        self.btn_in.setObjectName("ZoomBtn")
        self.btn_in.setFixedSize(34, self._BTN_H)
        self.btn_in.clicked.connect(self.zoomInClicked.emit)
        main_layout.addWidget(self.btn_in, 0, Qt.AlignVCenter)

        # Preset dropdown (shows current %)
        self.btn_preset = QtWidgets.QPushButton("100%")
        self.btn_preset.setObjectName("ZoomPresetBtn")
        self.btn_preset.setFixedHeight(self._BTN_H)
        self.btn_preset.clicked.connect(self._show_preset_menu)
        main_layout.addWidget(self.btn_preset, 0, Qt.AlignVCenter)

        # Fit button
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.setObjectName("ZoomBtn")
        self.btn_fit.setFixedHeight(self._BTN_H)
        self.btn_fit.clicked.connect(self.fitClicked.emit)
        main_layout.addWidget(self.btn_fit, 0, Qt.AlignVCenter)

        self.setStyleSheet("""
            QFrame#ZoomControls {
                background-color: rgba(30, 30, 30, 0.9);
                border-radius: 5px;
                border: 1px solid #444;
            }
            QFrame#ZoomControls QSlider {
                background: transparent;
            }
            QFrame#ZoomControls QPushButton#ZoomBtn,
            QFrame#ZoomControls QPushButton#ZoomPresetBtn {
                background-color: #383838;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                color: #ccc;
                font-size: 14px;
                font-weight: 600;
                padding: 0px 5px;
                margin: 0px;
                min-height: 0px;
                max-height: 30px;
            }
            QFrame#ZoomControls QPushButton#ZoomBtn:hover,
            QFrame#ZoomControls QPushButton#ZoomPresetBtn:hover {
                background-color: rgba(99, 102, 241, 0.2);
                border: 1px solid #6366f1;
                color: #fff;
            }
            QFrame#ZoomControls QPushButton#ZoomBtn:pressed,
            QFrame#ZoomControls QPushButton#ZoomPresetBtn:pressed {
                background-color: rgba(99, 102, 241, 0.35);
            }
        """)

    def _on_slider_changed(self, val: int) -> None:
        self.btn_preset.setText(f"{val}%")
        self.zoomChanged.emit(val / 100.0)

    def _show_preset_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                border: 1px solid #444;
                color: #e5e5e5;
                font-size: 12px;
            }
            QMenu::item { padding: 4px 20px; }
            QMenu::item:selected {
                background-color: #6366f1;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #444;
                margin: 2px 8px;
            }
        """)

        for level in [25, 50, 100, 200, 400]:
            action = menu.addAction(f"{level}%")
            action.triggered.connect(
                lambda checked=False, lv=level: self.zoomPresetSelected.emit(lv / 100.0)
            )

        menu.exec_(self.btn_preset.mapToGlobal(self.btn_preset.rect().bottomLeft()))

    def update_zoom(self, scale: float) -> None:
        val = max(1, min(400, int(round(scale * 100))))
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)
        self.btn_preset.setText(f"{val}%")

    # Backwards-compat shim for older tests referencing self.spin
    @property
    def spin(self) -> QtWidgets.QPushButton:
        return self.btn_preset
