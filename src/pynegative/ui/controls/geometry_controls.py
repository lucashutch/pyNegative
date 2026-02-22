from PySide6 import QtCore, QtGui, QtWidgets

from ..widgets import CollapsibleSection
from .base import BaseControlWidget


class GeometryControls(BaseControlWidget):
    cropToggled = QtCore.Signal(bool)
    aspectRatioChanged = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.val_flip_h = False
        self.val_flip_v = False
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.geometry_section = CollapsibleSection("GEOMETRY")
        self.geometry_section.resetClicked.connect(self.reset_section)
        layout.addWidget(self.geometry_section)

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

        # Flip Buttons
        self.btn_flip_h = QtWidgets.QPushButton()
        self.btn_flip_v = QtWidgets.QPushButton()

        for btn, name, is_h in [
            (self.btn_flip_h, "Horizontal", True),
            (self.btn_flip_v, "Vertical", False),
        ]:
            btn.setCheckable(True)
            btn.setFixedSize(26, 18)
            btn.setToolTip(f"Flip {name}")
            pixmap = QtGui.QPixmap(32, 32)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor("#ccc"), 2)
            painter.setPen(pen)

            if is_h:
                painter.drawLine(16, 6, 16, 26)
                tri_left = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(14, 10),
                        QtCore.QPointF(4, 16),
                        QtCore.QPointF(14, 22),
                    ]
                )
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
                painter.drawLine(6, 16, 26, 16)
                tri_top = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(10, 14),
                        QtCore.QPointF(16, 4),
                        QtCore.QPointF(22, 14),
                    ]
                )
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

        self.btn_flip_h.toggled.connect(self._on_flip_h_toggled)
        self.btn_flip_v.toggled.connect(self._on_flip_v_toggled)
        self.crop_btn.toggled.connect(self._on_crop_toggled)

        crop_layout.addWidget(self.crop_btn)
        crop_layout.addStretch()
        crop_layout.addWidget(self.btn_flip_h)
        crop_layout.addWidget(self.btn_flip_v)
        crop_layout.addWidget(self.aspect_ratio_combo)

        self.aspect_ratio_combo.hide()
        self.btn_flip_h.hide()
        self.btn_flip_v.hide()

        self.geometry_section.add_widget(crop_widget)

        self.rotation_frame = self._add_slider(
            "Rotation",
            -45.0,
            45.0,
            0.0,
            "rotation",
            0.1,
            section=self.geometry_section,
            unit="deg",
        )
        self.rotation_frame.hide()

    def _on_flip_h_toggled(self, checked):
        self.val_flip_h = checked
        self.settingChanged.emit("flip_h", checked)

    def _on_flip_v_toggled(self, checked):
        self.val_flip_v = checked
        self.settingChanged.emit("flip_v", checked)

    def _on_crop_toggled(self, checked):
        if checked:
            self.crop_btn.setText("Done")
            self.aspect_ratio_combo.show()
            self.btn_flip_h.show()
            self.btn_flip_v.show()
            self.rotation_frame.show()
        else:
            self.crop_btn.setText("Crop Tool")
            self.aspect_ratio_combo.hide()
            self.btn_flip_h.hide()
            self.btn_flip_v.hide()
            self.rotation_frame.hide()
        self.cropToggled.emit(checked)

    def _on_aspect_ratio_changed(self, index):
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

    def reset_section(self):
        self.set_slider_value("rotation", 0.0)
        self.settingChanged.emit("rotation", 0.0)
        self.btn_flip_h.setChecked(False)
        self.btn_flip_v.setChecked(False)
        self.aspect_ratio_combo.setCurrentIndex(0)  # Reset to "Unlocked"
        self.settingChanged.emit("crop", None)
