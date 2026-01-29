from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, Signal

class HorizontalListWidget(QtWidgets.QListWidget):
    """A ListWidget that scrolls horizontally with the mouse wheel."""
    def wheelEvent(self, event):
        if event.angleDelta().y():
            # Scroll horizontally instead of vertically
            delta = event.angleDelta().y()
            # Most mice return 120 per notch. We apply a small multiplier for speed.
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)

class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with a header and a content area."""
    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Header
        self.header = QtWidgets.QPushButton(title)
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.clicked.connect(self.toggle)
        self.layout.addWidget(self.header)

        # Content Area
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout.setSpacing(2)
        self.layout.addWidget(self.content)

        if not expanded:
            self.content.hide()

    def toggle(self):
        if self.header.isChecked():
            self.content.show()
        else:
            self.content.hide()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

class ResetableSlider(QtWidgets.QSlider):
    """A QSlider that resets to a default value on double-click."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.default_slider_value = 0

    def mouseDoubleClickEvent(self, event):
        self.setValue(self.default_slider_value)
        # Trigger valueChanged signal explicitly if needed, but setValue does it
        super().mouseDoubleClickEvent(event)


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#1a1a1a")))

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        # Background item (Low-res 1000-1500px, GPU scaled)
        self._bg_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._bg_item)
        self._bg_item.setZValue(0)

        # Foreground item (High-res ROI patch)
        self._fg_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._fg_item)
        self._fg_item.setZValue(1)

        self._current_zoom = 1.0
        self._is_fitting = True
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # Signal for redraw on pan
        self.horizontalScrollBar().valueChanged.connect(self._sync_view)
        self.verticalScrollBar().valueChanged.connect(self._sync_view)

    def _sync_view(self):
        if not self._is_fitting:
            self.zoomChanged.emit(self._current_zoom)

    def set_pixmaps(self, bg_pix, full_w, full_h, roi_pix=None, roi_x=0, roi_y=0, roi_w=0, roi_h=0):
        """Unified update for both layers to ensure alignment."""
        # 1. Update Background
        self._bg_item.setPixmap(bg_pix)
        if bg_pix.width() > 0:
            s_w = full_w / bg_pix.width()
            s_h = full_h / bg_pix.height()
            self._bg_item.setTransform(QtGui.QTransform.fromScale(s_w, s_h))

        # 2. Update Scene Rect
        self._scene.setSceneRect(0, 0, full_w, full_h)

        # 3. Update ROI
        if roi_pix:
            self._fg_item.setPixmap(roi_pix)
            self._fg_item.setPos(roi_x, roi_y)
            # GPU Scale ROI if it was processed at lower resolution for performance
            if roi_w > 0 and roi_pix.width() > 0:
                rs_w = roi_w / roi_pix.width()
                rs_h = roi_h / roi_pix.height()
                self._fg_item.setTransform(QtGui.QTransform.fromScale(rs_w, rs_h))
            else:
                self._fg_item.setTransform(QtGui.QTransform())
            self._fg_item.show()
        else:
            self._fg_item.hide()

    def reset_zoom(self):
        if self._bg_item.pixmap().isNull() and self._scene.sceneRect().isEmpty():
             return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._current_zoom = self.transform().m11()
        self._is_fitting = True
        self.zoomChanged.emit(self._current_zoom)

    def set_zoom(self, scale, manual=True):
        if manual:
            self._is_fitting = False
        self._current_zoom = scale
        self.setTransform(QtGui.QTransform.fromScale(scale, scale))
        self.zoomChanged.emit(self._current_zoom)

    def wheelEvent(self, event):
        if self._scene.sceneRect().isEmpty():
            return

        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9

        self._current_zoom = self.transform().m11()
        new_zoom = self._current_zoom * factor
        new_zoom = max(0.5, min(new_zoom, 4.0))

        if new_zoom != self._current_zoom:
            self.set_zoom(new_zoom, manual=True)

        event.accept()

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
