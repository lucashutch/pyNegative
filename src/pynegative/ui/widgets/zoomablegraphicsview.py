from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = Signal(float)
    doubleClicked = Signal()

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

    def set_pixmaps(
        self,
        bg_pix,
        full_w,
        full_h,
        roi_pix=None,
        roi_x=0,
        roi_y=0,
        roi_w=0,
        roi_h=0,
    ):
        """Unified update for both layers to ensure alignment."""
        if bg_pix is None:
            bg_pix = QtGui.QPixmap()
        if roi_pix is None:
            roi_pix = QtGui.QPixmap()
        # 1. Update Background
        self._bg_item.setPixmap(bg_pix)
        if not bg_pix.isNull() and bg_pix.width() > 0:
            s_w = full_w / bg_pix.width()
            s_h = full_h / bg_pix.height()
            self._bg_item.setTransform(QtGui.QTransform.fromScale(s_w, s_h))

        # 2. Update Scene Rect
        self._scene.setSceneRect(0, 0, full_w, full_h)

        # 3. Update ROI
        if not roi_pix.isNull():
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
        bg_pixmap = self._bg_item.pixmap()
        if bg_pixmap is None or (
            bg_pixmap.isNull() and self._scene.sceneRect().isEmpty()
        ):
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
        new_zoom = max(0.1, min(new_zoom, 4.0))

        if new_zoom != self._current_zoom:
            self.set_zoom(new_zoom, manual=True)

        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)
