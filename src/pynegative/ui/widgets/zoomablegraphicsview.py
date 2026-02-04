from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal, QRectF
from .crop_item import CropRectItem


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = Signal(float)
    doubleClicked = Signal()
    cropRectChanged = Signal(QRectF)

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
        # Fix ghosting during interactive item moves
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        # Explicitly set empty scene rect to prevent auto-calculation
        self._scene.setSceneRect(0, 0, 0, 0)

        # Background item (Low-res 1000-1500px, GPU scaled)
        self._bg_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._bg_item)
        self._bg_item.setZValue(0)

        # Foreground item (High-res ROI patch)
        self._fg_item = QtWidgets.QGraphicsPixmapItem()
        self._fg_item.setZValue(1)

        # Crop Item (Overlay)
        self._crop_item = CropRectItem()
        self._scene.addItem(self._crop_item)
        self._crop_item.setZValue(10)  # On top of everything
        self._crop_item.hide()
        # forward signal
        self._crop_item.cropChanged.connect(self.cropRectChanged.emit)
        self._crop_item.cropChanged.connect(self._on_crop_rect_changed)

        self._current_zoom = 1.0
        self._fit_in_view_scale = 1.0
        self._is_fitting = True
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # Signal for redraw on pan
        self.horizontalScrollBar().valueChanged.connect(self._sync_view)
        self.verticalScrollBar().valueChanged.connect(self._sync_view)

    def _sync_view(self):
        if not self._is_fitting:
            self.zoomChanged.emit(self._current_zoom)

    def _update_fit_in_view_scale(self):
        """Calculates and stores the scale factor for fitting the image to the view."""
        if self.sceneRect().isEmpty() or self.viewport().width() <= 0:
            return

        view_rect = self.viewport().rect()
        scene_rect = self.sceneRect()

        x_scale = view_rect.width() / scene_rect.width()
        y_scale = view_rect.height() / scene_rect.height()
        self._fit_in_view_scale = min(x_scale, y_scale)

    def resizeEvent(self, event):
        """Handle viewport resizing."""
        super().resizeEvent(event)
        self._update_fit_in_view_scale()
        if self._is_fitting:
            self.reset_zoom()

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
        self._update_fit_in_view_scale()

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
        # Only fit view if there's actual content (non-null pixmap)
        if bg_pixmap is None or bg_pixmap.isNull():
            self._current_zoom = 1.0
            self._is_fitting = True
            return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._current_zoom = self.transform().m11()
        self._is_fitting = True
        self.zoomChanged.emit(self._current_zoom)

    def set_zoom(self, scale, manual=True):
        self._is_fitting = False  # Any call to set_zoom breaks fitting
        if manual:
            # Clamp to the dynamic fit-in-view scale for manual user actions
            scale = max(self._fit_in_view_scale, scale)

        self._current_zoom = scale
        self.setTransform(QtGui.QTransform.fromScale(scale, scale))
        self.zoomChanged.emit(self._current_zoom)

    def wheelEvent(self, event):
        # Don't allow zooming if there's no content
        bg_pixmap = self._bg_item.pixmap()
        if bg_pixmap is None or bg_pixmap.isNull():
            return

        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9

        self._current_zoom = self.transform().m11()
        new_zoom = self._current_zoom * factor
        # Use the dynamic fit-in-view scale as the minimum
        new_zoom = max(self._fit_in_view_scale, min(new_zoom, 4.0))

        if abs(new_zoom - self._current_zoom) > 0.001:
            self.set_zoom(new_zoom, manual=True)

        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def set_crop_mode(self, enabled):
        if enabled:
            # If no rect set, default to scene rect
            if self._crop_item.get_rect().isEmpty():
                self._crop_item.set_rect(self._scene.sceneRect())
            self._crop_item.show()
        else:
            self._crop_item.hide()

    def set_crop_rect(self, rect):
        """Set visual crop rect (in scene coordinates)"""
        if rect:
            self._crop_item.set_rect(rect)

    def get_crop_rect(self):
        """Get visual crop rect (in scene coordinates)"""
        return self._crop_item.get_rect()

    def set_aspect_ratio(self, ratio):
        """Set crop aspect ratio lock (0.0 for free)"""
        self._crop_item.set_aspect_ratio(ratio)

    def set_crop_safe_bounds(self, rect):
        """Set the bounds within which the crop rectangle must stay."""
        self._crop_item.set_safe_bounds(rect)

    def fit_crop_in_view(self):
        """Scale the view to fit the current crop rectangle comfortably."""
        rect = self._crop_item.get_rect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.KeepAspectRatio)
            # Zoom out slightly for breathing room
            self.scale(0.9, 0.9)
            self._current_zoom = self.transform().m11()
            self._is_fitting = False
            self.zoomChanged.emit(self._current_zoom)

    def _on_crop_rect_changed(self, rect):
        """Keep the crop rectangle centered in the viewport."""
        if not self._crop_item.isVisible() or rect.isEmpty():
            return

        # Center the view on the new crop rectangle
        self.centerOn(rect.center())
