import logging
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal, QRectF
from .crop_item import CropRectItem

logger = logging.getLogger(__name__)


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = Signal(float)
    doubleClicked = Signal()
    cropRectChanged = Signal(QRectF)
    rotationChanged = Signal(float)  # Forward rotation changes from crop item
    interactionFinished = Signal()

    ZOOM_LEVELS = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

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



        # Crop Item (Overlay)
        self._crop_item = CropRectItem()
        self._scene.addItem(self._crop_item)
        self._crop_item.setZValue(10)  # On top of everything
        self._crop_item.hide()
        # forward signals
        self._crop_item.cropChanged.connect(self.cropRectChanged.emit)
        self._crop_item.cropChanged.connect(self._on_crop_rect_changed)
        self._crop_item.rotationChanged.connect(self.rotationChanged.emit)
        self._crop_item.interactionFinished.connect(self.interactionFinished.emit)

        self._current_zoom = 1.0
        self._fit_in_view_scale = 1.0
        self._is_fitting = True
        self._rendered_rotation = 0.0  # The rotation baked into current pixmaps
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
        # Allow zooming out to 50% of the fit scale to ensure handles are visible
        self._fit_in_view_scale = min(x_scale, y_scale) * 0.5

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
        rotation=0.0,
    ):
        """Unified update for both layers to ensure alignment."""
        self._rendered_rotation = rotation

        if bg_pix is None:
            bg_pix = QtGui.QPixmap()

        # 1. Update Background
        self._bg_item.setPixmap(bg_pix)
        if not bg_pix.isNull() and bg_pix.width() > 0:
            s_w = full_w / bg_pix.width()
            s_h = full_h / bg_pix.height()
            self._bg_item.setTransform(QtGui.QTransform.fromScale(s_w, s_h))

        # 2. Update Scene Rect
        self._scene.setSceneRect(0, 0, full_w, full_h)
        self._update_fit_in_view_scale()

        # 3. Sync interactive rotation
        # If the user is rotating, they expect to see (current_angle - rendered_rotation)
        # applied via GPU rotation on top of the rendered pixmaps.
        current_angle = self._crop_item.get_rotation()
        rel_rotation = current_angle - self._rendered_rotation
        center = self._scene.sceneRect().center()

        self._bg_item.setTransformOriginPoint(self._bg_item.mapFromScene(center))
        self._bg_item.setRotation(rel_rotation)

    def reset_zoom(self):
        bg_pixmap = self._bg_item.pixmap()
        # Only fit view if there's actual content (non-null pixmap)
        if bg_pixmap is None or bg_pixmap.isNull():
            self._current_zoom = 1.0
            self._is_fitting = True
            return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        # If crop mode is active, zoom out a bit more to see handles
        if self._crop_item.isVisible():
            self.scale(0.8, 0.8)

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

    def zoom_in(self):
        """Zoom in to the next predefined level."""
        current = self.transform().m11()
        for level in self.ZOOM_LEVELS:
            if level > current + 0.001:
                self.set_zoom(level, manual=True)
                return
        # If already at or above max level, zoom by 10%
        self.set_zoom(min(current * 1.1, 4.0), manual=True)

    def zoom_out(self):
        """Zoom out to the next predefined level."""
        current = self.transform().m11()
        for level in reversed(self.ZOOM_LEVELS):
            if level < current - 0.001:
                # Still respect the dynamic fit-in-view minimum
                self.set_zoom(max(level, self._fit_in_view_scale), manual=True)
                return
        # If already at or below min level, zoom by 10%
        self.set_zoom(max(current * 0.9, self._fit_in_view_scale), manual=True)

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
            # Zoom out slightly for breathing room and to see rotation handles
            self.scale(0.8, 0.8)
            self._current_zoom = self.transform().m11()
            self._is_fitting = False
            self.zoomChanged.emit(self._current_zoom)

    def _on_crop_rect_changed(self, rect):
        """Keep the crop rectangle centered in the viewport."""
        if not self._crop_item.isVisible() or rect.isEmpty():
            return

        # Center the view on the new crop rectangle
        self.centerOn(rect.center())

    def set_rotation(self, angle: float) -> None:
        """Set rotation angle on crop item."""
        self._crop_item.set_rotation(angle)

        # Immediate visual feedback via GPU rotation
        rel_rotation = angle - self._rendered_rotation

        # Rotate around scene center to maintain alignment with GeometryResolver
        center = self._scene.sceneRect().center()

        self._bg_item.setTransformOriginPoint(self._bg_item.mapFromScene(center))
        self._bg_item.setRotation(rel_rotation)

    def get_rotation(self) -> float:
        """Get rotation angle from crop item."""
        return self._crop_item.get_rotation()
