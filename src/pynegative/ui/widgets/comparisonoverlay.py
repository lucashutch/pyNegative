from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ComparisonHandle(QWidget):
    """Draggable handle for the comparison split line."""

    dragged = Signal(float)  # Returns global X position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(12)
        self.setFixedHeight(80)
        self.setCursor(Qt.SizeHorCursor)
        self._dragging = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(138, 43, 226))
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.dragged.emit(event.globalPosition().x())
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()


class ComparisonOverlay(QWidget):
    """Drawing layer for comparison. Does not capture mouse events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._comparison_active = False
        self._split_position = 0.5
        self._unedited_pixmap = None
        self._edited_pixmap = None
        self._view_ref = None

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def setSplitPosition(self, position):
        self._split_position = max(0.0, min(1.0, position))
        self.update()

    def setComparisonActive(self, active):
        self._comparison_active = active
        self.update()

    def setUneditedPixmap(self, pixmap):
        self._unedited_pixmap = pixmap
        self.update()

    def setEditedPixmap(self, pixmap):
        self._edited_pixmap = pixmap
        self.update()

    def updateEditedPixmap(self, pixmap):
        self._edited_pixmap = pixmap
        self.update()

    def setView(self, view):
        self._view_ref = view
        if hasattr(view, "zoomChanged"):
            view.zoomChanged.connect(self.update)
        view.horizontalScrollBar().valueChanged.connect(self.update)
        view.verticalScrollBar().valueChanged.connect(self.update)

    def paintEvent(self, event):
        if self._comparison_active and self._unedited_pixmap and self._edited_pixmap:
            self._paint_active(event)

    def _get_image_viewport_rect(self):
        if not self._view_ref or not self._view_ref._bg_item:
            return None
        scene_rect = self._view_ref._bg_item.sceneBoundingRect()
        view_points = self._view_ref.mapFromScene(scene_rect)
        return view_points.boundingRect()

    def _paint_active(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = self.rect()
        w, h = rect.width(), rect.height()
        split_x = int(w * self._split_position)

        image_rect = self._get_image_viewport_rect()
        if image_rect is None or self._unedited_pixmap is None:
            return

        target_x, target_y = image_rect.x(), image_rect.y()
        target_w, target_h = image_rect.width(), image_rect.height()

        if split_x > target_x:
            visible_w = min(split_x - target_x, target_w)
            pix_w = self._unedited_pixmap.width()
            source_w = (visible_w / target_w) * pix_w
            target_left = QRectF(target_x, target_y, visible_w, target_h)
            source_left = QRectF(0, 0, source_w, self._unedited_pixmap.height())
            painter.drawPixmap(target_left, self._unedited_pixmap, source_left)

        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        painter.drawLine(split_x, 0, split_x, h)
