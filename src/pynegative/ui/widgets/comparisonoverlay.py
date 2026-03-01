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

        # Get the base viewport from the background item
        scene_rect = self._view_ref._bg_item.sceneBoundingRect()
        view_points = self._view_ref.mapFromScene(scene_rect)
        base_rect = view_points.boundingRect()

        # If we have comparison pixmaps, calculate a viewport that fits both
        if self._unedited_pixmap and self._edited_pixmap:
            if not self._unedited_pixmap.isNull() and not self._edited_pixmap.isNull():
                # Get aspect ratios of comparison images
                left_ratio = self._unedited_pixmap.width() / max(
                    1, self._unedited_pixmap.height()
                )
                right_ratio = self._edited_pixmap.width() / max(
                    1, self._edited_pixmap.height()
                )

                # Use the wider aspect ratio
                max_ratio = max(left_ratio, right_ratio)

                # Calculate new viewport dimensions that fit within the base rect
                # while maintaining the wider aspect ratio
                base_w = base_rect.width()
                base_h = base_rect.height()
                base_ratio = base_w / max(1, base_h)

                if max_ratio > base_ratio:
                    # Need wider viewport - expand width, reduce height
                    new_h = base_w / max_ratio
                    new_y = base_rect.y() + (base_h - new_h) / 2
                    return QRectF(base_rect.x(), new_y, base_w, new_h)
                elif max_ratio < base_ratio:
                    # Need taller viewport - reduce width, expand height
                    new_w = base_h * max_ratio
                    new_x = base_rect.x() + (base_w - new_w) / 2
                    return QRectF(new_x, base_rect.y(), new_w, base_h)

        return base_rect

    def _calculate_scaled_rect(self, pixmap, target_rect):
        """Calculate scaled rect that preserves aspect ratio within target area.

        Returns (dest_rect, source_rect) where:
        - dest_rect: where to draw on the widget (centered in target)
        - source_rect: which portion of the pixmap to draw (full pixmap)
        """
        if pixmap is None or pixmap.isNull():
            return QRectF(), QRectF()

        pix_w = pixmap.width()
        pix_h = pixmap.height()
        target_w = target_rect.width()
        target_h = target_rect.height()

        if pix_w <= 0 or pix_h <= 0 or target_w <= 0 or target_h <= 0:
            return QRectF(), QRectF()

        # Calculate scale factor to fit pixmap within target while preserving aspect ratio
        scale_w = target_w / pix_w
        scale_h = target_h / pix_h
        scale = min(scale_w, scale_h)

        # Calculate scaled dimensions
        scaled_w = pix_w * scale
        scaled_h = pix_h * scale

        # Center the scaled image within the target rect
        dest_x = target_rect.x() + (target_w - scaled_w) / 2
        dest_y = target_rect.y() + (target_h - scaled_h) / 2

        dest_rect = QRectF(dest_x, dest_y, scaled_w, scaled_h)
        source_rect = QRectF(0, 0, pix_w, pix_h)

        return dest_rect, source_rect

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

        target_x = image_rect.x()
        target_y = image_rect.y()
        target_w = image_rect.width()
        target_h = image_rect.height()

        if target_w <= 0 or target_h <= 0:
            return

        # Define the full viewport area for scaling reference
        viewport_rect = QRectF(target_x, target_y, target_w, target_h)

        # Fill the viewport background to hide the underlying image
        painter.fillRect(viewport_rect, QColor(0, 0, 0))

        # Left side: Draw unedited pixmap with aspect ratio preserved
        if split_x > target_x:
            # Calculate scaled rect that preserves aspect ratio
            dest_rect, source_rect = self._calculate_scaled_rect(
                self._unedited_pixmap, viewport_rect
            )

            if dest_rect.isEmpty() or source_rect.isEmpty():
                return

            dest_w = dest_rect.width()

            # Clip to left side of split
            if dest_rect.right() > split_x and dest_w > 0:
                # Calculate what portion of the source should be visible
                clip_w = split_x - dest_rect.left()
                if clip_w > 0:
                    source_clip_w = (clip_w / dest_w) * source_rect.width()
                    if source_clip_w > 0:
                        clipped_source = QRectF(
                            source_rect.x(),
                            source_rect.y(),
                            source_clip_w,
                            source_rect.height(),
                        )
                        clipped_dest = QRectF(
                            dest_rect.left(),
                            dest_rect.top(),
                            clip_w,
                            dest_rect.height(),
                        )
                        painter.drawPixmap(
                            clipped_dest, self._unedited_pixmap, clipped_source
                        )
            elif dest_rect.right() <= split_x:
                # Entire image fits on left side
                painter.drawPixmap(dest_rect, self._unedited_pixmap, source_rect)

        # Right side: Draw edited pixmap with aspect ratio preserved
        right_start = max(split_x, target_x)
        right_end = target_x + target_w
        if right_end > right_start:
            # Calculate scaled rect that preserves aspect ratio
            dest_rect, source_rect = self._calculate_scaled_rect(
                self._edited_pixmap, viewport_rect
            )

            if dest_rect.isEmpty() or source_rect.isEmpty():
                return

            dest_w = dest_rect.width()

            # Clip to right side of split
            if dest_rect.left() < right_start and dest_w > 0:
                # Calculate what portion of the source should be visible
                clip_x = right_start - dest_rect.left()
                clip_w = min(dest_rect.right(), right_end) - right_start
                if clip_w > 0 and clip_x >= 0:
                    source_clip_x = (clip_x / dest_w) * source_rect.width()
                    source_clip_w = (clip_w / dest_w) * source_rect.width()
                    if source_clip_w > 0:
                        clipped_source = QRectF(
                            source_rect.x() + source_clip_x,
                            source_rect.y(),
                            source_clip_w,
                            source_rect.height(),
                        )
                        clipped_dest = QRectF(
                            right_start, dest_rect.top(), clip_w, dest_rect.height()
                        )
                        painter.drawPixmap(
                            clipped_dest, self._edited_pixmap, clipped_source
                        )
            elif dest_rect.left() >= right_start:
                # Entire image is on the right side
                visible_w = min(dest_rect.right(), right_end) - dest_rect.left()
                if visible_w > 0 and dest_w > 0:
                    source_clip_w = (visible_w / dest_w) * source_rect.width()
                    if source_clip_w > 0:
                        clipped_source = QRectF(
                            source_rect.x(),
                            source_rect.y(),
                            source_clip_w,
                            source_rect.height(),
                        )
                        clipped_dest = QRectF(
                            dest_rect.left(),
                            dest_rect.top(),
                            visible_w,
                            dest_rect.height(),
                        )
                        painter.drawPixmap(
                            clipped_dest, self._edited_pixmap, clipped_source
                        )

        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        painter.drawLine(split_x, 0, split_x, h)
