from PySide6 import QtCore, QtGui

from .starrating import StarRatingWidget


class PreviewStarRatingWidget(StarRatingWidget):
    """A larger star rating widget for preview mode with 30px stars."""

    def _create_star_pixmap(self, filled):
        size = 30
        self.setFixedHeight(size)
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        font = self.font()
        try:
            font.setPointSize(24)
        except Exception:
            pass
        painter.setFont(font)

        if filled:
            painter.setPen(QtGui.QColor("#f0c419"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
        else:
            painter.setPen(QtGui.QColor("#808080"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "☆")

        painter.end()
        return pixmap
