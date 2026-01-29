from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt


class GalleryItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.star_pixmap = self._create_star_pixmap()

    def _create_star_pixmap(self):
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#f0c419"))
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
        painter.end()
        return pixmap

    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        rating = index.data(QtCore.Qt.UserRole + 1)
        if rating is not None and rating > 0:
            # Adjust position and size as needed
            y = option.rect.y() + 5
            for i in range(rating):
                x = option.rect.x() + 5 + (i * (self.star_pixmap.width() + 2))
                painter.drawPixmap(x, y, self.star_pixmap)
