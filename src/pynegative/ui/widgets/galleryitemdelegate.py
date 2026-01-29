from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt


class GalleryItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.star_filled = self._create_star_pixmap(True)
        self.star_empty = self._create_star_pixmap(False)

    def _create_star_pixmap(self, filled):
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        if filled:
            painter.setPen(QtGui.QColor("#f0c419"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
        else:
            painter.setPen(QtGui.QColor("#909090"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "☆")
        painter.end()
        return pixmap

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        
        list_widget = self.parent()
        is_hovered = list_widget.get_hovered_item() is not None and list_widget.get_hovered_item().text() == index.data()

        rating = index.data(QtCore.Qt.UserRole + 1)
        if rating is None:
            rating = 0

        if rating > 0 or is_hovered:
            y = option.rect.y() + 5
            for i in range(5):
                star_icon = self.star_empty
                if i < rating:
                    star_icon = self.star_filled
                
                x = option.rect.x() + 5 + (i * (self.star_empty.width() + 2))
                painter.drawPixmap(x, y, star_icon)

    def editorEvent(self, event, model, option, index):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            list_widget = self.parent()
            is_hovered = list_widget.get_hovered_item() is not None and list_widget.get_hovered_item().text() == index.data()

            if is_hovered:
                # Check if click is on the stars
                y = option.rect.y() + 5
                x_start = option.rect.x() + 5
                star_width = self.star_empty.width()
                star_height = self.star_empty.height()

                if (event.pos().y() >= y and event.pos().y() <= y + star_height):
                    for i in range(5):
                        x = x_start + (i * (star_width + 2))
                        if event.pos().x() >= x and event.pos().x() <= x + star_width:
                            new_rating = i + 1
                            current_rating = index.data(QtCore.Qt.UserRole + 1)
                            if current_rating == new_rating:
                                new_rating = 0 # Allow clearing
                            model.setData(index, new_rating, QtCore.Qt.UserRole + 1)
                            return True

        return super().editorEvent(event, model, option, index)
