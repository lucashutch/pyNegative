from PySide6 import QtWidgets, QtGui, QtCore


class StarRatingWidget(QtWidgets.QWidget):
    ratingChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self._rating = 0
        self.star_filled_pixmap = self._create_star_pixmap(True)
        self.star_empty_pixmap = self._create_star_pixmap(False)
        self.setFixedHeight(24)
        self.setMouseTracking(True)
        self._hover_rating = -1

    def _create_star_pixmap(self, filled):
        pixmap = QtGui.QPixmap(24, 24)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        font = self.font()
        font.setPointSize(20)
        painter.setFont(font)

        if filled:
            painter.setPen(QtGui.QColor("#f0c419"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
        else:
            painter.setPen(QtGui.QColor("#808080"))  # gray
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "☆")

        painter.end()
        return pixmap

    def set_rating(self, rating):
        if self._rating != rating:
            self._rating = rating
            self.update()
            self.ratingChanged.emit(self._rating)

    def rating(self):
        return self._rating

    def sizeHint(self):
        return QtCore.QSize(
            self.star_empty_pixmap.width() * 5 + 4 * 4, self.star_empty_pixmap.height()
        )

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        for i in range(5):
            x = i * (self.star_empty_pixmap.width() + 4)
            star_icon = self.star_empty_pixmap

            is_filled = i < self._rating
            if self._hover_rating != -1:
                is_filled = i < self._hover_rating

            if is_filled:
                star_icon = self.star_filled_pixmap

            painter.drawPixmap(x, 0, star_icon)

    def mouseMoveEvent(self, event):
        if not self.isEnabled():
            return

        star_full_width = self.star_empty_pixmap.width() + 4  # Star width + spacing

        # Calculate which star is being hovered over
        # Check if event.position().x() is within the bounds of the 5 stars
        if 0 <= event.position().x() < (5 * star_full_width):
            hovered_star_index = int(event.position().x() / star_full_width)
            self._hover_rating = hovered_star_index + 1
        else:
            self._hover_rating = -1  # Outside the star area
        self.update()

    def mousePressEvent(self, event):
        if not self.isEnabled():
            return

        if self._hover_rating != -1:
            if self._rating == self._hover_rating:
                self.set_rating(0)  # Allow clearing rating
            else:
                self.set_rating(self._hover_rating)
        else:
            self.set_rating(0)

    def leaveEvent(self, event):
        self._hover_rating = -1
        self.update()
