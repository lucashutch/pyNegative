from PySide6 import QtCore, QtGui, QtWidgets


class ComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Force the combobox to have a dropdown arrow
        self.setEditable(False)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        # Disable wheel event to prevent scrolling through items
        pass

    def paintEvent(self, event):
        super().paintEvent(event)

        # Draw custom dropdown arrow
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Get the dropdown rectangle
        rect = self.rect()
        dropdown_rect = QtCore.QRect(
            rect.right() - 30, rect.top() + 1, 29, rect.height() - 2
        )

        # Draw dropdown background
        dropdown_color = QtGui.QColor(53, 53, 53)  # #353535
        if self.underMouse():
            dropdown_color = QtGui.QColor(64, 64, 64)  # #404040

        painter.fillRect(dropdown_rect, dropdown_color)

        # Draw left border
        border_color = QtGui.QColor(64, 64, 64)  # #404040
        if self.underMouse():
            border_color = QtGui.QColor(99, 102, 241)  # #6366f1

        painter.setPen(QtGui.QPen(border_color, 1))
        painter.drawLine(
            dropdown_rect.left(),
            dropdown_rect.top(),
            dropdown_rect.left(),
            dropdown_rect.bottom(),
        )

        # Draw arrow
        arrow_color = QtGui.QColor(163, 163, 163)  # #a3a3a3
        if self.underMouse():
            arrow_color = QtGui.QColor(229, 229, 229)  # #e5e5e5

        painter.setBrush(QtGui.QBrush(arrow_color))
        painter.setPen(QtCore.Qt.NoPen)

        # Draw triangle arrow pointing down
        arrow_rect = QtCore.QRect(
            dropdown_rect.center().x() - 6, dropdown_rect.center().y() - 3, 12, 6
        )

        polygon = QtGui.QPolygon(
            [
                QtCore.QPoint(arrow_rect.left(), arrow_rect.top()),
                QtCore.QPoint(arrow_rect.right(), arrow_rect.top()),
                QtCore.QPoint(arrow_rect.center().x(), arrow_rect.bottom()),
            ]
        )
        painter.drawPolygon(polygon)

        painter.end()
