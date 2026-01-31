from PySide6 import QtWidgets, QtCore


class ComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.showPopup()

    def wheelEvent(self, event):
        # Disable wheel event to prevent scrolling through items
        pass
