from PySide6 import QtWidgets


class GalleryListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._hovered_item = None

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        item = self.itemAt(event.pos())
        if item != self._hovered_item:
            if self._hovered_item:
                # Trigger update on old item to remove hover effect
                self.update(self.visualItemRect(self._hovered_item))
            self._hovered_item = item
            if self._hovered_item:
                # Trigger update on new item to show hover effect
                self.update(self.visualItemRect(self._hovered_item))

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self._hovered_item:
            self.update(self.visualItemRect(self._hovered_item))
            self._hovered_item = None

    def get_hovered_item(self):
        return self._hovered_item
