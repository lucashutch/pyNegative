from PySide6 import QtWidgets

class HorizontalListWidget(QtWidgets.QListWidget):
    """A ListWidget that scrolls horizontally with the mouse wheel."""
    def wheelEvent(self, event):
        if event.angleDelta().y():
            # Scroll horizontally instead of vertically
            delta = event.angleDelta().y()
            # Most mice return 120 per notch. We apply a small multiplier for speed.
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)
