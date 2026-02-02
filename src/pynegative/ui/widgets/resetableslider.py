from PySide6 import QtWidgets, QtCore


class ResetableSlider(QtWidgets.QSlider):
    """A QSlider that resets to a default value on double-click and uses up/down keys."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.default_slider_value = 0
        self.setPageStep(10)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Up:
            self.setValue(self.value() + self.singleStep())
            event.accept()
        elif event.key() == QtCore.Qt.Key_Down:
            self.setValue(self.value() - self.singleStep())
            event.accept()
        elif event.key() == QtCore.Qt.Key_PageUp:
            self.setValue(self.value() + self.pageStep())
            event.accept()
        elif event.key() == QtCore.Qt.Key_PageDown:
            self.setValue(self.value() - self.pageStep())
            event.accept()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.setValue(self.default_slider_value)
        super().mouseDoubleClickEvent(event)
