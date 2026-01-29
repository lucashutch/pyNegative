from PySide6 import QtWidgets

class ResetableSlider(QtWidgets.QSlider):
    """A QSlider that resets to a default value on double-click."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.default_slider_value = 0

    def mouseDoubleClickEvent(self, event):
        self.setValue(self.default_slider_value)
        # Trigger valueChanged signal explicitly if needed, but setValue does it
        super().mouseDoubleClickEvent(event)
