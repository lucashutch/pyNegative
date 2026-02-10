from PySide6 import QtWidgets
from ..widgets import CollapsibleSection
from .base import BaseControlWidget


class ToneControls(BaseControlWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tone_section = CollapsibleSection("TONE", expanded=True)
        self.tone_section.resetClicked.connect(self.reset_section)
        layout.addWidget(self.tone_section)

        self._add_slider(
            "Exposure", -4.0, 4.0, 0.0, "val_exposure", 0.01, self.tone_section
        )
        self._add_slider(
            "Contrast", 0.5, 2.0, 1.0, "val_contrast", 0.01, self.tone_section
        )
        self._add_slider(
            "Highlights", -1.0, 1.0, 0.0, "val_highlights", 0.01, self.tone_section
        )
        self._add_slider(
            "Shadows", -1.0, 1.0, 0.0, "val_shadows", 0.01, self.tone_section
        )
        self._add_slider(
            "Whites", 0.5, 1.5, 1.0, "val_whites", 0.01, self.tone_section, flipped=True
        )
        self._add_slider(
            "Blacks",
            -0.2,
            0.2,
            0.0,
            "val_blacks",
            0.001,
            self.tone_section,
            flipped=True,
        )

    def reset_section(self):
        params = [
            ("val_exposure", 0.0, "exposure"),
            ("val_contrast", 1.0, "contrast"),
            ("val_highlights", 0.0, "highlights"),
            ("val_shadows", 0.0, "shadows"),
            ("val_whites", 1.0, "whites"),
            ("val_blacks", 0.0, "blacks"),
        ]
        for var_name, default, setting_name in params:
            self.set_slider_value(var_name, default)
            self.settingChanged.emit(setting_name, default)
