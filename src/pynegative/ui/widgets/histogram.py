import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class HistogramWidget(QtWidgets.QWidget):
    """A widget that displays an image histogram (Luminance, RGB, or YUV)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.data = None
        self.mode = "Auto"  # Auto, Luminance, RGB, YUV
        self._is_grayscale = False

    def set_mode(self, mode):
        """Set display mode."""
        self.mode = mode
        self.update()

    def _check_grayscale(self):
        """Check if R, G, B channels are identical to optimize 'Auto' display."""
        if not self.data or "R" not in self.data:
            self._is_grayscale = False
            return

        r, g, b = self.data["R"], self.data["G"], self.data["B"]
        # Basic check: are they close enough?
        self._is_grayscale = np.allclose(r, g, atol=1e-5) and np.allclose(
            g, b, atol=1e-5
        )

    def set_data(self, data):
        """Set histogram data. Expects a dict with 'R', 'G', 'B' or 'Y', 'U', 'V' keys."""
        self.data = data
        self._check_grayscale()

        # Pre-calculate display paths to keep paintEvent extremely fast
        self._cached_paths = {}
        if self.data:
            self._prepare_paths()

        self.update()

    def _prepare_paths(self):
        """Pre-calculate the drawing paths."""
        mode = self.mode
        if mode == "Auto":
            mode = "Luminance" if self._is_grayscale else "RGB"

        if mode == "RGB" and "R" in self.data:
            self._cached_paths["R"] = self._create_path(self.data["R"])
            self._cached_paths["G"] = self._create_path(self.data["G"])
            self._cached_paths["B"] = self._create_path(self.data["B"])
        elif mode == "YUV" and "Y" in self.data:
            self._cached_paths["Y"] = self._create_path(self.data["Y"])
            self._cached_paths["U"] = self._create_path(self.data["U"])
            self._cached_paths["V"] = self._create_path(self.data["V"])
        elif "Y" in self.data:
            self._cached_paths["Y"] = self._create_path(self.data["Y"])
        elif "R" in self.data:
            lum = (self.data["R"] + self.data["G"] + self.data["B"]) / 3.0
            self._cached_paths["L"] = self._create_path(lum)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # Using Antialiasing is fine for paths, but we can toggle it if needed
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))

        if not self.data or not hasattr(self, "_cached_paths"):
            painter.setPen(QtGui.QColor("#808080"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No Data")
            return

        # Draw pre-calculated paths with specific colors
        for key, path in self._cached_paths.items():
            if key == "R":
                color = QtGui.QColor(255, 50, 50, 140)
            elif key == "G":
                color = QtGui.QColor(50, 255, 50, 140)
            elif key == "B":
                color = QtGui.QColor(50, 100, 255, 140)
            elif key == "Y":
                color = QtGui.QColor(255, 255, 255, 160)
            elif key == "U":
                color = QtGui.QColor(0, 255, 255, 120)
            elif key == "V":
                color = QtGui.QColor(255, 0, 255, 120)
            else:
                color = QtGui.QColor(200, 200, 200, 180)

            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(color.lighter(130), 1.5))
            painter.drawPath(path)

    def _create_path(self, hist):
        """Creates a QPainterPath from histogram data, normalized for the widget size."""
        if hist is None or len(hist) == 0:
            return QtGui.QPainterPath()

        rect = self.rect()
        w, h = rect.width(), rect.height()
        bins = len(hist)

        # Normalize height using sqrt scale
        hist_transformed = np.sqrt(hist)
        body_data = (
            hist_transformed[2:-2] if len(hist_transformed) > 10 else hist_transformed
        )
        max_val = np.max(body_data) if len(body_data) > 0 else np.max(hist_transformed)

        if max_val == 0:
            max_val = 1.0

        path = QtGui.QPainterPath()
        path.moveTo(0, h)

        # Optimization: Don't draw every single bin if the widget is small
        # This significantly reduces the complexity of the path
        step = max(1, bins // w)

        for i in range(0, bins, step):
            x = (i / (bins - 1)) * w
            val = min(1.2, hist_transformed[i] / max_val)
            y = h - (val * h * 0.8)
            path.lineTo(x, y)

        path.lineTo(w, h)
        path.closeSubpath()
        return path
