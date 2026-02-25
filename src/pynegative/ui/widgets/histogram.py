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
        """Set display mode and rebuild cached paths."""
        self.mode = mode
        self._cached_paths = {}
        if self.data:
            self._prepare_paths()
        self._waveform_pixmap = None
        if self.data and mode in ("Waveform (RGB)", "Waveform (Luma)"):
            self._prepare_waveform()
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

        self._waveform_pixmap = None
        if self.data and self.mode in ("Waveform (RGB)", "Waveform (Luma)"):
            self._prepare_waveform()

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

    @staticmethod
    def _normalise_waveform(wf):
        """Flip, normalise with aggressive power-curve scaling for bright display."""
        wf = wf[::-1].copy()
        max_val = wf.max()
        if max_val == 0:
            max_val = 1.0
        # Power curve 0.2 brings dim regions way up; much brighter than 0.4
        return np.power(wf / max_val, 0.2)

    def _prepare_waveform(self):
        """Build a QPixmap from per-channel or luma waveform arrays."""
        if self.mode == "Waveform (Luma)":
            wf_y = self.data.get("waveform_Y")
            if wf_y is None:
                return
            norm = self._normalise_waveform(wf_y)
            h, w = norm.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            # Scale luminance to 100–255 range for bright display
            val = (100 + norm * 155).astype(np.uint8)
            rgba[:, :, 0] = val
            rgba[:, :, 1] = val
            rgba[:, :, 2] = val
            # Alpha based on intensity: opaque only where waveform data exists
            rgba[:, :, 3] = (norm * 255).astype(np.uint8)
        else:
            # RGB waveform — overlay R, G, B with bright colors
            wf_r = self.data.get("waveform_R")
            wf_g = self.data.get("waveform_G")
            wf_b = self.data.get("waveform_B")
            if wf_r is None or wf_g is None or wf_b is None:
                return
            nr = self._normalise_waveform(wf_r)
            ng = self._normalise_waveform(wf_g)
            nb = self._normalise_waveform(wf_b)
            h, w = nr.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            # Scale channels to 80–255 range for bright display
            rgba[:, :, 0] = (80 + nr * 175).astype(np.uint8)
            rgba[:, :, 1] = (80 + ng * 175).astype(np.uint8)
            rgba[:, :, 2] = (80 + nb * 175).astype(np.uint8)
            # Alpha based on max channel intensity: opaque only where waveform data exists
            combined = np.maximum(nr, np.maximum(ng, nb))
            rgba[:, :, 3] = (combined * 255).astype(np.uint8)

        qimg = QtGui.QImage(rgba.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
        self._waveform_rgba = rgba  # prevent GC
        self._waveform_pixmap = QtGui.QPixmap.fromImage(qimg)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))

        if not self.data:
            painter.setPen(QtGui.QColor("#808080"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No Data")
            return

        # Waveform mode: draw pre-rendered pixmap scaled to widget
        if self.mode in ("Waveform (RGB)", "Waveform (Luma)"):
            pm = getattr(self, "_waveform_pixmap", None)
            if pm and not pm.isNull():
                painter.drawPixmap(self.rect(), pm)
            else:
                painter.setPen(QtGui.QColor("#808080"))
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No Data")
            return

        if not self._cached_paths:
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
