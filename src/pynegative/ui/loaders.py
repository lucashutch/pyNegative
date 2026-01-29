from pathlib import Path
from PIL import Image, ImageQt
from PySide6 import QtGui, QtCore
from .. import core as pynegative

# ----------------- Async Thumbnail Loader -----------------
class ThumbnailLoaderSignals(QtCore.QObject):
    finished = QtCore.Signal(str, object)  # path, QPixmap

class ThumbnailLoader(QtCore.QRunnable):
    def __init__(self, path, size=200):
        super().__init__()
        self.path = Path(path)
        self.size = size
        self.signals = ThumbnailLoaderSignals()

    def run(self):
        try:
            # use the optimized extract_thumbnail from core
            pil_img = pynegative.extract_thumbnail(self.path)
            if pil_img:
                # Resize for thumbnail grid
                pil_img.thumbnail((self.size, self.size))
                q_image = ImageQt.ImageQt(pil_img)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.signals.finished.emit(str(self.path), pixmap)
            else:
                self.signals.finished.emit(str(self.path), None)
        except Exception:
            self.signals.finished.emit(str(self.path), None)


# ----------------- Gallery Widget -----------------
class RawLoaderSignals(QtCore.QObject):
    finished = QtCore.Signal(str, object, object) # path, numpy array, settings_dict

class RawLoader(QtCore.QRunnable):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.signals = RawLoaderSignals()

    def run(self):
        try:
            # 1. Load Proxy (Half-Res)
            img = pynegative.open_raw(self.path, half_size=True)

            # 2. Check for Sidecar Settings
            settings = pynegative.load_sidecar(self.path)

            # 3. Fallback to Auto-Exposure
            if not settings:
                settings = pynegative.calculate_auto_exposure(img)

            self.signals.finished.emit(str(self.path), img, settings)
        except Exception as e:
            print(f"Error loading RAW {self.path}: {e}")
            self.signals.finished.emit(str(self.path), None, None)
