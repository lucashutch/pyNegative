from pathlib import Path
from PIL import ImageQt
from PySide6 import QtGui, QtCore
from .. import core as pynegative


# ----------------- Async Thumbnail Loader -----------------
class ThumbnailLoaderSignals(QtCore.QObject):
    finished = QtCore.Signal(str, object, dict)  # path, QPixmap, metadata


# Global cache to share thumbnails and metadata
# Key: (path_str, mtime, size) -> (QPixmap, metadata)
_THUMBNAIL_CACHE = {}
_PENDING_LOADS = {}  # path -> list of signals


class ThumbnailLoader(QtCore.QRunnable):
    def __init__(self, path, size=400):
        super().__init__()
        self.path = Path(path)
        self.size = size
        self.signals = ThumbnailLoaderSignals()

    def run(self):
        try:
            path_str = str(self.path)
            if not self.path.exists():
                return

            mtime = self.path.stat().st_mtime
            cache_key = (path_str, mtime, self.size)

            # 1. Check Memory Cache first
            if cache_key in _THUMBNAIL_CACHE:
                pixmap, metadata = _THUMBNAIL_CACHE[cache_key]
                self.signals.finished.emit(path_str, pixmap, metadata)
                return

            # 2. Check Disk Cache
            pil_img, metadata = pynegative.load_cached_thumbnail(self.path, self.size)
            if pil_img:
                q_image = ImageQt.ImageQt(pil_img)
                pixmap = QtGui.QPixmap.fromImage(q_image)

                # Store in memory cache
                _THUMBNAIL_CACHE[cache_key] = (pixmap, metadata)

                self.signals.finished.emit(path_str, pixmap, metadata)
                return

            # 3. Generate from scratch
            pil_img = pynegative.extract_thumbnail(self.path)
            metadata = {}

            if pil_img:
                # Basic metadata
                metadata["width"] = pil_img.width
                metadata["height"] = pil_img.height

                # Try to get date if it's a standard image (already opened by PIL)
                if self.path.suffix.lower() in pynegative.STD_EXTS:
                    metadata["date"] = pynegative.get_exif_capture_date(self.path)

                # Resize for thumbnail grid
                pil_img.thumbnail((self.size, self.size))
                q_image = ImageQt.ImageQt(pil_img)
                pixmap = QtGui.QPixmap.fromImage(q_image)

                # Store in Memory cache
                _THUMBNAIL_CACHE[cache_key] = (pixmap, metadata)

                # Store on Disk Cache
                pynegative.save_cached_thumbnail(self.path, pil_img, metadata, self.size)

                self.signals.finished.emit(path_str, pixmap, metadata)
            else:
                self.signals.finished.emit(path_str, None, {})
        except Exception:
            self.signals.finished.emit(str(self.path), None, {})


# ----------------- Gallery Widget -----------------
class RawLoaderSignals(QtCore.QObject):
    finished = QtCore.Signal(str, object, object)  # path, numpy array, settings_dict


class RawLoader(QtCore.QRunnable):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.signals = RawLoaderSignals()

    def run(self):
        try:
            # 1. Load Full-Res image for editing
            img = pynegative.open_raw(self.path, half_size=False)

            # 2. Check for Sidecar Settings
            settings = pynegative.load_sidecar(self.path)

            # 3. Fallback to Auto-Exposure
            if not settings:
                settings = pynegative.calculate_auto_exposure(img)

            self.signals.finished.emit(str(self.path), img, settings)
        except Exception as e:
            print(f"Error loading RAW {self.path}: {e}")
            self.signals.finished.emit(str(self.path), None, None)
