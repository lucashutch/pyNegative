from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from .widgets import GalleryListWidget, CarouselDelegate
from .loaders import ThumbnailLoader
from .. import core as pynegative


class ExportGalleryManager(QtCore.QObject):
    """Manages the export gallery with filtering and selection."""

    # Signals
    selectionChanged = QtCore.Signal(list)  # selected image paths
    selectionCountChanged = QtCore.Signal(int)  # count of selected items
    imagesLoaded = QtCore.Signal(int)  # number of images loaded

    def __init__(self, thread_pool, parent=None):
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.current_folder = None
        self._selected_paths = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the gallery widget."""
        self.list_widget = GalleryListWidget()
        self.list_widget.setObjectName("ExportGrid")
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(180, 180))
        self.list_widget.setGridSize(QtCore.QSize(220, 240))
        self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Set up carousel delegate for selection circles
        self.carousel_delegate = CarouselDelegate(self.list_widget)
        self.list_widget.setItemDelegate(self.carousel_delegate)

    def _connect_signals(self):
        """Connect internal signals."""
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.selectionChanged.connect(self._update_circle_visibility)

    def get_widget(self):
        """Get the gallery widget for embedding in layout."""
        return self.list_widget

    def load_folder(self, folder, filter_mode="Match", filter_rating=0):
        """Load images from a folder with optional filtering."""
        self.current_folder = Path(folder)
        self.list_widget.clear()

        files = [
            f
            for f in self.current_folder.iterdir()
            if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS
        ]

        loaded_count = 0
        for path in files:
            sidecar_settings = pynegative.load_sidecar(str(path))
            rating = sidecar_settings.get("rating", 0) if sidecar_settings else 0

            # Apply filter
            if filter_rating > 0:
                if filter_mode == "Match" and rating != filter_rating:
                    continue
                if filter_mode == "Less" and rating >= filter_rating:
                    continue
                if filter_mode == "Greater" and rating <= filter_rating:
                    continue

            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            item.setData(QtCore.Qt.UserRole + 1, rating)
            item.setIcon(
                self.list_widget.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
            )
            self.list_widget.addItem(item)

            # Async load thumbnail
            loader = ThumbnailLoader(str(path))
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

            loaded_count += 1

        self._update_circle_visibility()
        self.imagesLoaded.emit(loaded_count)

    def set_images(self, image_list, filter_mode="Match", filter_rating=0):
        """Set specific images in the gallery with optional filtering."""
        self.list_widget.clear()

        loaded_count = 0
        for path_str in image_list:
            path = Path(path_str)
            sidecar_settings = pynegative.load_sidecar(str(path))
            rating = sidecar_settings.get("rating", 0) if sidecar_settings else 0

            # Apply filter
            if filter_rating > 0:
                if filter_mode == "Match" and rating != filter_rating:
                    continue
                if filter_mode == "Less" and rating >= filter_rating:
                    continue
                if filter_mode == "Greater" and rating <= filter_rating:
                    continue

            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            item.setData(QtCore.Qt.UserRole + 1, rating)
            item.setIcon(
                self.list_widget.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
            )
            self.list_widget.addItem(item)

            # Async load thumbnail
            loader = ThumbnailLoader(str(path))
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

            loaded_count += 1

        self._update_circle_visibility()
        self.imagesLoaded.emit(loaded_count)

    def get_selected_paths(self):
        """Get list of selected image paths."""
        return [
            item.data(QtCore.Qt.UserRole) for item in self.list_widget.selectedItems()
        ]

    def get_selected_count(self):
        """Get count of selected items."""
        return len(self.list_widget.selectedItems())

    def clear_selection(self):
        """Clear all selections."""
        self.list_widget.clearSelection()

    def select_all(self):
        """Select all items in the gallery."""
        self.list_widget.selectAll()

    def clear(self):
        """Clear the gallery."""
        self.list_widget.clear()
        self._selected_paths = []

    def _on_thumbnail_loaded(self, path, pixmap):
        """Handle thumbnail loading completion."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))
                break

    def _on_selection_changed(self):
        """Handle selection changes."""
        count = self.get_selected_count()
        self._selected_paths = self.get_selected_paths()
        self.selectionCountChanged.emit(count)
        self.selectionChanged.emit(self._selected_paths)

    def _update_circle_visibility(self):
        """Update circle visibility based on gallery state."""
        show_circles = self.list_widget.count() > 1
        self.carousel_delegate.set_show_selection_circles(show_circles)
