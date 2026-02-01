from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from .loaders import ThumbnailLoader
from .widgets import HorizontalListWidget, CarouselDelegate
from .. import core as pynegative


class CarouselManager(QtCore.QObject):
    # Signals
    imageSelected = QtCore.Signal(str)  # path
    selectionChanged = QtCore.Signal(list)  # selected_paths
    contextMenuRequested = QtCore.Signal(str, object)  # context_type, position

    def __init__(self, thread_pool, parent=None):
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.current_folder = None

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """Setup the carousel UI components."""
        # Carousel (Bottom)
        self.carousel = HorizontalListWidget()
        self.carousel.setObjectName("Carousel")
        self.carousel.setViewMode(QtWidgets.QListView.IconMode)
        self.carousel.setFlow(QtWidgets.QListView.LeftToRight)  # Horizontal
        self.carousel.setWrapping(False)
        self.carousel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.carousel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.carousel.setFixedHeight(145)
        self.carousel.setIconSize(QtCore.QSize(100, 100))
        self.carousel.setSpacing(5)

        # Set up carousel delegate for selection circles
        self.carousel_delegate = CarouselDelegate(self.carousel)
        self.carousel.setItemDelegate(self.carousel_delegate)

        # Set up carousel context menu
        self.carousel.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def _setup_connections(self):
        """Setup signal connections."""
        self.carousel.itemClicked.connect(self._on_item_clicked)
        self.carousel.selectionChanged.connect(self._on_selection_changed)
        self.carousel.customContextMenuRequested.connect(self._show_context_menu)

    def get_widget(self):
        """Get the carousel widget for embedding in layout."""
        return self.carousel

    def load_folder(self, folder):
        """Load images from a folder into the carousel."""
        self.current_folder = Path(folder)
        self.carousel.clear()
        self._update_circle_visibility()  # Update circle visibility

        files = sorted(
            [
                f
                for f in self.current_folder.iterdir()
                if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS
            ]
        )

        for path in files:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            item.setIcon(
                self.carousel.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
            )
            self.carousel.addItem(item)

            # Async load thumbnail
            loader = ThumbnailLoader(path, size=100)
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

    def set_images(self, image_list, current_path):
        """Set specific images in the carousel."""
        self.carousel.clear()

        for path_str in image_list:
            f = Path(path_str)
            item = QtWidgets.QListWidgetItem(f.name)
            item.setData(QtCore.Qt.UserRole, str(f))
            item.setIcon(
                self.carousel.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
            )
            self.carousel.addItem(item)
            if f == current_path:
                self.carousel.setCurrentItem(item)

            # Async load thumbnail
            loader = ThumbnailLoader(f, size=100)
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

        self._update_circle_visibility()  # Update circle visibility

    def select_image(self, path):
        """Select a specific image in the carousel."""
        for i in range(self.carousel.count()):
            item = self.carousel.item(i)
            if Path(item.data(QtCore.Qt.UserRole)) == path:
                self.carousel.setCurrentItem(item)
                break

    def clear(self):
        """Clear the carousel."""
        self.carousel.clear()

    def get_selected_paths(self):
        """Get list of selected image paths."""
        return self.carousel.get_selected_paths()

    def get_current_path(self):
        """Get the currently selected path."""
        current_item = self.carousel.currentItem()
        if current_item:
            return Path(current_item.data(QtCore.Qt.UserRole))
        return None

    def _on_thumbnail_loaded(self, path, pixmap):
        """Handle thumbnail loading completion."""
        for i in range(self.carousel.count()):
            item = self.carousel.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))
                break

    def _on_item_clicked(self, item):
        """Handle item click."""
        path = item.data(QtCore.Qt.UserRole)
        self.imageSelected.emit(path)

    def _on_selection_changed(self):
        """Handle selection changes."""
        selected_paths = self.get_selected_paths()
        self.selectionChanged.emit(selected_paths)
        self._update_circle_visibility()

    def _update_circle_visibility(self):
        """Update circle visibility based on carousel state."""
        show_circles = self.carousel.should_show_circles()
        self.carousel_delegate.set_show_selection_circles(show_circles)

    def _show_context_menu(self, pos):
        """Show context menu for carousel."""
        item = self.carousel.itemAt(pos)
        if not item:
            return

        # Get item under mouse
        item_path = item.data(QtCore.Qt.UserRole)
        self.contextMenuRequested.emit("carousel", (pos, item_path, self.carousel))
