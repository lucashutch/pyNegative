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
        self._current_image_list = []
        self._current_height = 210
        self._pending_height = 210

        # Performance optimization: throttle item resizing during drag
        self._resize_timer = QtCore.QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(16)  # ~60fps target for layout updates
        self._resize_timer.timeout.connect(self._do_deferred_resize)

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
        self.carousel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.carousel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.carousel.setFixedHeight(self._current_height)

        # Calculate item sizes proportional to height, leaving room for padding and scrollbar
        grid_size = self._current_height - 35
        icon_width = grid_size - 20
        icon_height = grid_size - 50

        self.carousel.setIconSize(QtCore.QSize(icon_width, icon_height))
        self.carousel.setGridSize(QtCore.QSize(grid_size, grid_size))
        self.carousel.setResizeMode(QtWidgets.QListView.Adjust)
        self.carousel.setUniformItemSizes(True)
        self.carousel.setSpacing(8)

        # Set up carousel delegate for selection circles
        self.carousel_delegate = CarouselDelegate(self.carousel)
        self.carousel.setItemDelegate(self.carousel_delegate)

        # Set up carousel context menu
        self.carousel.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def _setup_connections(self):
        """Setup signal connections."""
        self.carousel.itemClicked.connect(self._on_item_clicked)
        self.carousel.itemSelectionChanged.connect(self._on_selection_changed)
        self.carousel.customContextMenuRequested.connect(self._show_context_menu)

    def get_widget(self):
        """Get the carousel widget for embedding in layout."""
        return self.carousel

    def load_folder(self, folder):
        """Load images from a folder into the carousel."""
        self.current_folder = Path(folder)
        self._current_image_list = []
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
            loader = ThumbnailLoader(path, size=400)
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

    def set_images(self, image_list, current_path):
        """Set specific images in the carousel."""
        # Convert all to strings for consistent comparison
        image_list_str = [str(p) for p in image_list]
        current_path_str = str(current_path) if current_path else None

        if image_list_str == self._current_image_list:
            # same list, just update selection
            if current_path_str:
                self.select_image(current_path_str)
            return

        self._current_image_list = image_list_str
        self.carousel.clear()

        for path_str in image_list_str:
            f = Path(path_str)
            item = QtWidgets.QListWidgetItem(f.name)
            item.setData(QtCore.Qt.UserRole, path_str)
            item.setIcon(
                self.carousel.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
            )
            self.carousel.addItem(item)
            if path_str == current_path_str:
                self.carousel.setCurrentItem(item)

            # Async load thumbnail
            loader = ThumbnailLoader(path_str, size=400)
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

        self._update_circle_visibility()  # Update circle visibility

    def select_image(self, path):
        """Select a specific image in the carousel."""
        path_str = str(path)
        for i in range(self.carousel.count()):
            item = self.carousel.item(i)
            if item.data(QtCore.Qt.UserRole) == path_str:
                self.carousel.setCurrentItem(item)
                # Ensure it's visible
                self.carousel.scrollToItem(
                    item, QtWidgets.QAbstractItemView.PositionAtCenter
                )
                break

    def select_previous(self):
        """Select the previous image in the carousel, wrapping to the end if at the start."""
        current_row = self.carousel.currentRow()
        total = self.carousel.count()
        if total == 0:
            return
        new_row = (current_row - 1) % total
        self.carousel.setCurrentRow(new_row)
        item = self.carousel.item(new_row)
        if item:
            self.carousel.clearSelection()
            item.setSelected(True)

    def select_next(self):
        """Select the next image in the carousel, wrapping to the start if at the end."""
        current_row = self.carousel.currentRow()
        total = self.carousel.count()
        if total == 0:
            return
        new_row = (current_row + 1) % total
        self.carousel.setCurrentRow(new_row)
        item = self.carousel.item(new_row)
        if item:
            self.carousel.clearSelection()
            item.setSelected(True)

    def clear(self):
        """Clear the carousel."""
        self._current_image_list = []
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

    def _on_thumbnail_loaded(self, path, pixmap, metadata):
        """Handle thumbnail loading completion."""
        for i in range(self.carousel.count()):
            item = self.carousel.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))
                break

    def set_carousel_height(self, height):
        """Request a height update for the carousel layout."""
        if abs(self._current_height - height) < 1:
            return

        self._pending_height = height

        # If the timer isn't running, start it.
        # We use a small interval to allow some real-time feedback but not every pixel.
        if not self._resize_timer.isActive():
            self._resize_timer.start()

    def _do_deferred_resize(self):
        """Perform the actual expensive layout update."""
        height = self._pending_height
        self._current_height = height

        # Calculate item sizes proportional to height, leaving room for padding and scrollbar
        grid_size = height - 35
        icon_width = grid_size - 20
        icon_height = grid_size - 50

        # These calls are expensive as they trigger full multi-item layout recalculations
        self.carousel.setGridSize(QtCore.QSize(grid_size, grid_size))
        self.carousel.setIconSize(QtCore.QSize(icon_width, icon_height))

        # Update delegate behavior
        self.carousel_delegate.set_item_size(grid_size)

        # Refresh layout
        self.carousel.doItemsLayout()

    def _on_item_clicked(self, item):
        """Handle item click."""
        path = item.data(QtCore.Qt.UserRole)
        self.imageSelected.emit(path)

    def _on_selection_changed(self):
        """Handle selection changes."""
        selected_paths = self.get_selected_paths()
        self.selectionChanged.emit(selected_paths)
        current_path = self.get_current_path()
        if current_path and (
            not selected_paths
            or (len(selected_paths) == 1 and str(current_path) in selected_paths)
        ):
            self.imageSelected.emit(str(current_path))
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
