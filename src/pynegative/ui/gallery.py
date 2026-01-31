from pathlib import Path
from PySide6 import QtWidgets, QtGui, QtCore
from .. import core as pynegative
from .loaders import ThumbnailLoader
from .widgets import GalleryItemDelegate, GalleryListWidget
from .editor import EditorWidget


class GalleryWidget(QtWidgets.QWidget):
    imageSelected = QtCore.Signal(str)
    ratingChanged = QtCore.Signal(str, int)
    imageListChanged = QtCore.Signal(list)
    folderLoaded = QtCore.Signal(str)
    viewModeChanged = QtCore.Signal(bool)  # True for Large Preview, False for Grid

    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.settings = QtCore.QSettings("pyNegative", "Gallery")
        self._is_large_preview = False
        self._init_ui()

    def _init_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Stack to switch between empty state and grid view
        self.stack = QtWidgets.QStackedWidget()
        self.main_layout.addWidget(self.stack)

        # Empty State (shown when no folder is loaded)
        self.empty_state = self._create_empty_state()
        self.stack.addWidget(self.empty_state)

        # Grid View Container
        self.grid_container = QtWidgets.QWidget()
        grid_layout = QtWidgets.QVBoxLayout(self.grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(0)

        # Top Bar (only visible when folder is loaded)
        top_bar = QtWidgets.QHBoxLayout()
        grid_layout.addLayout(top_bar)

        # Grid View
        self.list_widget = GalleryListWidget()
        self.list_widget.setObjectName("GalleryGrid")
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(180, 180))
        self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.setItemDelegate(GalleryItemDelegate(self.list_widget))
        self.list_widget.model().dataChanged.connect(self._on_rating_changed)
        grid_layout.addWidget(self.list_widget)

        self.stack.addWidget(self.grid_container)

        # Large Preview View
        self.preview_widget = EditorWidget(self.thread_pool)
        self.preview_widget.set_preview_mode(True)
        self.preview_widget.imageDoubleClicked.connect(self.toggle_view_mode)
        self.preview_widget.ratingChanged.connect(self.ratingChanged.emit)
        self.stack.addWidget(self.preview_widget)

        # Floating Toggle Button
        self.btn_toggle_view = QtWidgets.QPushButton("⊞", self)  # Grid icon placeholder
        self.btn_toggle_view.setObjectName("ViewToggleButton")
        self.btn_toggle_view.setFixedSize(50, 50)
        self.btn_toggle_view.setToolTip("Toggle Grid/Preview")
        self.btn_toggle_view.clicked.connect(self.toggle_view_mode)
        self.btn_toggle_view.hide()  # Hide until folder is loaded

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position the button in the bottom right corner
        if hasattr(self, "btn_toggle_view"):
            self.btn_toggle_view.move(
                self.width() - self.btn_toggle_view.width() - 20,
                self.height() - self.btn_toggle_view.height() - 20,
            )
            self.btn_toggle_view.raise_()

    def toggle_view_mode(self):
        self._is_large_preview = not self._is_large_preview
        if self._is_large_preview:
            self.btn_toggle_view.setText("❐")  # Preview icon placeholder
            self.stack.setCurrentWidget(self.preview_widget)

            # Load current selection into preview
            current_item = self.list_widget.currentItem()
            if current_item:
                path = current_item.data(QtCore.Qt.UserRole)
                self.preview_widget.open(path, self.get_current_image_list())
            else:
                # If no selection, try first item
                image_list = self.get_current_image_list()
                if image_list:
                    self.preview_widget.open(image_list[0], image_list)
        else:
            self.btn_toggle_view.setText("⊞")
            self.stack.setCurrentWidget(self.grid_container)

            # Sync selection back to grid
            if self.preview_widget.raw_path:
                path_str = str(self.preview_widget.raw_path)
                for i in range(self.list_widget.count()):
                    item = self.list_widget.item(i)
                    if item.data(QtCore.Qt.UserRole) == path_str:
                        self.list_widget.setCurrentItem(item)
                        break

        self.viewModeChanged.emit(self._is_large_preview)

    def _create_empty_state(self):
        """Create centered empty state with Open Folder button."""
        empty_widget = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(empty_widget)
        empty_layout.setAlignment(QtCore.Qt.AlignCenter)

        # Icon or placeholder
        icon_label = QtWidgets.QLabel("📁")
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 64px; color: #666;")
        empty_layout.addWidget(icon_label)

        # Message
        message = QtWidgets.QLabel("No folder opened")
        message.setAlignment(QtCore.Qt.AlignCenter)
        message.setStyleSheet("font-size: 18px; color: #a3a3a3; margin-top: 16px;")
        empty_layout.addWidget(message)

        # Open Folder Button
        open_btn = QtWidgets.QPushButton("Open Folder")
        open_btn.setObjectName("SaveButton")  # Use primary button style
        open_btn.setMinimumWidth(200)
        open_btn.clicked.connect(self.browse_folder)
        empty_layout.addWidget(open_btn, alignment=QtCore.Qt.AlignCenter)
        empty_layout.addSpacing(20)

        return empty_widget

    def _load_last_folder(self):
        """Load and open the last used folder if available."""
        last_folder = self.settings.value("last_folder", None)
        if last_folder and Path(last_folder).exists():
            self.load_folder(last_folder)
        else:
            # Show empty state
            self.stack.setCurrentWidget(self.empty_state)
            self.btn_toggle_view.hide()

    def browse_folder(self):
        # Start from last folder if available
        start_dir = ""
        if self.current_folder and self.current_folder.exists():
            start_dir = str(self.current_folder)

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open Folder", start_dir
        )
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.current_folder = Path(folder)
        self.list_widget.clear()

        # Save to settings
        self.settings.setValue("last_folder", str(self.current_folder))

        # Switch to grid view
        self.stack.setCurrentWidget(self.grid_container)
        self.btn_toggle_view.show()
        self.btn_toggle_view.raise_()

        files = [
            f
            for f in self.current_folder.iterdir()
            if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS
        ]

        # The filter widgets are now in MainWindow, so we need to get the values from there.
        # This is a bit of a hack. A better way would be to pass the filter values
        # into load_folder, or use a shared model.
        main_window = self.window()
        filter_mode = main_window.filter_combo.currentText()
        filter_rating = main_window.filter_rating_widget.rating()

        for path in files:
            sidecar_settings = pynegative.load_sidecar(str(path))
            rating = sidecar_settings.get("rating", 0) if sidecar_settings else 0

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
            # Set placeholder icon
            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.list_widget.addItem(item)

            # Start async load
            loader = ThumbnailLoader(str(path))
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

        self.imageListChanged.emit(self.get_current_image_list())
        self.folderLoaded.emit(str(folder))

    def _apply_filter(self):
        if self.current_folder:
            self.load_folder(str(self.current_folder))

    def apply_filter_from_main(self):
        self._apply_filter()

    def _on_thumbnail_loaded(self, path, pixmap):
        # find the item with this path
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))
                break

    def _on_item_double_clicked(self, item):
        path = item.data(QtCore.Qt.UserRole)
        if self._is_large_preview:
            # This shouldn't really happen from grid, but for consistency:
            self.preview_widget.open(path, self.get_current_image_list())
        else:
            self.toggle_view_mode()
            # The toggle_view_mode will load the current selection,
            # and double click also selects the item.

    def _on_rating_changed(self, top_left_index, bottom_right_index):
        if top_left_index != bottom_right_index:
            return

        item = self.list_widget.itemFromIndex(top_left_index)
        if item:
            path_str = item.data(QtCore.Qt.UserRole)
            rating = item.data(QtCore.Qt.UserRole + 1)

            settings = pynegative.load_sidecar(path_str) or {}
            settings["rating"] = rating
            pynegative.save_sidecar(path_str, settings)

            self.ratingChanged.emit(path_str, rating)

    def get_current_image_list(self):
        paths = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            paths.append(item.data(QtCore.Qt.UserRole))
        return paths

    def update_rating_for_item(self, path, rating):
        # Update both grid and preview
        self.preview_widget.update_rating_for_path(path, rating)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                item.setData(QtCore.Qt.UserRole + 1, rating)
                self.list_widget.update(self.list_widget.visualItemRect(item))
                break
