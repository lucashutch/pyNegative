from pathlib import Path
from PySide6 import QtWidgets, QtGui, QtCore
from .. import core as pynegative
from .loaders import ThumbnailLoader
from .widgets import GalleryItemDelegate, GalleryListWidget, ComboBox
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
        self._sort_by = self.settings.value("sort_by", "Filename")
        self._sort_ascending = self.settings.value("sort_ascending", True, type=bool)
        self._grid_size = int(self.settings.value("grid_size", 200))
        self._is_large_preview = False
        # Grid size timer for throttling
        self._grid_resize_timer = QtCore.QTimer()
        self._grid_resize_timer.setSingleShot(True)
        self._grid_resize_timer.setInterval(16)
        self._grid_resize_timer.timeout.connect(self._do_deferred_grid_resize)

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
        top_bar.setContentsMargins(10, 2, 10, 2)
        grid_layout.addLayout(top_bar)

        top_bar.addWidget(QtWidgets.QLabel("Sort by:"))

        self.sort_combo = ComboBox()
        self.sort_combo.setObjectName("GallerySortCombo")
        self.sort_combo.setStyleSheet("""
            QComboBox#GallerySortCombo {
                padding: 2px 8px;
                min-height: 24px;
            }
        """)
        self.sort_combo.addItems(["Filename", "Date Taken", "Rating", "Last Edited"])
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        top_bar.addWidget(self.sort_combo)

        self.sort_order_btn = QtWidgets.QToolButton()
        self.sort_order_btn.setText("↑")
        self.sort_order_btn.setToolTip("Sort Order: Ascending")
        self.sort_order_btn.setCheckable(True)
        self.sort_order_btn.clicked.connect(self._on_sort_order_changed)
        top_bar.addWidget(self.sort_order_btn)

        top_bar.addStretch()  # Push controls to left

        # Grid View
        self.list_widget = GalleryListWidget()
        self.list_widget.setObjectName("GalleryGrid")
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(self._grid_size - 20, self._grid_size - 50))
        self.list_widget.setGridSize(QtCore.QSize(self._grid_size, self._grid_size))
        self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_widget.setSpacing(4)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        delegate = GalleryItemDelegate(self.list_widget)
        delegate.set_item_size(self._grid_size)
        self.list_widget.setItemDelegate(delegate)
        self.list_widget.model().dataChanged.connect(self._on_rating_changed)

        # Set initial values from settings
        index = self.sort_combo.findText(self._sort_by)
        if index >= 0:
            self.sort_combo.setCurrentIndex(index)
        self.sort_order_btn.setChecked(not self._sort_ascending)
        self._update_sort_order_button()

        grid_layout.addWidget(self.list_widget)

        # Bottom Bar for Grid Size Slider
        bottom_bar = QtWidgets.QHBoxLayout()
        bottom_bar.setContentsMargins(10, 5, 10, 5)
        grid_layout.addLayout(bottom_bar)

        bottom_bar.addStretch()
        bottom_bar.addWidget(QtWidgets.QLabel("Grid Size:"))
        self.size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_slider.setRange(100, 400)
        self.size_slider.setValue(self._grid_size)
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_grid_size_changed)
        bottom_bar.addWidget(self.size_slider)

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
                self.height() - self.btn_toggle_view.height() - 40,
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
            self.window(),
            "Open Folder",
            start_dir,
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        new_folder_path = Path(folder)
        is_same_folder = self.current_folder == new_folder_path
        self.current_folder = new_folder_path
        self.list_widget.clear()

        # Save to settings
        self.settings.setValue("last_folder", str(self.current_folder))

        # Determine if we should be in large preview or grid
        # If it's a new folder, always reset to grid
        if not is_same_folder:
            self._is_large_preview = False

        self.btn_toggle_view.show()
        self.btn_toggle_view.raise_()

        files = [
            f
            for f in self.current_folder.iterdir()
            if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS
        ]

        # The filter widgets are now in MainWindow, so we need to get the values from there.
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

            # Fast metadata fallback (will be refined by async loader)
            mtime = path.stat().st_mtime
            item.setData(QtCore.Qt.UserRole + 2, pynegative.format_date(mtime))
            item.setData(QtCore.Qt.UserRole + 3, mtime)

            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.list_widget.addItem(item)

            # Start async load
            loader = ThumbnailLoader(str(path), size=400)
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

        # Sync UI Stack and Toggle Button
        if self._is_large_preview:
            image_list = self.get_current_image_list()
            if not image_list:
                # Filtered out everything, fallback to grid
                self._is_large_preview = False
                self.stack.setCurrentWidget(self.grid_container)
                self.btn_toggle_view.setText("⊞")
            else:
                self.stack.setCurrentWidget(self.preview_widget)
                self.btn_toggle_view.setText("❐")

                # Check if current preview image is still in list
                current_path = (
                    str(self.preview_widget.raw_path)
                    if self.preview_widget.raw_path
                    else None
                )
                if current_path not in image_list:
                    # Current image gone, open first available
                    self.preview_widget.open(image_list[0], image_list)
                else:
                    # Current image still here, just update carousel
                    self.preview_widget.set_carousel_images(image_list, current_path)
        else:
            self.stack.setCurrentWidget(self.grid_container)
            self.btn_toggle_view.setText("⊞")

        self._apply_sort()
        self.folderLoaded.emit(str(folder))

    def _apply_filter(self):
        if self.current_folder:
            self.load_folder(str(self.current_folder))

    def apply_filter_from_main(self):
        self._apply_filter()

    def _on_thumbnail_loaded(self, path, pixmap, metadata):
        # find the item with this path
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))

                # Update with real EXIF date if available
                if "date" in metadata:
                    item.setData(QtCore.Qt.UserRole + 2, metadata["date"])
                break

    def _on_item_double_clicked(self, item):
        path = item.data(QtCore.Qt.UserRole)
        # Safety check: if we think we are in large preview but stack says otherwise, fix it
        if self._is_large_preview and self.stack.currentWidget() != self.preview_widget:
            self._is_large_preview = False

        if self._is_large_preview:
            self.preview_widget.open(path, self.get_current_image_list())
        else:
            self.toggle_view_mode()

    def _on_rating_changed(self, top_left_index, bottom_right_index, roles=None):
        if roles and (QtCore.Qt.UserRole + 1) not in roles:
            return

        if top_left_index != bottom_right_index:
            return

        item = self.list_widget.itemFromIndex(top_left_index)
        if item:
            path_str = item.data(QtCore.Qt.UserRole)
            rating = item.data(QtCore.Qt.UserRole + 1)

            settings = pynegative.load_sidecar(path_str) or {}
            settings["rating"] = rating
            pynegative.save_sidecar(path_str, settings)

            # Update last edited timestamp in item data
            last_edited = pynegative.get_sidecar_mtime(path_str)
            item.setData(QtCore.Qt.UserRole + 3, last_edited)

            # Re-sort if sorting by last edited or rating
            if self._sort_by in ["Last Edited", "Rating"]:
                self._apply_sort()

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

                # Update last edited timestamp in item data
                last_edited = pynegative.get_sidecar_mtime(path)
                item.setData(QtCore.Qt.UserRole + 3, last_edited)

                self.list_widget.update(self.list_widget.visualItemRect(item))

                # Re-sort if sorting by last edited or rating
                if self._sort_by in ["Last Edited", "Rating"]:
                    self._apply_sort()
                break

    def _on_sort_changed(self, sort_by):
        """Handle sort criteria change."""
        self._sort_by = sort_by
        self.settings.setValue("sort_by", sort_by)
        self._apply_sort()

    def _on_sort_order_changed(self):
        """Handle sort order toggle."""
        self._sort_ascending = not self.sort_order_btn.isChecked()
        self.settings.setValue("sort_ascending", self._sort_ascending)
        self._update_sort_order_button()
        self._apply_sort()

    def _update_sort_order_button(self):
        """Update sort order button appearance."""
        if self._sort_ascending:
            self.sort_order_btn.setText("↑")
            self.sort_order_btn.setToolTip("Sort Order: Ascending")
        else:
            self.sort_order_btn.setText("↓")
            self.sort_order_btn.setToolTip("Sort Order: Descending")

    def _on_grid_size_changed(self, value):
        """Handle grid size change from slider."""
        self._grid_size = value
        self.settings.setValue("grid_size", value)

        # Throttled update
        if not self._grid_resize_timer.isActive():
            self._grid_resize_timer.start()

    def _do_deferred_grid_resize(self):
        """Actually perform the heavy grid resize."""
        value = self._grid_size

        # Update list widget
        self.list_widget.setGridSize(QtCore.QSize(value, value))
        self.list_widget.setIconSize(QtCore.QSize(value - 20, value - 50))

        # Notify delegate
        delegate = self.list_widget.itemDelegate()
        if hasattr(delegate, "set_item_size"):
            delegate.set_item_size(value)

        # Trigger layout update
        self.list_widget.doItemsLayout()

    def _apply_sort(self):
        """Sort the current gallery items based on selected criteria."""
        if self.list_widget.count() == 0:
            self.imageListChanged.emit([])
            return

        # Extract all items with their data
        items_data = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path = item.data(QtCore.Qt.UserRole)
            rating = item.data(QtCore.Qt.UserRole + 1)
            date_taken = item.data(QtCore.Qt.UserRole + 2)  # Cache
            last_edited = item.data(QtCore.Qt.UserRole + 3)  # Cache

            items_data.append(
                {
                    "item": item,
                    "path": path,
                    "rating": rating,
                    "date_taken": date_taken,
                    "last_edited": last_edited,
                }
            )

        # Sort based on criteria
        if self._sort_by == "Filename":
            items_data.sort(key=lambda x: Path(x["path"]).name.lower())
        elif self._sort_by == "Date Taken":
            # Fallback to empty string for sorting
            items_data.sort(key=lambda x: x["date_taken"] or "0000-00-00")
        elif self._sort_by == "Rating":
            # Secondary sort by filename when ratings equal
            items_data.sort(key=lambda x: (x["rating"], Path(x["path"]).name.lower()))
        elif self._sort_by == "Last Edited":
            items_data.sort(key=lambda x: x["last_edited"] or 0)

        # Apply sort order
        if not self._sort_ascending:
            items_data.reverse()

        # Rebuild list widget
        # We need to block signals to avoid multiple updates
        self.list_widget.blockSignals(True)

        # Remove all items from list widget without deleting them
        while self.list_widget.count() > 0:
            self.list_widget.takeItem(0)

        for data in items_data:
            self.list_widget.addItem(data["item"])
        self.list_widget.blockSignals(False)

        # Update image list for preview mode
        current_images = self.get_current_image_list()
        self.imageListChanged.emit(current_images)

        # Sync preview carousel if in preview mode
        if self._is_large_preview and self.preview_widget.raw_path:
            current_path = str(self.preview_widget.raw_path)
            self.preview_widget.set_carousel_images(current_images, current_path)
