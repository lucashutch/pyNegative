import logging
from pathlib import Path
from PySide6 import QtWidgets, QtGui, QtCore

from .. import core as pynegative
from .gallery import GalleryWidget
from .editor import EditorWidget
from .export_tab import ExportWidget
from .widgets import StarRatingWidget

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_path=None):
        super().__init__()
        self.setWindowTitle("pyNegative")
        self.resize(1000, 700)

        self.thread_pool = QtCore.QThreadPool()
        self.thread_pool.setMaxThreadCount(
            max(4, QtCore.QThread.idealThreadCount() - 1)
        )

        # Load QSS Stylesheet
        self._load_stylesheet()

        # Central Widget & Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Views
        self.gallery = GalleryWidget(self.thread_pool)
        self.editor = EditorWidget(self.thread_pool)
        self.export_tab = ExportWidget(self.thread_pool)

        # Top Bar (Tabs)
        self._setup_top_bar(main_layout)

        # Stack
        self.stack = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stack)

        self.stack.addWidget(self.gallery)
        self.stack.addWidget(self.editor)
        self.stack.addWidget(self.export_tab)

        # Signals
        self.gallery.imageSelected.connect(self.open_editor)
        self.gallery.folderLoaded.connect(self.export_tab.load_folder)
        self.editor.ratingChanged.connect(self.gallery.update_rating_for_item)
        self.gallery.ratingChanged.connect(self.editor.update_rating_for_path)
        self.gallery.imageListChanged.connect(self._on_gallery_list_changed)
        self.gallery.viewModeChanged.connect(self._on_gallery_view_mode_changed)

        # Sync metadata button state when folder loads
        self.gallery.folderLoaded.connect(self._sync_metadata_btn_state)

        # Setup Menu (File operations only)
        self._create_menu()

        # Start in Gallery
        self.switch_to_gallery()

        # Load initial folder or file
        if initial_path:
            p = Path(initial_path)
            if p.is_dir():
                self.gallery.load_folder(str(p))
            elif p.is_file():
                # Load the parent folder in the gallery
                self.gallery.load_folder(str(p.parent))
                # Open the editor for this file
                self.open_editor(str(p))
        else:
            self.gallery._load_last_folder()

    def _load_stylesheet(self):
        """Load the QSS stylesheet from file."""
        # Find style.qss in the parent package (src/pynegative)
        # Actually it's better to keep it in src/pynegative/styles.qss
        # as it was, or move it.
        # Original was Path(__file__).parent / "styles.qss" in ui.py
        # Now MainWindow is in ui/main_window.py, so it's reaching up.
        style_path = Path(__file__).parent.parent / "styles.qss"
        try:
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at {style_path}")

    def _setup_top_bar(self, parent_layout):
        bar_frame = QtWidgets.QFrame()
        bar_frame.setObjectName("TopBar")
        bar_layout = QtWidgets.QHBoxLayout(bar_frame)
        bar_layout.setContentsMargins(10, 0, 10, 0)

        # Buttongroup for exclusivity logic is manual here for styling flexibility
        self.btn_gallery = QtWidgets.QPushButton("GALLERY")
        self.btn_gallery.setObjectName("TabButton")
        self.btn_gallery.setCheckable(True)
        self.btn_gallery.clicked.connect(self.switch_to_gallery)

        self.btn_edit = QtWidgets.QPushButton("EDIT")
        self.btn_edit.setObjectName("TabButton")
        self.btn_edit.setCheckable(True)
        self.btn_edit.clicked.connect(self.switch_to_edit)

        self.btn_export = QtWidgets.QPushButton("EXPORT")
        self.btn_export.setObjectName("TabButton")
        self.btn_export.setCheckable(True)
        self.btn_export.clicked.connect(self.switch_to_export)

        bar_layout.addWidget(self.btn_gallery)
        bar_layout.addWidget(self.btn_edit)
        bar_layout.addWidget(self.btn_export)
        bar_layout.addStretch()

        # Filtering
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.setContentsMargins(0, 5, 0, 5)
        filter_layout.setAlignment(QtCore.Qt.AlignVCenter)

        filter_label = QtWidgets.QLabel("Filter:")
        filter_layout.addWidget(filter_label)

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(["Match", "Greater", "Less"])
        self.filter_combo.setCurrentText("Greater")
        self.filter_combo.setMaximumWidth(120)
        self.filter_combo.currentIndexChanged.connect(
            self.gallery.apply_filter_from_main
        )
        self.filter_combo.currentIndexChanged.connect(
            self.export_tab.apply_filter_from_main
        )
        filter_layout.addWidget(self.filter_combo)

        self.filter_rating_widget = StarRatingWidget()
        self.filter_rating_widget.ratingChanged.connect(
            self.gallery.apply_filter_from_main
        )
        self.filter_rating_widget.ratingChanged.connect(
            self.export_tab.apply_filter_from_main
        )
        filter_layout.addWidget(self.filter_rating_widget)
        bar_layout.addLayout(filter_layout)

        # Metadata Panel Toggle
        self.metadata_btn = QtWidgets.QToolButton()
        self.metadata_btn.setText("ⓘ")
        self.metadata_btn.setObjectName("MetadataButton")
        self.metadata_btn.setToolTip("Metadata")
        self.metadata_btn.setCheckable(True)
        self.metadata_btn.setEnabled(False)
        self.metadata_btn.clicked.connect(self._on_metadata_toggle)
        bar_layout.addWidget(self.metadata_btn)

        self.btn_open_folder = QtWidgets.QPushButton("Open Folder")
        self.btn_open_folder.setObjectName("OpenFolderButton")
        self.btn_open_folder.clicked.connect(self.gallery.browse_folder)
        bar_layout.addWidget(self.btn_open_folder)

        parent_layout.addWidget(bar_frame)

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_folder_act = QtGui.QAction("Open Folder...", self)
        open_folder_act.triggered.connect(self.gallery.browse_folder)
        file_menu.addAction(open_folder_act)

        open_file_act = QtGui.QAction("Open File...", self)
        open_file_act.triggered.connect(self.open_single_file)
        file_menu.addAction(open_file_act)

    def _on_metadata_toggle(self):
        """Toggle metadata panel on the currently active view."""
        visible = self.metadata_btn.isChecked()
        current = self.stack.currentWidget()

        if current == self.editor:
            self.editor._metadata_panel_visible = visible
            self.editor.settings.setValue("metadata_panel_visible", visible)
            self.editor.metadata_panel.setVisible(visible)
            if visible and self.editor.raw_path:
                self.editor._load_metadata()
        elif current == self.gallery:
            if self.gallery._is_large_preview:
                pw = self.gallery.preview_widget
                pw._metadata_panel_visible = visible
                pw.settings.setValue("metadata_panel_visible", visible)
                pw.metadata_panel.setVisible(visible)
                if visible and pw.raw_path:
                    pw._load_metadata()
            else:
                # Gallery grid view
                self.gallery._gallery_metadata_visible = visible
                self.gallery.settings.setValue("gallery_metadata_visible", visible)
                self.gallery.gallery_metadata_panel.setVisible(visible)
                if visible:
                    # Trigger a selection update to populate metadata
                    self.gallery._on_gallery_selection_changed()

    def _sync_metadata_btn_state(self, *args):
        """Sync the metadata button state with the active view."""
        current = self.stack.currentWidget()
        if current == self.editor:
            has_image = self.editor.raw_path is not None
            self.metadata_btn.setEnabled(has_image)
            self.metadata_btn.setChecked(self.editor._metadata_panel_visible)
        elif current == self.gallery:
            if self.gallery._is_large_preview:
                pw = self.gallery.preview_widget
                has_image = pw.raw_path is not None
                self.metadata_btn.setEnabled(has_image)
                self.metadata_btn.setChecked(pw._metadata_panel_visible)
            else:
                # Grid view - always enable, state depends on gallery setting
                has_folder = self.gallery.current_folder is not None
                self.metadata_btn.setEnabled(has_folder)
                self.metadata_btn.setChecked(self.gallery._gallery_metadata_visible)
        else:
            self.metadata_btn.setEnabled(False)
            self.metadata_btn.setChecked(False)

    def switch_to_gallery(self):
        logger.info("Mode switch: GALLERY")
        self.stack.setCurrentWidget(self.gallery)
        self.btn_gallery.setChecked(True)
        self.btn_edit.setChecked(False)
        self.btn_export.setChecked(False)
        self._sync_metadata_btn_state()

    def switch_to_export(self):
        logger.info("Mode switch: EXPORT")
        self.stack.setCurrentWidget(self.export_tab)
        self.btn_gallery.setChecked(False)
        self.btn_edit.setChecked(False)
        self.btn_export.setChecked(True)
        self._sync_metadata_btn_state()

    def switch_to_edit(self):
        logger.info("Mode switch: EDIT")
        # If editor already has an image, just switch
        if self.editor.raw_path:
            self.stack.setCurrentWidget(self.editor)
            self.btn_gallery.setChecked(False)
            self.btn_edit.setChecked(True)
            self.btn_export.setChecked(False)
            self._sync_metadata_btn_state()
            return

        # Editor is empty, try to populate it from gallery
        path_to_open = None
        if self.gallery._is_large_preview and self.gallery.preview_widget.raw_path:
            path_to_open = str(self.gallery.preview_widget.raw_path)
        else:
            current_item = self.gallery.list_widget.currentItem()
            if current_item:
                path_to_open = current_item.data(QtCore.Qt.UserRole)

        if path_to_open:
            self.open_editor(path_to_open)
        else:
            # No selection, try first item
            image_list = self.gallery.get_current_image_list()
            if image_list:
                self.open_editor(image_list[0])
            else:
                # Gallery is empty, just switch to empty editor
                self.stack.setCurrentWidget(self.editor)
                self.btn_gallery.setChecked(False)
                self.btn_edit.setChecked(True)
                self.btn_export.setChecked(False)
                self._sync_metadata_btn_state()

    def open_editor(self, path):
        image_list = self.gallery.get_current_image_list()
        self.editor.open(path, image_list=image_list)
        self.stack.setCurrentWidget(self.editor)
        self.btn_gallery.setChecked(False)
        self.btn_edit.setChecked(True)
        self.btn_export.setChecked(False)
        self._sync_metadata_btn_state()

    def open_single_file(self):
        extensions = " ".join(["*" + e for e in pynegative.SUPPORTED_EXTS])
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window(),
            "Open Image",
            "",
            f"Images ({extensions})",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.open_editor(path)

    def _on_gallery_view_mode_changed(self, is_large_preview):
        # Sync metadata button state when gallery view mode changes
        self._sync_metadata_btn_state()

    def _on_gallery_list_changed(self, image_list):
        # Update Gallery Preview if active
        if self.gallery._is_large_preview and self.gallery.preview_widget.raw_path:
            current_path = str(self.gallery.preview_widget.raw_path)
            if current_path not in image_list:
                if image_list:
                    self.gallery.preview_widget.open(image_list[0], image_list)
                else:
                    # Gallery is empty, GalleryWidget.load_folder should have handled this,
                    # but we ensure consistency here.
                    self.gallery.toggle_view_mode()
            else:
                self.gallery.preview_widget.set_carousel_images(
                    image_list, current_path
                )

        # Update separate Editor tab if active
        if not self.editor.raw_path:
            return  # Editor isn't open, nothing to do

        current_path_str = str(self.editor.raw_path)
        if current_path_str in image_list:
            # Current image is still in the list, just update the carousel
            self.editor.set_carousel_images(image_list, self.editor.raw_path)
        else:
            # Current image has been filtered out
            if image_list:
                # Open the first image of the new list
                self.open_editor(image_list[0])
            else:
                # The filtered list is empty, clear the editor
                self.editor.clear()
                # Optionally, switch back to the gallery
                self.switch_to_gallery()
