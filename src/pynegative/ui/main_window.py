import sys
from pathlib import Path
from PySide6 import QtWidgets, QtGui, QtCore

from .. import core as pynegative
from .gallery import GalleryWidget
from .editor import EditorWidget

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyNegative")
        self.resize(1000, 700)

        self.thread_pool = QtCore.QThreadPool()

        # Load QSS Stylesheet
        self._load_stylesheet()

        # Central Widget & Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top Bar (Tabs)
        self._setup_top_bar(main_layout)

        # Stack
        self.stack = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stack)

        # Views
        self.gallery = GalleryWidget(self.thread_pool)
        self.editor = EditorWidget(self.thread_pool)

        self.stack.addWidget(self.gallery)
        self.stack.addWidget(self.editor)

        # Signals
        self.gallery.imageSelected.connect(self.open_editor)

        # Setup Menu (File operations only)
        self._create_menu()

        # Start in Gallery
        self.switch_to_gallery()

    def _load_stylesheet(self):
        """Load the QSS stylesheet from file."""
        # Find style.qss in the parent package (src/pynegative)
        # Actually it's better to keep it in src/pynegative/styles.qss
        # as it was, or move it.
        # Original was Path(__file__).parent / "styles.qss" in ui.py
        # Now MainWindow is in ui/main_window.py, so it's reaching up.
        style_path = Path(__file__).parent.parent / "styles.qss"
        try:
            with open(style_path, 'r') as f:
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

        bar_layout.addWidget(self.btn_gallery)
        bar_layout.addWidget(self.btn_edit)
        bar_layout.addStretch() # Push left

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

    def switch_to_gallery(self):
        self.stack.setCurrentWidget(self.gallery)
        self.btn_gallery.setChecked(True)
        self.btn_edit.setChecked(False)

    def switch_to_edit(self):
        self.stack.setCurrentWidget(self.editor)
        self.btn_gallery.setChecked(False)
        self.btn_edit.setChecked(True)

    def open_editor(self, path):
        # Determine folder from path
        path = Path(path)
        folder = path.parent

        # Load the image
        self.editor.load_image(path)

        # Update carousel if needed
        if self.editor.current_folder != folder:
            self.editor.load_carousel_folder(folder)

        self.switch_to_edit()

    def open_single_file(self):
        extensions = ' '.join(['*'+e for e in pynegative.SUPPORTED_EXTS])
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open RAW", "", f"RAW ({extensions})")
        if path:
            self.open_editor(path)
