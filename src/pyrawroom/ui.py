import sys
import os
import json
import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtWidgets, QtGui, QtCore

from . import core as pyrawroom

# ----------------- Async Thumbnail Loader -----------------
class ThumbnailLoaderSignals(QtCore.QObject):
    finished = QtCore.Signal(str, object)  # path, QPixmap

class ThumbnailLoader(QtCore.QRunnable):
    def __init__(self, path, size=200):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = ThumbnailLoaderSignals()

    def run(self):
        try:
            # use the optimized extract_thumbnail from core
            pil_img = pyrawroom.extract_thumbnail(self.path)
            if pil_img:
                # Resize for thumbnail grid
                pil_img.thumbnail((self.size, self.size))
                q_image = ImageQt.ImageQt(pil_img)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.signals.finished.emit(self.path, pixmap)
            else:
                self.signals.finished.emit(self.path, None)
        except Exception:
            self.signals.finished.emit(self.path, None)


# ----------------- Gallery Widget -----------------
class GalleryWidget(QtWidgets.QWidget):
    imageSelected = QtCore.Signal(str) # Path

    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Top Bar
        top_bar = QtWidgets.QHBoxLayout()
        self.btn_open_folder = QtWidgets.QPushButton("Open Folder")
        self.btn_open_folder.clicked.connect(self.browse_folder)
        top_bar.addWidget(self.btn_open_folder)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Grid View
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(180, 180))
        self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.current_folder = folder
        self.list_widget.clear()

        # Find raw files
        files = [f for f in os.listdir(folder) if any(f.endswith(ext) for ext in pyrawroom.SUPPORTED_EXTS)]

        for f in files:
            path = os.path.join(folder, f)
            item = QtWidgets.QListWidgetItem(os.path.basename(f))
            item.setData(QtCore.Qt.UserRole, path)
            # Set placeholder icon
            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.list_widget.addItem(item)

            # Start async load
            loader = ThumbnailLoader(path)
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

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
        self.imageSelected.emit(path)


# ----------------- Editor Widget -----------------
class EditorWidget(QtWidgets.QWidget):
    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.raw_path = None
        self.base_img_full = None
        self.base_img_preview = None
        self.current_qpixmap = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # --- Left Panel (Controls) ---
        self.panel = QtWidgets.QFrame()
        self.panel.setContentsMargins(10, 10, 10, 10)
        self.panel.setFixedWidth(350)
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        main_layout.addWidget(self.panel)

        # --- Canvas (Right Side) ---
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas_frame)

        self.canvas_label = QtWidgets.QLabel()
        self.canvas_label.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas_label.setStyleSheet("background-color: #2b2b2b;")

        canvas_layout = QtWidgets.QVBoxLayout(self.canvas_frame)
        canvas_layout.addWidget(self.canvas_label)

        self.canvas_label.installEventFilter(self)

        # Carousel (Bottom)
        self.carousel = QtWidgets.QListWidget()
        self.carousel.setViewMode(QtWidgets.QListView.IconMode)
        self.carousel.setFlow(QtWidgets.QListView.LeftToRight) # Horizontal
        self.carousel.setFixedHeight(120)
        self.carousel.setIconSize(QtCore.QSize(100, 100))
        self.carousel.setSpacing(5)
        self.carousel.itemClicked.connect(self._on_carousel_item_clicked)
        canvas_layout.addWidget(self.carousel)

        self._setup_controls()

    def eventFilter(self, watched, event):
        if watched == self.canvas_label and event.type() == QtCore.QEvent.Type.Resize:
            self.update_preview()
        return super().eventFilter(watched, event)

    def _setup_controls(self):
        self.lbl_info = QtWidgets.QLabel("No file loaded")
        self.lbl_info.setWordWrap(True)
        self.panel_layout.addWidget(self.lbl_info)
        self.panel_layout.addSpacing(10)

        # Tone Controls
        self.val_exposure = 0.0
        self.val_whites = 1.0
        self.val_blacks = 0.0
        self._add_slider("Exposure", -4.0, 4.0, self.val_exposure, "val_exposure", 0.01)
        self._add_slider("Contrast (Whites)", 0.5, 2.0, self.val_whites, "val_whites", 0.01)
        self._add_slider("Blacks", -0.2, 0.2, self.val_blacks, "val_blacks", 0.001)

        self._add_separator()
        self.val_highlights = 0.0
        self.val_shadows = 0.0
        self._add_slider("Highlights", -1.0, 1.0, self.val_highlights, "val_highlights", 0.01)
        self._add_slider("Shadows", -1.0, 1.0, self.val_shadows, "val_shadows", 0.01)

        # Sharpening
        self._add_separator()
        self.var_sharpen_enabled = False
        self.sharpen_checkbox = QtWidgets.QCheckBox("Enable Sharpening")
        self.sharpen_checkbox.toggled.connect(self._update_sharpen_state)
        self.panel_layout.addWidget(self.sharpen_checkbox)

        self.val_radius = 2.0
        self.val_percent = 150
        self._add_slider("Sharpen Radius", 0.5, 5.0, self.val_radius, "val_radius", 0.01)
        self._add_slider("Sharpen Amount", 0, 300, self.val_percent, "val_percent", 1)

        # Save Button
        self._add_separator(20)
        self.btn_save = QtWidgets.QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setEnabled(False)
        self.panel_layout.addWidget(self.btn_save)

        self.panel_layout.addStretch()

    def _add_separator(self, spacing=10):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.panel_layout.addWidget(line)
        if spacing: self.panel_layout.addSpacing(spacing)

    def _add_slider(self, label_text, min_val, max_val, default, var_name, step_size):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        layout.addWidget(QtWidgets.QLabel(label_text))

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        multiplier = 1000
        slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
        slider.setValue(int(default * multiplier))

        val_lbl = QtWidgets.QLabel(f"{default:.2f}")
        val_lbl.setAlignment(QtCore.Qt.AlignRight)

        def on_change(val):
            actual = val / multiplier
            val_lbl.setText(f"{actual:.2f}")
            setattr(self, var_name, actual)
            self.request_update()

        slider.valueChanged.connect(on_change)

        # Store refs
        setattr(self, f"{var_name}_slider", slider)

        layout.addWidget(slider)
        layout.addWidget(val_lbl)
        self.panel_layout.addWidget(frame)

    def _update_sharpen_state(self, checked):
        self.var_sharpen_enabled = checked
        self.request_update()

    def load_image(self, path):
        self.lbl_info.setText(f"Loading: {os.path.basename(path)}")
        QtWidgets.QApplication.processEvents()

        try:
            self.base_img_full = pyrawroom.open_raw(path)
            self.raw_path = path

            # Create preview (max 1000px)
            h, w, _ = self.base_img_full.shape
            scale = 1000 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            temp_pil = Image.fromarray((self.base_img_full * 255).astype(np.uint8))
            temp_pil = temp_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            self.base_img_preview = np.array(temp_pil).astype(np.float32) / 255.0

            self.btn_save.setEnabled(True)
            self.request_update()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def request_update(self):
        if self.base_img_preview is not None:
            self.update_preview()

    def update_preview(self):
        if self.base_img_preview is None: return

        # Process
        img, _ = pyrawroom.apply_tone_map(
            self.base_img_preview,
            exposure=self.val_exposure, blacks=self.val_blacks, whites=self.val_whites,
            shadows=self.val_shadows, highlights=self.val_highlights
        )
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        if self.var_sharpen_enabled:
            pil_img = pyrawroom.sharpen_image(pil_img, self.val_radius, self.val_percent)

        # Display
        q_img = ImageQt.ImageQt(pil_img)
        pixmap = QtGui.QPixmap.fromImage(q_img)

        if not self.canvas_label.size().isEmpty():
            pixmap = pixmap.scaled(
                self.canvas_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        self.canvas_label.setPixmap(pixmap)

    def save_file(self):
        if self.base_img_full is None: return

        input_dir = os.path.dirname(self.raw_path)
        default_name = os.path.splitext(os.path.basename(self.raw_path))[0] + ".jpg"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save", os.path.join(input_dir, default_name), "JPEG (*.jpg);;HEIF (*.heic)")

        if path:
            try:
                # Process full resolution
                img, _ = pyrawroom.apply_tone_map(self.base_img_full, exposure=self.val_exposure, blacks=self.val_blacks, whites=self.val_whites, shadows=self.val_shadows, highlights=self.val_highlights)
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                if self.var_sharpen_enabled:
                    pil_img = pyrawroom.sharpen_image(pil_img, self.val_radius, self.val_percent)

                pyrawroom.save_image(pil_img, path)
                QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def load_carousel_folder(self, folder):
        self.current_folder = folder
        self.carousel.clear()

        files = [f for f in os.listdir(folder) if any(f.endswith(ext) for ext in pyrawroom.SUPPORTED_EXTS)]
        files.sort()

        for f in files:
            path = os.path.join(folder, f)
            item = QtWidgets.QListWidgetItem(os.path.basename(f))
            item.setData(QtCore.Qt.UserRole, path)
            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.carousel.addItem(item)

            # Async load thumbnail
            loader = ThumbnailLoader(path, size=100)
            loader.signals.finished.connect(self._on_carousel_thumbnail_loaded)
            self.thread_pool.start(loader)

    def _on_carousel_thumbnail_loaded(self, path, pixmap):
        for i in range(self.carousel.count()):
            item = self.carousel.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))
                break

    def _on_carousel_item_clicked(self, item):
        path = item.data(QtCore.Qt.UserRole)
        # Avoid reloading if same image
        if path != self.raw_path:
            self.load_image(path)


# ----------------- Main Window -----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyRawRoom")
        self.resize(1200, 800)

        self.thread_pool = QtCore.QThreadPool()

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

    def _setup_top_bar(self, parent_layout):
        bar_frame = QtWidgets.QFrame()
        bar_frame.setStyleSheet("""
            QFrame { background-color: #2b2b2b; border-bottom: 1px solid #3d3d3d; }
            QPushButton {
                background-color: transparent;
                border: none;
                color: #888;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:checked {
                color: #fff;
                border-bottom: 2px solid #dcdcdc;
            }
            QPushButton:hover { color: #ccc; }
        """)
        bar_layout = QtWidgets.QHBoxLayout(bar_frame)
        bar_layout.setContentsMargins(10, 0, 10, 0)

        # Buttongroup for exclusivity logic is manual here for styling flexibility
        self.btn_gallery = QtWidgets.QPushButton("GALLERY")
        self.btn_gallery.setCheckable(True)
        self.btn_gallery.clicked.connect(self.switch_to_gallery)

        self.btn_edit = QtWidgets.QPushButton("EDIT")
        self.btn_edit.setCheckable(True)
        self.btn_edit.clicked.connect(self.switch_to_edit)

        bar_layout.addWidget(self.btn_gallery)
        bar_layout.addWidget(self.btn_edit)
        bar_layout.addStretch() # Push left

        parent_layout.addWidget(bar_frame)

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Removed view switching actions from menu

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
        folder = os.path.dirname(path)

        # Load the image
        self.editor.load_image(path)

        # Update carousel if needed
        if self.editor.current_folder != folder:
            self.editor.load_carousel_folder(folder)

        if self.editor.current_folder != folder:
            self.editor.load_carousel_folder(folder)

        self.switch_to_edit()

    def open_single_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open RAW", "", f"RAW ({' '.join(['*'+e for e in pyrawroom.SUPPORTED_EXTS])})")
        if path:
            self.open_editor(path)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
