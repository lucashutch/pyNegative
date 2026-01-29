import sys
from pathlib import Path
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
        self.path = Path(path)
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
            img = pyrawroom.open_raw(self.path, half_size=True)

            # 2. Check for Sidecar Settings
            settings = pyrawroom.load_sidecar(self.path)
            mode = "sidecar"

            # 3. Fallback to Auto-Exposure
            if not settings:
                settings = pyrawroom.calculate_auto_exposure(img)
                mode = "auto"

            self.signals.finished.emit(str(self.path), img, settings)
        except Exception as e:
            print(f"Error loading RAW {self.path}: {e}")
            self.signals.finished.emit(str(self.path), None, None)


class GalleryWidget(QtWidgets.QWidget):
    imageSelected = QtCore.Signal(str) # Path

    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.settings = QtCore.QSettings("pyRawRoom", "Gallery")
        self._init_ui()
        self._load_last_folder()

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
        self.btn_open_folder = QtWidgets.QPushButton("Open Folder")
        self.btn_open_folder.clicked.connect(self.browse_folder)
        top_bar.addWidget(self.btn_open_folder)
        top_bar.addStretch()
        grid_layout.addLayout(top_bar)

        # Grid View
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setObjectName("GalleryGrid")
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(180, 180))
        self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        grid_layout.addWidget(self.list_widget)

        self.stack.addWidget(self.grid_container)

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

    def browse_folder(self):
        # Start from last folder if available
        start_dir = ""
        if self.current_folder and self.current_folder.exists():
            start_dir = str(self.current_folder)

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Folder", start_dir)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.current_folder = Path(folder)
        self.list_widget.clear()

        # Save to settings
        self.settings.setValue("last_folder", str(self.current_folder))

        # Switch to grid view
        self.stack.setCurrentWidget(self.grid_container)

        # Find raw files
        files = [f for f in self.current_folder.iterdir() if f.is_file() and f.suffix.lower() in pyrawroom.SUPPORTED_EXTS]

        for path in files:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
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

        # Auto-save timer
        self.save_timer = QtCore.QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._auto_save_sidecar)

        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # --- Left Panel (Controls) ---
        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("EditorPanel")
        self.panel.setContentsMargins(10, 10, 10, 10)
        self.panel.setFixedWidth(350)
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        main_layout.addWidget(self.panel)

        # --- Canvas (Right Side) ---
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setObjectName("CanvasFrame")
        self.canvas_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas_frame)

        self.canvas_label = QtWidgets.QLabel()
        self.canvas_label.setObjectName("CanvasLabel")
        self.canvas_label.setAlignment(QtCore.Qt.AlignCenter)

        canvas_layout = QtWidgets.QVBoxLayout(self.canvas_frame)
        canvas_layout.addWidget(self.canvas_label)

        self.canvas_label.installEventFilter(self)

        # Carousel (Bottom)
        self.carousel = QtWidgets.QListWidget()
        self.carousel.setObjectName("Carousel")
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
        self.lbl_info.setObjectName("InfoLabel")
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

        # Saturation
        self._add_separator()
        self.val_saturation = 1.0
        self._add_slider("Saturation", 0.0, 2.0, self.val_saturation, "val_saturation", 0.01)

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
        self.btn_save.setObjectName("SaveButton")
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

            # Trigger auto-save
            self.save_timer.start(1000) # Save after 1 second of inactivity

        slider.valueChanged.connect(on_change)

        # Store refs
        setattr(self, f"{var_name}_slider", slider)
        setattr(self, f"{var_name}_label", val_lbl) # Store label for updates

        layout.addWidget(slider)
        layout.addWidget(val_lbl)
        self.panel_layout.addWidget(frame)

    def _set_slider_value(self, var_name, value):
        slider = getattr(self, f"{var_name}_slider", None)
        label = getattr(self, f"{var_name}_label", None)
        if slider:
            multiplier = 1000 # Assuming all sliders use this multiplier
            slider.setValue(int(value * multiplier))
            setattr(self, var_name, value)
            if label:
                label.setText(f"{value:.2f}")


    def _update_sharpen_state(self, checked):
        self.var_sharpen_enabled = checked
        self.request_update()
        self.save_timer.start(500)

    def _auto_save_sidecar(self):
        if not self.raw_path:
            return

        settings = {
            "exposure": self.val_exposure,
            "whites": self.val_whites,
            "blacks": self.val_blacks,
            "highlights": self.val_highlights,
            "shadows": self.val_shadows,
            "saturation": self.val_saturation,
            "sharpen_enabled": self.var_sharpen_enabled,
            "sharpen_radius": self.val_radius,
            "sharpen_percent": self.val_percent
        }
        pyrawroom.save_sidecar(self.raw_path, settings)

    def load_image(self, path):
        path = Path(path)
        self.lbl_info.setText(f"Loading: {path.name}")
        self.raw_path = path

        # STAGE 1: Instant Preview (Thumbnail or Cached)
        # Check if we have a fast thumbnail available
        # EXPERIMENT: Skip embedded thumbnail to test proxy speed/flicker
        # try:
        #     thumb_pil = pyrawroom.extract_thumbnail(path)
        #     if thumb_pil:
        #         # Show this immediately
        #         h, w = thumb_pil.size
        #         # We can't do full processing on a thumbnail usually, but we can display it
        #         self.base_img_preview = np.array(thumb_pil).astype(np.float32) / 255.0
        #
        #         # If thumbnail is rotated (common in exif), we might need handling,
        #         # but for now we assume extract_thumbnail (via rawpy) handles it or returns simple RGB.
        #         # Just show it:
        #         self.request_update()
        # except Exception:
        #     pass # Fallback to waiting

        QtWidgets.QApplication.processEvents()

        # STAGE 2: Async Full Load
        self.btn_save.setEnabled(False) # Disable save until full load

        loader = RawLoader(path)
        loader.signals.finished.connect(self._on_raw_loaded)
        self.thread_pool.start(loader)

    def _on_raw_loaded(self, path, img_arr, settings):
        if Path(path) != self.raw_path:
            return # User switched images already

        if img_arr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load image")
            return

        self.base_img_full = img_arr # Actually this is now the proxy

        # Apply Auto-Expose Settings
        if settings:
            self._set_slider_value("val_exposure", settings.get("exposure", 0.0))
            self._set_slider_value("val_whites", settings.get("whites", 1.0))
            self._set_slider_value("val_blacks", settings.get("blacks", 0.0))
            self._set_slider_value("val_saturation", settings.get("saturation", 1.0))
            self._set_slider_value("val_highlights", settings.get("highlights", 0.0))
            self._set_slider_value("val_shadows", settings.get("shadows", 0.0))

            # Sharpening state
            sharpen_on = settings.get("sharpen_enabled", False)
            self.sharpen_checkbox.setChecked(sharpen_on)
            self.var_sharpen_enabled = sharpen_on
            self._set_slider_value("val_radius", settings.get("sharpen_radius", 2.0))
            self._set_slider_value("val_percent", settings.get("sharpen_percent", 150))

        # Create high-quality preview (max 1000px) from proxy
        # Since proxy is smaller, it might already be close to screen size
        h, w, _ = self.base_img_full.shape
        scale = 1000 / max(h, w)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            temp_pil = Image.fromarray((self.base_img_full * 255).astype(np.uint8))
            temp_pil = temp_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            self.base_img_preview = np.array(temp_pil).astype(np.float32) / 255.0
        else:
            self.base_img_preview = self.base_img_full.copy()

        self.btn_save.setEnabled(True)
        self.request_update()

        # Indicate if using proxy
        is_proxy = " (Proxy)" if h < 4000 else "" # Heuristic
        self.lbl_info.setText(f"Loaded: {self.raw_path.name}{is_proxy}")

    def request_update(self):
        if self.base_img_preview is not None:
            self.update_preview()

    def update_preview(self):
        if self.base_img_preview is None: return

        # Process
        img, _ = pyrawroom.apply_tone_map(
            self.base_img_preview,
            exposure=self.val_exposure, blacks=self.val_blacks, whites=self.val_whites,
            shadows=self.val_shadows, highlights=self.val_highlights,
            saturation=self.val_saturation
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

        input_dir = self.raw_path.parent
        default_name = self.raw_path.with_suffix(".jpg").name
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save", str(input_dir / default_name), "JPEG (*.jpg);;HEIF (*.heic)")

        if path:
            path = Path(path)
            try:
                # RELOAD FULL RESOLUTION FOR SAVING
                self.lbl_info.setText("Processing Full Res...")
                QtWidgets.QApplication.processEvents()

                # Explicitly request full size (half_size=False)
                full_img = pyrawroom.open_raw(self.raw_path, half_size=False)

                # Process full resolution with current settings
                img, _ = pyrawroom.apply_tone_map(
                    full_img,
                    exposure=self.val_exposure,
                    blacks=self.val_blacks,
                    whites=self.val_whites,
                    shadows=self.val_shadows,
                    highlights=self.val_highlights,
                    saturation=self.val_saturation
                )
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                if self.var_sharpen_enabled:
                    pil_img = pyrawroom.sharpen_image(pil_img, self.val_radius, self.val_percent)

                pyrawroom.save_image(pil_img, path)
                QtWidgets.QMessageBox.information(self, "Saved", f"Saved full resolution to {path}")
                self.lbl_info.setText(f"Saved: {path.name}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
                self.lbl_info.setText("Error saving")

    def load_carousel_folder(self, folder):
        self.current_folder = Path(folder)
        self.carousel.clear()

        files = sorted([f for f in self.current_folder.iterdir() if f.is_file() and f.suffix.lower() in pyrawroom.SUPPORTED_EXTS])

        for path in files:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
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
        if Path(path) != self.raw_path:
            self.load_image(path)


# ----------------- Main Window -----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyRawRoom")
        self.resize(1200, 800)

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
        style_path = Path(__file__).parent / "styles.qss"
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
        path = Path(path)
        folder = path.parent

        # Load the image
        self.editor.load_image(path)

        # Update carousel if needed
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
