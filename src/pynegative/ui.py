import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, Signal, Slot

from . import core as pynegative

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
            pil_img = pynegative.extract_thumbnail(self.path)
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
            img = pynegative.open_raw(self.path, half_size=True)

            # 2. Check for Sidecar Settings
            settings = pynegative.load_sidecar(self.path)
            mode = "sidecar"

            # 3. Fallback to Auto-Exposure
            if not settings:
                settings = pynegative.calculate_auto_exposure(img)
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
        self.settings = QtCore.QSettings("pyNegative", "Gallery")
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
        files = [f for f in self.current_folder.iterdir() if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS]

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


# ----------------- Custom Widgets -----------------
class HorizontalListWidget(QtWidgets.QListWidget):
    """A ListWidget that scrolls horizontally with the mouse wheel."""
    def wheelEvent(self, event):
        if event.angleDelta().y():
            # Scroll horizontally instead of vertically
            delta = event.angleDelta().y()
            # Most mice return 120 per notch. We apply a small multiplier for speed.
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)

class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with a header and a content area."""
    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Header
        self.header = QtWidgets.QPushButton(title)
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.clicked.connect(self.toggle)
        self.layout.addWidget(self.header)

        # Content Area
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout.setSpacing(2)
        self.layout.addWidget(self.content)

        if not expanded:
            self.content.hide()

    def toggle(self):
        if self.header.isChecked():
            self.content.show()
        else:
            self.content.hide()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

class ResetableSlider(QtWidgets.QSlider):
    """A QSlider that resets to a default value on double-click."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.default_slider_value = 0

    def mouseDoubleClickEvent(self, event):
        self.setValue(self.default_slider_value)
        # Trigger valueChanged signal explicitly if needed, but setValue does it
        super().mouseDoubleClickEvent(event)


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#1a1a1a")))

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        # Background item (Low-res 1000-1500px, GPU scaled)
        self._bg_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._bg_item)
        self._bg_item.setZValue(0)

        # Foreground item (High-res ROI patch)
        self._fg_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._fg_item)
        self._fg_item.setZValue(1)

        self._current_zoom = 1.0
        self._is_fitting = True
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # Signal for redraw on pan
        self.horizontalScrollBar().valueChanged.connect(self._sync_view)
        self.verticalScrollBar().valueChanged.connect(self._sync_view)

    def _sync_view(self):
        if not self._is_fitting:
            self.zoomChanged.emit(self._current_zoom)

    def set_pixmaps(self, bg_pix, full_w, full_h, roi_pix=None, roi_x=0, roi_y=0, roi_w=0, roi_h=0):
        """Unified update for both layers to ensure alignment."""
        # 1. Update Background
        self._bg_item.setPixmap(bg_pix)
        if bg_pix.width() > 0:
            s_w = full_w / bg_pix.width()
            s_h = full_h / bg_pix.height()
            self._bg_item.setTransform(QtGui.QTransform.fromScale(s_w, s_h))

        # 2. Update Scene Rect
        self._scene.setSceneRect(0, 0, full_w, full_h)

        # 3. Update ROI
        if roi_pix:
            self._fg_item.setPixmap(roi_pix)
            self._fg_item.setPos(roi_x, roi_y)
            # GPU Scale ROI if it was processed at lower resolution for performance
            if roi_w > 0 and roi_pix.width() > 0:
                rs_w = roi_w / roi_pix.width()
                rs_h = roi_h / roi_pix.height()
                self._fg_item.setTransform(QtGui.QTransform.fromScale(rs_w, rs_h))
            else:
                self._fg_item.setTransform(QtGui.QTransform())
            self._fg_item.show()
        else:
            self._fg_item.hide()

    def reset_zoom(self):
        if self._bg_item.pixmap().isNull() and self._scene.sceneRect().isEmpty():
             return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._current_zoom = self.transform().m11()
        self._is_fitting = True
        self.zoomChanged.emit(self._current_zoom)

    def set_zoom(self, scale, manual=True):
        if manual:
            self._is_fitting = False
        self._current_zoom = scale
        self.setTransform(QtGui.QTransform.fromScale(scale, scale))
        self.zoomChanged.emit(self._current_zoom)

    def wheelEvent(self, event):
        if self._scene.sceneRect().isEmpty():
            return

        # Zoom everywhere in the view for a better feel,
        # unless you specifically want to limit it to the image bounds.
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9

        # Calculate new zoom relative to current actual transform
        # in case _current_zoom got out of sync
        self._current_zoom = self.transform().m11()
        new_zoom = self._current_zoom * factor

        # Clamp to 50% - 400% range
        new_zoom = max(0.5, min(new_zoom, 4.0))

        if new_zoom != self._current_zoom:
            self.set_zoom(new_zoom, manual=True)

        event.accept()

class ZoomControls(QtWidgets.QFrame):
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ZoomControls")
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(10, 5, 10, 5)
        self.layout.setSpacing(10)

        # Slider (50 to 400)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(50, 400)
        self.slider.setValue(100)
        self.slider.setFixedWidth(120)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.layout.addWidget(self.slider)

        # Percentage Box
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(50, 400)
        self.spin.setValue(100)
        self.spin.setSuffix("%")
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin.setAlignment(Qt.AlignCenter)
        self.spin.setFixedWidth(60)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.layout.addWidget(self.spin)

        self.setStyleSheet("""
            QFrame#ZoomControls {
                background-color: rgba(36, 36, 36, 0.8);
                border-radius: 8px;
                border: 1px solid #404040;
            }
            QSpinBox {
                background-color: #1a1a1a;
                border: 1px solid #303030;
                border-radius: 4px;
                color: #e5e5e5;
            }
        """)

    def _on_slider_changed(self, val):
        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self.zoomChanged.emit(val / 100.0)

    def _on_spin_changed(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)
        self.zoomChanged.emit(val / 100.0)

    def update_zoom(self, scale):
        val = int(scale * 100)
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(val)
        self.spin.setValue(val)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)


# ----------------- Editor Widget -----------------
class EditorWidget(QtWidgets.QWidget):
    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.raw_path = None
        self.base_img_full = None  # The High-Res Proxy (e.g. 4000px)
        self._base_img_uint8 = None # Cached uint8 version for resizing
        self.current_qpixmap = None

        # Auto-save timer
        self.save_timer = QtCore.QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._auto_save_sidecar)

        # Throttled Render Timer (30 FPS)
        self.render_timer = QtCore.QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._on_render_timer_timeout)
        self._render_pending = False
        self._is_rendering_locked = False

        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # --- Left Panel (Controls) ---
        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("EditorPanel")
        self.panel.setFixedWidth(280)
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(10, 10, 10, 10)
        self.panel_layout.setSpacing(2)
        main_layout.addWidget(self.panel)

        # --- Canvas (Right Side) ---
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setObjectName("CanvasFrame")
        self.canvas_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas_frame)

        # View replaced with ZoomableGraphicsView
        self.view = ZoomableGraphicsView()

        # Layout for canvas + zoom controls
        self.canvas_container = QtWidgets.QGridLayout(self.canvas_frame)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.canvas_container.addWidget(self.view, 0, 0)

        # Zoom Controls (Bottom Right overlay)
        self.zoom_ctrl = ZoomControls()
        self.zoom_ctrl.zoomChanged.connect(lambda z: self.view.set_zoom(z, manual=True))
        self.view.zoomChanged.connect(self.zoom_ctrl.update_zoom)

        # Trigger ROI re-render when zoom or pan changes
        self.view.zoomChanged.connect(self.request_update)

        self.canvas_container.addWidget(self.zoom_ctrl, 0, 0, Qt.AlignBottom | Qt.AlignRight)
        self.zoom_ctrl.setContentsMargins(0, 0, 20, 20)

        # Carousel (Bottom)
        self.carousel = HorizontalListWidget()
        self.carousel.setObjectName("Carousel")
        self.carousel.setViewMode(QtWidgets.QListView.IconMode)
        self.carousel.setFlow(QtWidgets.QListView.LeftToRight) # Horizontal
        self.carousel.setWrapping(False)
        self.carousel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.carousel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.carousel.setFixedHeight(145)
        self.carousel.setIconSize(QtCore.QSize(100, 100))
        self.carousel.setSpacing(5)
        self.carousel.itemClicked.connect(self._on_carousel_item_clicked)
        self.canvas_container.addWidget(self.carousel, 1, 0)

        self._setup_controls()

    def resizeEvent(self, event):
        # Auto-fit image if in fitting mode
        # Using hasattr check to be absolutely safe against racing AttributeErrors
        if hasattr(self, 'base_img_full') and self.base_img_full is not None:
            if getattr(self.view, '_is_fitting', False):
                 self.view.reset_zoom()
        super().resizeEvent(event)

    def _setup_controls(self):
        # Wrap everything in a scroll area just in case
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.panel_layout.addWidget(scroll)

        container = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(container)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(5)
        scroll.setWidget(container)

        self.lbl_info = QtWidgets.QLabel("No file loaded")
        self.lbl_info.setObjectName("InfoLabel")
        self.lbl_info.setWordWrap(True)
        self.controls_layout.addWidget(self.lbl_info)

        # --- Tone Section ---
        self.tone_section = CollapsibleSection("TONE", expanded=True)
        self.controls_layout.addWidget(self.tone_section)

        self.val_exposure = 0.0
        self.val_contrast = 1.0
        self.val_whites = 1.0
        self.val_blacks = 0.0
        self.val_highlights = 0.0
        self.val_shadows = 0.0

        self._add_slider("Exposure", -4.0, 4.0, self.val_exposure, "val_exposure", 0.01, self.tone_section)
        self._add_slider("Contrast", 0.5, 2.0, self.val_contrast, "val_contrast", 0.01, self.tone_section)
        self._add_slider("Highlights", -1.0, 1.0, self.val_highlights, "val_highlights", 0.01, self.tone_section)
        self._add_slider("Shadows", -1.0, 1.0, self.val_shadows, "val_shadows", 0.01, self.tone_section)
        self._add_slider("Whites", 0.5, 1.5, self.val_whites, "val_whites", 0.01, self.tone_section, flipped=True)
        self._add_slider("Blacks", -0.2, 0.2, self.val_blacks, "val_blacks", 0.001, self.tone_section)

        # --- Color Section ---
        self.color_section = CollapsibleSection("COLOR", expanded=False)
        self.controls_layout.addWidget(self.color_section)

        self.val_saturation = 1.0
        self._add_slider("Saturation", 0.0, 2.0, self.val_saturation, "val_saturation", 0.01, self.color_section)

        # --- Details Section ---
        self.details_section = CollapsibleSection("DETAILS", expanded=False)
        self.controls_layout.addWidget(self.details_section)

        self.var_sharpen_enabled = False
        self.sharpen_checkbox = QtWidgets.QCheckBox("Enable Processing")
        self.sharpen_checkbox.toggled.connect(self._update_sharpen_state)
        self.details_section.add_widget(self.sharpen_checkbox)

        self.val_radius = 2.0
        self.val_percent = 150
        self.val_denoise = 0
        self._add_slider("Sharpen Radius", 0.5, 5.0, self.val_radius, "val_radius", 0.01, self.details_section)
        self._add_slider("Sharpen Amount", 0, 300, self.val_percent, "val_percent", 1, self.details_section)
        self._add_slider("De-noise", 0, 5, self.val_denoise, "val_denoise", 1, self.details_section)

        # Save Button
        self.controls_layout.addSpacing(10)
        self.btn_save = QtWidgets.QPushButton("Save Result")
        self.btn_save.setObjectName("SaveButton")
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setEnabled(False)
        self.controls_layout.addWidget(self.btn_save)

        self.controls_layout.addStretch()

    def _add_separator(self, spacing=10):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.panel_layout.addWidget(line)
        if spacing: self.panel_layout.addSpacing(spacing)

    def _add_slider(self, label_text, min_val, max_val, default, var_name, step_size, section=None, flipped=False):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(0)

        # Store flip state for programmatic updates
        setattr(self, f"{var_name}_flipped", flipped)

        # Top row: Label and Value
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label_text)
        val_lbl = QtWidgets.QLabel(f"{default:.2f}")
        val_lbl.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(lbl)
        row.addWidget(val_lbl)
        layout.addLayout(row)

        slider = ResetableSlider(QtCore.Qt.Horizontal)
        multiplier = 1000
        slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
        # Default value on initial setup
        slider.default_slider_value = int(default * multiplier)
        slider.setValue(int(default * multiplier))

        def on_change(val):
            actual = val / multiplier
            if flipped:
                # Map slider min..max to max..min
                # Formula: actual = s_max + s_min - actual
                actual = max_val + min_val - actual

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
        if section:
            section.add_widget(frame)
        else:
            self.panel_layout.addWidget(frame)

    def _set_slider_value(self, var_name, value):
        slider = getattr(self, f"{var_name}_slider", None)
        label = getattr(self, f"{var_name}_label", None)
        flipped = getattr(self, f"{var_name}_flipped", False)

        if slider:
            multiplier = 1000
            if flipped:
                # Need to find the min/max to calculate the flipped slider position
                # But we can just use the vars passed to _add_slider if we stored them
                # For now let's use the slider's own range
                s_min = slider.minimum() / multiplier
                s_max = slider.maximum() / multiplier
                val_to_set = (s_max + s_min) - value
                slider.setValue(int(val_to_set * multiplier))
            else:
                slider.setValue(int(value * multiplier))

            # Since this is a programmatic update (likely from load),
            # we also update the reset value.
            slider.default_slider_value = slider.value()

        if label:
            label.setText(f"{value:.2f}")
        setattr(self, var_name, value)


    def _update_sharpen_state(self, checked):
        self.var_sharpen_enabled = checked
        self.request_update()
        self.save_timer.start(500)

    def _auto_save_sidecar(self):
        if not self.raw_path:
            return

        settings = {
            "exposure": self.val_exposure,
            "contrast": self.val_contrast,
            "whites": self.val_whites,
            "blacks": self.val_blacks,
            "highlights": self.val_highlights,
            "shadows": self.val_shadows,
            "saturation": self.val_saturation,
            "sharpen_enabled": self.var_sharpen_enabled,
            "sharpen_radius": self.val_radius,
            "sharpen_percent": self.val_percent,
            "de_noise": self.val_denoise
        }
        pynegative.save_sidecar(self.raw_path, settings)

    def load_image(self, path):
        path = Path(path)
        self.lbl_info.setText(f"Loading: {path.name}")
        self.raw_path = path

        QtWidgets.QApplication.processEvents()

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

        # Apply Auto-Expose Settings
        if settings:
            self._set_slider_value("val_exposure", settings.get("exposure", 0.0))
            self._set_slider_value("val_contrast", settings.get("contrast", 1.0))
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
            self._set_slider_value("val_denoise", settings.get("de_noise", 0))

        self.base_img_full = img_arr # The half-res proxy

        # Clear caches for the new image
        self._base_img_uint8 = None
        if hasattr(self, '_img_render_base'):
            del self._img_render_base

        self.btn_save.setEnabled(True)
        self.request_update()

        # Request a fit once the UI settles
        QtCore.QTimer.singleShot(50, self.view.reset_zoom)
        QtCore.QTimer.singleShot(200, self.view.reset_zoom)

        self.lbl_info.setText(f"Loaded: {self.raw_path.name}")

    def request_update(self):
        """
        Requests an image redraw. Uses a throttled timer to maintain 30 FPS.
        """
        if self.base_img_full is None:
            return

        self._render_pending = True

        if not self._is_rendering_locked:
            self._process_pending_update()

    def _process_pending_update(self):
        """Processes the actual redraw and manages the throttle lockout."""
        if not self._render_pending or self.base_img_full is None:
            return

        # Perform the actual update
        self.update_preview()
        self._render_pending = False

        # Lock rendering for 33ms (30 FPS)
        self._is_rendering_locked = True
        self.render_timer.start(33) # 33ms interval for throttle

    def _on_render_timer_timeout(self):
        """Called when the 33ms throttle window expires."""
        self._is_rendering_locked = False
        # If another update was requested during the lockout, process it now
        if self._render_pending:
            self._process_pending_update()

    def update_preview(self):
        if self.base_img_full is None: return

        # Strategy: Single-Layer Dynamic ROI
        # 1. Determine the viewport size
        # 2. If zoomed out (Fit), process a 1500px global image.
        # 3. If zoomed in, process a crop matching the viewport size from base_img_full.

        full_h, full_w, _ = self.base_img_full.shape
        zoom_scale = self.view.transform().m11()
        vw, vh = self.view.viewport().width(), self.view.viewport().height()

        # Calculate fit scale
        fit_scale = 1.0
        if vw > 0 and vh > 0:
            fit_scale = min(vw / full_w, vh / full_h)

        is_zoomed_in = not self.view._is_fitting and (zoom_scale > fit_scale * 1.01 or zoom_scale > 0.99)

        # --- Part 1: Global Background ---
        # Cache the 1500px base
        if self._base_img_uint8 is None:
            self._base_img_uint8 = (self.base_img_full * 255).astype(np.uint8)

        scale = 1500 / max(full_h, full_w)
        target_h, target_w = int(full_h * scale), int(full_w * scale)
        if not hasattr(self, '_img_render_base') or self._img_render_base.shape[0] != target_h:
            temp_pil = Image.fromarray(self._base_img_uint8)
            temp_pil = temp_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
            self._img_render_base = np.array(temp_pil).astype(np.float32) / 255.0

        # Process Background
        processed_bg, _ = pynegative.apply_tone_map(
            self._img_render_base,
            exposure=self.val_exposure, contrast=self.val_contrast,
            blacks=self.val_blacks, whites=self.val_whites,
            shadows=self.val_shadows, highlights=self.val_highlights,
            saturation=self.val_saturation
        )
        processed_bg *= 255
        pil_bg = Image.fromarray(processed_bg.astype(np.uint8))
        pix_bg = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_bg))

        # --- Part 2: Detail ROI (Only if zoomed in) ---
        pix_roi, roi_x, roi_y, roi_w, roi_h = None, 0, 0, 0, 0
        if is_zoomed_in:
            roi = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
            ix_min = max(0, int(roi.left()))
            ix_max = min(full_w, int(roi.right()))
            iy_min = max(0, int(roi.top()))
            iy_max = min(full_h, int(roi.bottom()))

            rw = ix_max - ix_min
            rh = iy_max - iy_min

            if rw > 10 and rh > 10:
                # Performance Cap: Max 1.5 million pixels for real-time ROI
                MAX_REALTIME_PIXELS = 1_500_000
                current_pixels = rw * rh

                crop = self.base_img_full[iy_min:iy_max, ix_min:ix_max]

                if current_pixels > MAX_REALTIME_PIXELS:
                    # Scale down for processing, then GPU upscale later
                    p_scale = (MAX_REALTIME_PIXELS / current_pixels) ** 0.5
                    p_w, p_h = int(rw * p_scale), int(rh * p_scale)

                    pil_crop = Image.fromarray((crop * 255).astype(np.uint8))
                    pil_crop = pil_crop.resize((p_w, p_h), Image.Resampling.BILINEAR)
                    crop_to_proc = np.array(pil_crop).astype(np.float32) / 255.0
                else:
                    crop_to_proc = crop

                processed_roi, _ = pynegative.apply_tone_map(
                    crop_to_proc,
                    exposure=self.val_exposure, contrast=self.val_contrast,
                    blacks=self.val_blacks, whites=self.val_whites,
                    shadows=self.val_shadows, highlights=self.val_highlights,
                    saturation=self.val_saturation
                )
                processed_roi *= 255
                pil_roi = Image.fromarray(processed_roi.astype(np.uint8))

                if self.var_sharpen_enabled:
                    pil_roi = pynegative.sharpen_image(pil_roi, self.val_radius, self.val_percent)
                    if self.val_denoise > 0:
                        pil_roi = pynegative.de_noise_image(pil_roi, self.val_denoise)

                pix_roi = QtGui.QPixmap.fromImage(ImageQt.ImageQt(pil_roi))
                roi_x, roi_y = ix_min, iy_min
                roi_w, roi_h = rw, rh

        # --- Final Update ---
        self.view.set_pixmaps(pix_bg, full_w, full_h, pix_roi, roi_x, roi_y, roi_w, roi_h)

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
                full_img = pynegative.open_raw(self.raw_path, half_size=False)

                # Process full resolution with current settings
                img, _ = pynegative.apply_tone_map(
                    full_img,
                    exposure=self.val_exposure,
                    contrast=self.val_contrast,
                    blacks=self.val_blacks,
                    whites=self.val_whites,
                    shadows=self.val_shadows,
                    highlights=self.val_highlights,
                    saturation=self.val_saturation
                )
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                if self.var_sharpen_enabled:
                    pil_img = pynegative.sharpen_image(pil_img, self.val_radius, self.val_percent)
                    if self.val_denoise > 0:
                        pil_img = pynegative.de_noise_image(pil_img, self.val_denoise)

                pynegative.save_image(pil_img, path)
                QtWidgets.QMessageBox.information(self, "Saved", f"Saved full resolution to {path}")
                self.lbl_info.setText(f"Saved: {path.name}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
                self.lbl_info.setText("Error saving")

    def load_carousel_folder(self, folder):
        self.current_folder = Path(folder)
        self.carousel.clear()

        files = sorted([f for f in self.current_folder.iterdir() if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS])

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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open RAW", "", f"RAW ({' '.join(['*'+e for e in pynegative.SUPPORTED_EXTS])})")
        if path:
            self.open_editor(path)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
