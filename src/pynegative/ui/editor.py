from pathlib import Path
from functools import partial
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt

from .loaders import RawLoader
from .widgets import (
    ZoomControls,
    ToastWidget,
    ZoomableGraphicsView,
    StarRatingWidget,
    ComparisonOverlay,
    ComparisonHandle,
)
from .imageprocessing import ImageProcessingPipeline
from .editingcontrols import EditingControls
from .settingsmanager import SettingsManager
from .carouselmanager import CarouselManager
from .. import core as pynegative


class PreviewStarRatingWidget(StarRatingWidget):
    """A larger star rating widget for preview mode with 30px stars."""

    def _create_star_pixmap(self, filled):
        size = 30
        self.setFixedHeight(size)
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        font = self.font()
        font.setPointSize(24)
        painter.setFont(font)

        if filled:
            painter.setPen(QtGui.QColor("#f0c419"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
        else:
            painter.setPen(QtGui.QColor("#808080"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "☆")

        painter.end()
        return pixmap


class EditorWidget(QtWidgets.QWidget):
    ratingChanged = QtCore.Signal(str, int)
    imageDoubleClicked = QtCore.Signal()

    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.raw_path = None

        # Auto-save timer for metadata
        self.save_timer = QtCore.QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._auto_save_sidecar)

        # Debounce timer for UI settings (grid/carousel size)
        self._settings_save_timer = QtCore.QTimer()
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(500)  # Save 500ms after last move
        self._settings_save_timer.timeout.connect(self._save_ui_settings)

        # Initialize components
        self._init_components()
        self._init_ui()
        self._setup_connections()
        self._setup_keyboard_shortcuts()

    def _init_components(self):
        """Initialize all component instances."""
        self.image_processor = ImageProcessingPipeline(self.thread_pool, self)
        self.editing_controls = EditingControls(self)
        self.settings_manager = SettingsManager(self)
        self.carousel_manager = CarouselManager(self.thread_pool, self)

        # Throttling for rotation handle updates
        self._rotation_handle_throttle_timer = QtCore.QTimer()
        self._rotation_handle_throttle_timer.setSingleShot(True)
        self._rotation_handle_throttle_timer.setInterval(33)  # ~30fps
        self._rotation_handle_throttle_timer.timeout.connect(
            self._apply_pending_rotation
        )
        self._pending_rotation_from_handle = None

        # Comparison overlay state
        self._comparison_enabled = False

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Use a splitter to allow manual resizing of the sidebar
        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- Left Panel (Controls) ---
        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("EditorPanel")
        self.panel.setMinimumWidth(320)
        self.panel.setMaximumWidth(600)
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(8, 10, 8, 10)
        self.panel_layout.setSpacing(2)
        self.splitter.addWidget(self.panel)

        # Add editing controls to panel
        self.panel_layout.addWidget(self.editing_controls)

        # --- Canvas (Right Side) ---
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setObjectName("CanvasFrame")
        self.canvas_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.splitter.addWidget(self.canvas_frame)

        # Set initial sizes for the splitter: sidebar at 360px, canvas takes rest
        self.splitter.setSizes([360, 1000])

        # Layout for canvas + zoom controls + carousel
        self.canvas_container = QtWidgets.QVBoxLayout(self.canvas_frame)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.canvas_container.setSpacing(0)

        # Vertical splitter for canvas and carousel
        self.canvas_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.canvas_container.addWidget(self.canvas_splitter)

        # 1. View replaced with ZoomableGraphicsView
        self.view = ZoomableGraphicsView()
        self.canvas_splitter.addWidget(self.view)

        # 2. Add carousel from carousel manager
        self.carousel_widget = self.carousel_manager.get_widget()
        self.canvas_splitter.addWidget(self.carousel_widget)
        self.carousel_widget.installEventFilter(self)

        # Setup splitter properties
        self.canvas_splitter.setStretchFactor(0, 5)  # View takes more space
        self.canvas_splitter.setStretchFactor(
            1, 0
        )  # Carousel doesn't stretch more than needed
        self.canvas_splitter.setHandleWidth(4)  # Match QSS

        # Enforce height constraints via widget limits
        self.carousel_widget.setMinimumHeight(100)
        self.carousel_widget.setMaximumHeight(400)
        self.canvas_splitter.setCollapsible(1, False)

        self.canvas_splitter.splitterMoved.connect(self._on_carousel_splitter_moved)

        # Zoom Controls (Bottom Right overlay - will be manual positioned since QSplitter doesn't support overlays)
        self.zoom_ctrl = ZoomControls()
        self.zoom_ctrl.setParent(self.canvas_frame)  # Floating on top of splitter
        self.zoom_ctrl.zoomChanged.connect(lambda z: self.view.set_zoom(z, manual=True))
        self.view.zoomChanged.connect(self.zoom_ctrl.update_zoom)

        # Set up context menus
        self.view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_main_photo_context_menu)

        # Set view reference for image processor
        self.image_processor.set_view_reference(self.view)

        # Trigger ROI re-render when zoom or pan changes
        self.view.zoomChanged.connect(self._request_update_from_view)
        self.view.doubleClicked.connect(self.imageDoubleClicked.emit)

        # Toast widget for notifications
        self.toast = ToastWidget(self.canvas_frame)

        # Performance metric label
        self.perf_label = QtWidgets.QLabel(self.canvas_frame)
        self.perf_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 128); color: white; padding: 4px; border-radius: 4px;"
        )
        self.perf_label.setContentsMargins(10, 0, 0, 10)
        self.perf_label.hide()

        self.preview_rating_widget = PreviewStarRatingWidget(self.canvas_frame)
        self.preview_rating_widget.setObjectName("PreviewRatingWidget")
        self.preview_rating_widget.setStyleSheet("""
            QWidget#PreviewRatingWidget {
                background-color: rgba(0, 0, 0, 0.5);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self.preview_rating_widget.hide()

        # Comparison Drawing Layer
        self.comparison_overlay = ComparisonOverlay(self.canvas_frame)
        self.comparison_overlay.setView(self.view)
        self.comparison_overlay.raise_()

        # Comparison Toggle Button
        self.comparison_btn = QtWidgets.QToolButton(self.canvas_frame)
        self.comparison_btn.setFixedSize(30, 30)
        self.comparison_btn.setCheckable(True)
        self.comparison_btn.setToolTip("Compare (U)")
        self.comparison_btn.setIcon(self._create_comparison_icon())
        self.comparison_btn.setStyleSheet("""
            QToolButton {
                background-color: rgba(0, 0, 0, 128);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 4px;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: rgba(0, 0, 0, 180);
                border-color: rgba(255, 255, 255, 80);
            }
            QToolButton:checked {
                background-color: rgba(138, 43, 226, 180);
                border-color: rgba(138, 43, 226, 200);
            }
        """)
        self.comparison_btn.clicked.connect(self._toggle_comparison)
        self.comparison_btn.show()

        # Comparison Handle
        self.comparison_handle = ComparisonHandle(self.canvas_frame)
        self.comparison_handle.dragged.connect(self._on_comparison_handle_dragged)
        self.comparison_handle.hide()

        # Ensure UI controls are ABOVE everything
        self.zoom_ctrl.raise_()
        self.preview_rating_widget.raise_()
        self.toast.raise_()
        self.perf_label.raise_()
        self.comparison_btn.raise_()
        self.comparison_handle.raise_()

        # Initial positioning
        QtCore.QTimer.singleShot(100, self._reposition_floating_ui)
        QtCore.QTimer.singleShot(150, self._load_carousel_height)

    def _load_carousel_height(self):
        """Load carousel height from settings."""
        settings = QtCore.QSettings("pyNegative", "Editor")
        height = int(settings.value("carousel_height", 210))
        self.carousel_manager.set_carousel_height(height)

        # Adjust splitter sizes
        total_high = self.canvas_frame.height()
        if total_high > 0:
            self.canvas_splitter.setSizes([total_high - height, height])

    def _on_carousel_splitter_moved(self, pos, index):
        """Handle carousel resizing via splitter."""
        if index == 1:  # Handle between view and carousel
            # Update carousel contents (throttled inside manager)
            self.carousel_manager.set_carousel_height(self.carousel_widget.height())

            # Update floating UI (real-time)
            self._reposition_floating_ui()

            # Debounce settings save
            self._settings_save_timer.start()

    def _save_ui_settings(self):
        """Save UI state like carousel height."""
        height = self.carousel_widget.height()
        settings = QtCore.QSettings("pyNegative", "Editor")
        settings.setValue("carousel_height", height)

    def _setup_connections(self):
        """Setup signal/slot connections between components."""
        # Editing controls -> Image processor
        self.editing_controls.settingChanged.connect(self._on_setting_changed)
        self.editing_controls.cropToggled.connect(self._on_crop_toggled)
        self.editing_controls.aspectRatioChanged.connect(self.view.set_aspect_ratio)
        self.editing_controls.ratingChanged.connect(self._on_rating_changed)
        self.editing_controls.autoWbRequested.connect(self._on_auto_wb_requested)
        self.editing_controls.presetApplied.connect(self._on_preset_applied)

        # Rotation handles -> Editor
        self.view.rotationChanged.connect(self._on_rotation_handle_changed)

        # Histogram logic
        self.editing_controls.histogram_section.expandedChanged.connect(
            self.image_processor.set_histogram_enabled
        )
        self.image_processor.histogramUpdated.connect(
            self.editing_controls.histogram_widget.set_data
        )

        # Image processor -> View
        self.image_processor.previewUpdated.connect(self.view.set_pixmaps)
        self.image_processor.performanceMeasured.connect(self._on_performance_measured)

        # Comparison logic
        self.image_processor.uneditedPixmapUpdated.connect(
            self._on_unedited_pixmap_updated
        )
        self.image_processor.editedPixmapUpdated.connect(self._on_edited_pixmap_updated)

        # Settings manager
        self.settings_manager.showToast.connect(self.show_toast)
        self.settings_manager.settingsCopied.connect(self._on_settings_copied)
        self.settings_manager.settingsPasted.connect(self._on_settings_pasted)
        self.settings_manager.undoStateChanged.connect(self._on_undo_state_changed)

        # Carousel manager
        self.carousel_manager.imageSelected.connect(self._on_carousel_image_selected)
        self.carousel_manager.selectionChanged.connect(
            self._on_carousel_selection_changed
        )
        self.carousel_manager.selectionChanged.connect(
            self._on_carousel_keyboard_navigation
        )
        self.carousel_manager.contextMenuRequested.connect(
            self._handle_carousel_context_menu
        )

        # Preview rating widget
        self.preview_rating_widget.ratingChanged.connect(
            self._on_preview_rating_changed
        )

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, self._undo)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self, self._redo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), self, self._redo)
        QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Copy, self, self._handle_copy_shortcut
        )
        QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Paste, self, self._handle_paste_shortcut
        )
        QtGui.QShortcut(
            QtGui.QKeySequence("F12"), self, self._toggle_performance_overlay
        )
        QtGui.QShortcut(QtGui.QKeySequence("F10"), self, self._cycle_denoise_mode)

        # Comparison overlay shortcut
        QtGui.QShortcut(QtGui.QKeySequence("U"), self, self.comparison_btn.animateClick)

        # Rating shortcuts (1-5, 0)
        for key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_0]:
            if key == Qt.Key_0:
                QtGui.QShortcut(key, self, partial(self._set_rating_by_number, 0))
            else:
                QtGui.QShortcut(
                    key,
                    self,
                    partial(self._set_rating_by_number, key.value - Qt.Key_0.value),
                )

        # Navigation shortcuts - use ApplicationShortcut to work even when child widgets have focus
        nav_left = QtGui.QShortcut(Qt.Key_Left, self, self._navigate_previous)
        nav_left.setContext(Qt.ApplicationShortcut)
        nav_right = QtGui.QShortcut(Qt.Key_Right, self, self._navigate_next)
        nav_right.setContext(Qt.ApplicationShortcut)

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        # Auto-fit image if in fitting mode
        if (
            hasattr(self, "image_processor")
            and self.image_processor.base_img_full is not None
        ):
            if self.view._is_fitting:
                self.view.reset_zoom()

        # Position floating UI elements after layout settles
        QtCore.QTimer.singleShot(0, self._reposition_floating_ui)

    def _reposition_floating_ui(self):
        """Reposition floating controls (comparison btn, handle, etc)."""
        if not hasattr(self, "comparison_btn") or not hasattr(self, "zoom_ctrl"):
            return

        cw = self.canvas_frame.width()
        vx, vy = self.view.x(), self.view.y()
        vw, vh = self.view.width(), self.view.height()

        # 1. Position zoom controls at bottom right of the VIEW specifically
        if hasattr(self, "zoom_ctrl"):
            zx = vx + vw - self.zoom_ctrl.width() - 20
            zy = vy + vh - self.zoom_ctrl.height() - 20
            self.zoom_ctrl.move(zx, zy)
            self.zoom_ctrl.show()
            self.zoom_ctrl.raise_()

        # 2. Position preview rating at bottom left of the VIEW specifically
        if hasattr(self, "preview_rating_widget"):
            prx = vx + 20
            pry = vy + vh - self.preview_rating_widget.height() - 20
            self.preview_rating_widget.move(prx, pry)

        # 3. Position performance label above the rating widget
        if hasattr(self, "perf_label"):
            px = vx + 20
            py = (
                vy
                + vh
                - self.preview_rating_widget.height()
                - self.perf_label.height()
                - 30
            )
            self.perf_label.move(px, py)

        # 4. Position Comparison Button exactly 10px above zoom slider
        if hasattr(self, "comparison_btn") and hasattr(self, "zoom_ctrl"):
            bx = vx + vw - self.comparison_btn.width() - 20
            by = vy + vh - self.zoom_ctrl.height() - self.comparison_btn.height() - 30
            self.comparison_btn.move(bx, by)
            self.comparison_btn.show()
            self.comparison_btn.raise_()

        # 5. Position Comparison Handle and Overlay
        if hasattr(self, "comparison_overlay"):
            self.comparison_overlay.setGeometry(self.view.geometry())
            self.comparison_overlay.raise_()

        self._update_comparison_handle_position()

        # 6. Center Toast Widget
        if hasattr(self, "toast"):
            self.toast.move(
                (cw - self.toast.width()) // 2, (vh - self.toast.height()) // 2 + vy
            )

    def _update_comparison_handle_position(self):
        if hasattr(self, "comparison_handle") and hasattr(self, "comparison_overlay"):
            split_pos = self.comparison_overlay._split_position
            split_x = int(self.canvas_frame.width() * split_pos)
            hx = split_x - (self.comparison_handle.width() / 2)
            hy = (self.canvas_frame.height() - self.comparison_handle.height()) / 2
            self.comparison_handle.move(hx, hy)

    def _create_comparison_icon(self):
        pixmap = QtGui.QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        left_color = QtGui.QColor(100, 100, 100)
        right_color = QtGui.QColor(150, 150, 150)
        painter.setBrush(left_color)
        painter.setPen(Qt.NoPen)
        painter.drawRect(2, 4, 8, 16)
        painter.setBrush(right_color)
        painter.drawRect(14, 4, 8, 16)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        painter.drawLine(12, 2, 12, 22)
        painter.end()
        return QtGui.QIcon(pixmap)

    def _toggle_comparison(self):
        """Toggle comparison mode."""
        if hasattr(self, "comparison_btn"):
            # If called from button click, it's already toggled.
            # If called from keyboard, we toggle it manually.
            is_checked = self.comparison_btn.isChecked()
            # If called from keyboard shortcut (Qt.Key_U), we need to flip it
            # But wait, button.clicked already flips it if it's checkable.
            # The shortcut should just click the button.
            self._on_comparison_toggled(is_checked)

    def _on_comparison_toggled(self, enabled):
        """Handle comparison mode toggle."""
        self._comparison_enabled = enabled
        self.comparison_overlay.setComparisonActive(enabled)

        if enabled:
            self.comparison_handle.show()
            self._update_comparison_handle_position()
            unedited_pixmap = self.image_processor.get_unedited_pixmap()
            self.comparison_overlay.setUneditedPixmap(unedited_pixmap)
            if self.view._bg_item and self.view._bg_item.pixmap():
                self.comparison_overlay.setEditedPixmap(self.view._bg_item.pixmap())
            self.show_toast("Comparison enabled")
        else:
            self.comparison_handle.hide()
            self.show_toast("Comparison disabled")

    def _on_unedited_pixmap_updated(self, pixmap):
        """Handle unedited pixmap updates from image processor."""
        if self._comparison_enabled and self.comparison_overlay:
            self.comparison_overlay.setUneditedPixmap(pixmap)

    def _on_edited_pixmap_updated(self, pixmap):
        """Handle edited pixmap updates from image processor (live preview)."""
        if self._comparison_enabled and self.comparison_overlay:
            self.comparison_overlay.setEditedPixmap(pixmap)

    def _on_comparison_handle_dragged(self, global_x):
        local_pos = self.canvas_frame.mapFromGlobal(QtCore.QPointF(global_x, 0))
        split_pos = max(0.0, min(1.0, local_pos.x() / self.canvas_frame.width()))
        self.comparison_overlay.setSplitPosition(split_pos)
        self._update_comparison_handle_position()

    def eventFilter(self, obj, event):
        """Filter events to handle arrow key navigation in carousel."""
        if obj == self.carousel_widget and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == Qt.Key_Left:
                self._navigate_previous()
                return True
            elif event.key() == Qt.Key_Right:
                self._navigate_next()
                return True
        return super().eventFilter(obj, event)

    def clear(self):
        """Clear the editor state."""
        self.raw_path = None
        self.setWindowTitle("Editor")
        self.editing_controls.reset_sliders()
        self.editing_controls.set_rating(0)
        self.view.reset_zoom()
        self.view.set_pixmaps(QtGui.QPixmap(), 0, 0)
        self.carousel_manager.clear()
        self.settings_manager.clear_clipboard()

    def update_rating_for_path(self, path, rating):
        """Update rating for a specific path."""
        if self.raw_path and str(self.raw_path) == path:
            self.editing_controls.set_rating(rating)
            self.preview_rating_widget.set_rating(rating)

    def load_image(self, path):
        """Load an image for editing."""
        path = Path(path)
        self.raw_path = path

        # Reset Crop Tool
        self.editing_controls.set_crop_checked(False)
        self.view.set_crop_mode(False)

        # Update comparison overlay if enabled
        if self._comparison_enabled and self.comparison_overlay:
            unedited_pixmap = self.image_processor.get_unedited_pixmap()
            self.comparison_overlay.setUneditedPixmap(unedited_pixmap)

        QtWidgets.QApplication.processEvents()

        loader = RawLoader(path)
        loader.signals.finished.connect(self._on_raw_loaded)
        self.thread_pool.start(loader)

    def load_carousel_folder(self, folder):
        """Load a folder of images into the carousel."""
        self.current_folder = Path(folder)
        self.settings_manager.clear_clipboard()  # Clear clipboard on context change
        self.carousel_manager.load_folder(folder)

    def set_carousel_images(self, image_list, current_path):
        """Set specific images in the carousel."""
        self.carousel_manager.set_images(image_list, Path(current_path))
        self.settings_manager.clear_clipboard()  # Clear clipboard on context change

    def show_toast(self, message):
        """Show a toast notification."""
        self.toast.show_message(message)

    def set_preview_mode(self, enabled):
        """Set preview mode (hide/show controls panel)."""
        self.panel.setVisible(not enabled)
        self.preview_rating_widget.setVisible(enabled)

    def open(self, path, image_list=None):
        """Open an image for editing."""
        if not isinstance(path, Path):
            path = Path(path)

        if image_list:
            self.set_carousel_images(image_list, path)
        else:
            self.load_carousel_folder(path.parent)
            self.carousel_manager.select_image(path)

        self.load_image(path)

    # --- Signal handlers ---

    def _on_setting_changed(self, setting_name, value):
        """Handle setting change from editing controls."""
        self.image_processor.set_processing_params(**{setting_name: value})

        # Handle Flip mirroring of crop
        if setting_name in ["flip_h", "flip_v"]:
            current_settings = self.image_processor.get_current_settings()
            current_crop = current_settings.get("crop")
            if current_crop:
                c_left, c_top, c_right, c_bottom = current_crop
                if setting_name == "flip_h":
                    new_crop = (1.0 - c_right, c_top, 1.0 - c_left, c_bottom)
                else:  # flip_v
                    new_crop = (c_left, 1.0 - c_bottom, c_right, 1.0 - c_top)

                self.image_processor.set_processing_params(crop=new_crop)

                # Update visual overlay if active
                if self.editing_controls.crop_btn.isChecked():
                    scene_rect = self.view.sceneRect()
                    sw, sh = scene_rect.width(), scene_rect.height()
                    nl, nt, nr, nb = new_crop
                    rect = QtCore.QRectF(
                        nl * sw, nt * sh, (nr - nl) * sw, (nb - nt) * sh
                    )
                    self.view.set_crop_rect(rect)

        # Auto-crop on rotation to avoid black parts
        if (
            setting_name == "rotation"
            and self.image_processor.base_img_full is not None
        ):
            current_settings = self.image_processor.get_current_settings()
            rotate_val = current_settings.get("rotation", 0.0)

            # Update visual rotation handle position (if crop mode is active)
            if self.editing_controls.crop_btn.isChecked():
                self.view.set_rotation(rotate_val)

            h, w = self.image_processor.base_img_full.shape[:2]

            # Get current aspect ratio lock
            text = self.editing_controls.aspect_ratio_combo.currentText()
            ratio = None
            if text == "1:1":
                ratio = 1.0
            elif text == "4:3":
                ratio = 4.0 / 3.0
            elif text == "3:2":
                ratio = 3.0 / 2.0
            elif text == "16:9":
                ratio = 16.0 / 9.0

            # Calculate max safe crop
            safe_crop = pynegative.calculate_max_safe_crop(
                w, h, rotate_val, aspect_ratio=ratio
            )

            # Update safe bounds in view
            import math

            phi = abs(math.radians(rotate_val))
            W = w * math.cos(phi) + h * math.sin(phi)
            H = w * math.sin(phi) + h * math.cos(phi)

            c_left, c_top, c_right, c_bottom = safe_crop
            safe_rect = QtCore.QRectF(
                c_left * W, c_top * H, (c_right - c_left) * W, (c_bottom - c_top) * H
            )
            self.view.set_crop_safe_bounds(safe_rect)

            # If in crop mode, update visual overlay ONLY
            if self.editing_controls.crop_btn.isChecked():
                self.view.set_crop_rect(safe_rect)
                # Ensure the processor DOES NOT have a crop applied so user can see context
                self.image_processor.set_processing_params(crop=None)
            else:
                # Apply safe crop to processor if NOT in crop mode
                self.image_processor.set_processing_params(crop=safe_crop)

        self._request_update_from_view()
        self.save_timer.start(1000)  # Save after 1 second of inactivity

        # Schedule undo state
        current_settings = self.image_processor.get_current_settings()
        self.settings_manager.schedule_undo_state(
            f"Adjust {setting_name}", current_settings
        )

    def _on_rating_changed(self, rating):
        """Handle rating change."""
        self.settings_manager.set_current_settings(
            self.image_processor.get_current_settings(), rating
        )
        self.save_timer.start(500)
        if self.raw_path:
            self.ratingChanged.emit(str(self.raw_path), rating)
            self.settings_manager.push_immediate_undo_state(
                f"Rating changed to {rating} star{'s' if rating != 1 else ''}",
                self.image_processor.get_current_settings(),
            )

    def _on_auto_wb_requested(self):
        """Handle auto white balance request."""
        if self.image_processor.base_img_preview is None:
            return

        # Calculate Auto WB using current preview
        wb_settings = pynegative.calculate_auto_wb(
            self.image_processor.base_img_preview
        )

        # Apply to UI
        self.editing_controls.set_slider_value(
            "val_temperature", wb_settings["temperature"]
        )
        self.editing_controls.set_slider_value("val_tint", wb_settings["tint"])

        # Update processor
        self.image_processor.set_processing_params(**wb_settings)
        self._request_update_from_view()
        self.save_timer.start(1000)

        # Push undo state
        self.settings_manager.push_immediate_undo_state(
            "Auto White Balance", self.image_processor.get_current_settings()
        )

    def _on_preview_rating_changed(self, rating):
        """Handle rating change from preview widget."""
        self.editing_controls.set_rating(rating)
        self._on_rating_changed(rating)

    def _on_preset_applied(self, preset_type):
        """Handle preset application."""
        self.image_processor.set_processing_params(
            sharpen_value=self.editing_controls.val_sharpen_value,
            sharpen_radius=self.editing_controls.val_sharpen_radius,
            sharpen_percent=self.editing_controls.val_sharpen_percent,
            denoise_luma=self.editing_controls.val_denoise_luma,
            denoise_chroma=self.editing_controls.val_denoise_chroma,
        )
        self._request_update_from_view()

        # Push undo state for preset application
        self.settings_manager.push_immediate_undo_state(
            f"Apply {preset_type} preset", self.image_processor.get_current_settings()
        )

    def _cycle_denoise_mode(self):
        """Cycle denoise mode and show toast."""
        new_method = self.editing_controls.cycle_denoise_method()
        self.show_toast(f"Denoise: {new_method}")
        # Core debug log will already show this because settingChanged triggers a re-render
        # and de_noise_image has a logger.debug call.

    def _on_raw_loaded(self, path, img_arr, settings):
        """Handle raw image loading completion."""
        if Path(path) != self.raw_path:
            return  # User switched images already

        if img_arr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load image")
            return

        # 1. Set the image first (this clears existing params in the processor)
        self.image_processor.set_image(img_arr)

        # 2. Reset UI to defaults before applying new settings
        self.editing_controls.reset_sliders(silent=True)
        self.editing_controls.set_rating(0)
        self.preview_rating_widget.set_rating(0)

        # 3. Apply loaded settings if they exist
        if settings:
            # Set rating UI
            rating = settings.get("rating", 0)
            self.editing_controls.set_rating(rating)
            self.preview_rating_widget.set_rating(rating)
            self.settings_manager.set_current_settings(
                self.image_processor.get_current_settings(), rating
            )

            # Apply settings to editing controls UI (silent to avoid clearing crop)
            self.editing_controls.apply_settings(settings)

            # Update image processor params with loaded settings
            # We combine UI-mapped settings and direct sidecar settings
            all_params = self.editing_controls.get_all_settings()
            loaded_crop = settings.get("crop")
            rotate_val = settings.get("rotation", 0.0)

            # If rotation is present but no manual crop exists, apply safe crop
            if loaded_crop is None and abs(rotate_val) > 0.1:
                h, w = img_arr.shape[:2]
                loaded_crop = pynegative.calculate_max_safe_crop(w, h, rotate_val)

            all_params["crop"] = loaded_crop
            self.image_processor.set_processing_params(**all_params)

        # 3. Trigger a single unified update
        self._request_update_from_view()

        # 4. Request a fit once the UI settles
        QtCore.QTimer.singleShot(50, self.view.reset_zoom)
        QtCore.QTimer.singleShot(200, self.view.reset_zoom)

        # 5. Update comparison overlay if enabled
        if self._comparison_enabled and self.comparison_overlay:
            self.comparison_overlay.setUneditedPixmap(
                self.image_processor.get_unedited_pixmap()
            )

    def _request_update_from_view(self):
        """Request image update from current view state."""
        if (
            self.view is not None
            and hasattr(self, "image_processor")
            and hasattr(self.image_processor, "base_img_full")
            and self.image_processor.base_img_full is not None
        ):
            self.image_processor.request_update()

    def _auto_save_sidecar(self):
        """Auto-save settings to sidecar."""
        if not self.raw_path:
            return

        settings = self.image_processor.get_current_settings()

        # If we are in crop mode, the processor has crop=None to show full image.
        # We MUST save the intended crop from the visual tool so it persists.
        if self.editing_controls.crop_btn.isChecked():
            rect = self.view.get_crop_rect()
            scene_rect = self.view.sceneRect()
            w, h = scene_rect.width(), scene_rect.height()

            if w > 0 and h > 0:
                c_left = rect.left() / w
                c_top = rect.top() / h
                c_right = rect.right() / w
                c_bottom = rect.bottom() / h

                # Clamp
                c_left = max(0.0, min(1.0, c_left))
                c_top = max(0.0, min(1.0, c_top))
                c_right = max(0.0, min(1.0, c_right))
                c_bottom = max(0.0, min(1.0, c_bottom))

                settings["crop"] = (c_left, c_top, c_right, c_bottom)

        self.settings_manager.auto_save_sidecar(
            self.raw_path, settings, self.editing_controls.star_rating_widget.rating()
        )

    def _on_carousel_image_selected(self, path):
        """Handle carousel image selection."""
        # Avoid reloading if same image
        if Path(path) != self.raw_path:
            self.load_image(path)

    def _on_carousel_selection_changed(self, selected_paths):
        """Handle carousel selection changes."""
        pass  # Currently handled by carousel manager

    def _on_carousel_keyboard_navigation(self, selected_paths):
        """Handle carousel navigation from keyboard shortcuts."""
        current_path = self.carousel_manager.get_current_path()
        if current_path and current_path != self.raw_path:
            self.load_image(str(current_path))

    def _on_settings_copied(self, source_path, settings):
        """Handle settings copied event."""
        pass  # Currently handled by settings manager

    def _on_settings_pasted(self, target_paths, settings):
        """Handle settings pasted event."""
        pass  # Currently handled by settings manager

    def _on_undo_state_changed(self):
        """Handle undo/redo state changes."""
        pass  # Can be used for UI updates if needed

    # --- Context menus and shortcuts ---

    def _show_main_photo_context_menu(self, pos):
        """Show context menu for main photo view."""
        if not self.raw_path:
            return

        menu = QtWidgets.QMenu(self)

        copy_action = menu.addAction("Copy Settings")
        copy_action.triggered.connect(
            lambda: self.settings_manager.copy_settings_from_current(
                self.image_processor.get_current_settings()
            )
        )
        copy_action.setShortcut(QtGui.QKeySequence.StandardKey.Copy)

        paste_action = menu.addAction("Paste Settings")
        paste_action.triggered.connect(
            lambda: self.settings_manager.paste_settings_to_current(
                self.editing_controls.apply_settings
            )
        )
        paste_action.setEnabled(self.settings_manager.has_clipboard_content())
        paste_action.setShortcut(QtGui.QKeySequence.StandardKey.Paste)

        menu.exec_(self.view.mapToGlobal(pos))

    def _handle_carousel_context_menu(self, context_type, data):
        """Handle carousel context menu request."""
        if context_type == "carousel":
            pos, item_path, carousel_widget = data

            menu = QtWidgets.QMenu(self)
            selected_paths = carousel_widget.get_selected_paths()

            if item_path in selected_paths:
                # Item is selected - can copy from selection
                copy_action = menu.addAction("Copy Settings from Selected")
                copy_action.triggered.connect(
                    lambda: self.settings_manager.copy_settings_from_path(
                        Path(selected_paths[0]) if selected_paths else Path(item_path)
                    )
                )
                copy_action.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
            else:
                # Item is not selected - can copy from this specific item
                copy_action = menu.addAction(
                    f"Copy Settings from {Path(item_path).name}"
                )
                copy_action.triggered.connect(
                    lambda: self.settings_manager.copy_settings_from_path(item_path)
                )

            # Paste option
            paste_action = menu.addAction("Paste Settings to Selected")
            paste_action.triggered.connect(
                lambda: self.settings_manager.paste_settings_to_selected(
                    selected_paths, self.editing_controls.apply_settings
                )
            )
            paste_action.setEnabled(
                self.settings_manager.has_clipboard_content()
                and len(selected_paths) > 0
            )
            paste_action.setShortcut(QtGui.QKeySequence.StandardKey.Paste)

            menu.addSeparator()

            select_all_action = menu.addAction("Select All")
            select_all_action.triggered.connect(carousel_widget.select_all_items)
            select_all_action.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)

            menu.exec_(carousel_widget.mapToGlobal(pos))

    def _handle_copy_shortcut(self):
        """Handle copy shortcut."""
        selected_paths = self.carousel_manager.get_selected_paths()
        if len(selected_paths) > 0:
            if self.raw_path and str(self.raw_path) in selected_paths:
                self.settings_manager.copy_settings_from_current(
                    self.image_processor.get_current_settings()
                )
            else:
                self.settings_manager.copy_settings_from_path(Path(selected_paths[0]))
        else:
            self.settings_manager.copy_settings_from_current(
                self.image_processor.get_current_settings()
            )

    def _handle_paste_shortcut(self):
        """Handle paste shortcut."""
        selected_paths = self.carousel_manager.get_selected_paths()
        if len(selected_paths) > 0:
            self.settings_manager.paste_settings_to_selected(
                selected_paths, self.editing_controls.apply_settings
            )
        else:
            self.settings_manager.paste_settings_to_current(
                self.editing_controls.apply_settings
            )

    def _undo(self):
        """Handle undo action."""
        state = self.settings_manager.undo()
        if state:
            self._restore_state(state)

    def _redo(self):
        """Handle redo action."""
        state = self.settings_manager.redo()
        if state:
            self._restore_state(state)

    def _restore_state(self, state):
        """Restore editor state from undo/redo state."""
        settings = state["settings"]
        rating = state["rating"]

        # Apply all settings
        self.editing_controls.apply_settings(settings)
        self.image_processor.set_processing_params(**settings)
        self._request_update_from_view()

        # Restore rating
        self.editing_controls.set_rating(rating)
        self.settings_manager.set_current_settings(settings, rating)

    @QtCore.Slot(float)
    def _on_performance_measured(self, elapsed_ms):
        """Update the performance label."""
        self.perf_label.setText(f"{elapsed_ms:.1f} ms")

    def _toggle_performance_overlay(self):
        """Toggle the visibility of the performance metric overlay."""
        is_visible = not self.perf_label.isVisible()
        self.perf_label.setVisible(is_visible)
        self.show_toast(f"Performance Overlay {'On' if is_visible else 'Off'}")

    def _set_rating_shortcut(self, key):
        """Set rating from keyboard shortcut (1-5, 0)."""
        if key == Qt.Key_0:
            rating = 0
        else:
            rating = key.value - Qt.Key_0.value
        self.preview_rating_widget.set_rating(rating)
        self.editing_controls.set_rating(rating)
        self._on_rating_changed(rating)

    def _set_rating_by_number(self, rating):
        """Set rating by number (called from keyboard shortcuts)."""
        self.preview_rating_widget.set_rating(rating)
        self.editing_controls.set_rating(rating)
        self._on_rating_changed(rating)

    def _navigate_previous(self):
        """Navigate to previous image in carousel."""
        if not self.isVisible():
            return
        self.carousel_manager.select_previous()

    def _navigate_next(self):
        """Navigate to next image in carousel."""
        if not self.isVisible():
            return
        self.carousel_manager.select_next()

    def _on_crop_toggled(self, enabled):
        self.view.set_crop_mode(enabled)

        current_settings = self.image_processor.get_current_settings()

        if enabled:
            # Enter Crop Mode: Show full image (uncropped) with overlay
            current_crop = current_settings.get("crop")
            rotate_val = current_settings.get("rotation", 0.0)

            # Update rotation handle visual state
            self.view.set_rotation(rotate_val)

            # Calculate the dimensions of the FULL rotated image (uncropped)
            # We need these to correctly map the normalized crop coordinates to the scene.
            if self.image_processor.base_img_full is not None:
                h, w = self.image_processor.base_img_full.shape[:2]

                import math

                phi = abs(math.radians(rotate_val))
                W = w * math.cos(phi) + h * math.sin(phi)
                H = w * math.sin(phi) + h * math.cos(phi)

                # Map normalized crop to the FULL rotated scene dimensions
                if current_crop:
                    c_left, c_top, c_right, c_bottom = current_crop
                    rect = QtCore.QRectF(
                        c_left * W,
                        c_top * H,
                        (c_right - c_left) * W,
                        (c_bottom - c_top) * H,
                    )
                    self.view.set_crop_rect(rect)
                else:
                    # Default to full image if no crop exists
                    # Note: When first entering, sceneRect might be the original image size,
                    # but it will soon be updated to W, H by the processor.
                    # We use W, H here for consistency.
                    self.view.set_crop_rect(QtCore.QRectF(0, 0, W, H))

                # Set safe bounds for Crop Tool based on rotation
                # Get current aspect ratio lock
                text = self.editing_controls.aspect_ratio_combo.currentText()
                ratio = None
                if text == "1:1":
                    ratio = 1.0
                elif text == "4:3":
                    ratio = 4.0 / 3.0
                elif text == "3:2":
                    ratio = 3.0 / 2.0
                elif text == "16:9":
                    ratio = 16.0 / 9.0

                safe_crop = pynegative.calculate_max_safe_crop(
                    w, h, rotate_val, aspect_ratio=ratio
                )

                c_safe_l, c_safe_t, c_safe_r, c_safe_b = safe_crop
                safe_rect = QtCore.QRectF(
                    c_safe_l * W,
                    c_safe_t * H,
                    (c_safe_r - c_safe_l) * W,
                    (c_safe_b - c_safe_t) * H,
                )
                self.view.set_crop_safe_bounds(safe_rect)
            else:
                # Fallback if image not loaded
                if current_crop:
                    scene_rect = self.view.sceneRect()
                    sw, sh = scene_rect.width(), scene_rect.height()
                    if sw > 0 and sh > 0:
                        c_left, c_top, c_right, c_bottom = current_crop
                        rect = QtCore.QRectF(
                            c_left * sw,
                            c_top * sh,
                            (c_right - c_left) * sw,
                            (c_bottom - c_top) * sh,
                        )
                        self.view.set_crop_rect(rect)

            # Disable crop in pipeline temporarily to show full context
            self.image_processor.set_processing_params(crop=None)
            self._request_update_from_view()
            self.show_toast("Crop Mode Active: Drag to crop")

            # Center and fit the crop tool
            QtCore.QTimer.singleShot(100, self.view.fit_crop_in_view)

        else:
            # Exit Crop Mode: Apply crop
            rect = self.view.get_crop_rect()
            scene_rect = self.view.sceneRect()
            w, h = scene_rect.width(), scene_rect.height()

            c_val = None
            if w > 0 and h > 0:
                c_left = rect.left() / w
                c_top = rect.top() / h
                c_right = rect.right() / w
                c_bottom = rect.bottom() / h

                # Clamp
                c_left = max(0.0, min(1.0, c_left))
                c_top = max(0.0, min(1.0, c_top))
                c_right = max(0.0, min(1.0, c_right))
                c_bottom = max(0.0, min(1.0, c_bottom))

                # If covers whole image (within 0.5% tolerance), set to None
                # But if user explicitly cropped, we want it to apply.
                # Logic: If it's NOT covering everything, c_val = (l, t, r, b)
                # If it IS covering everything, c_val = None
                if (
                    c_left > 0.005
                    or c_top > 0.005
                    or c_right < 0.995
                    or c_bottom < 0.995
                ):
                    c_val = (c_left, c_top, c_right, c_bottom)
                else:
                    pass  # Crop covers full image, keep c_val as None

            self.image_processor.set_processing_params(crop=c_val)
            self._request_update_from_view()
            self.show_toast("Crop Applied")
            self._auto_save_sidecar()

    def _on_rotation_handle_changed(self, angle: float):
        """Handle rotation change from crop tool handles (throttled)."""
        # Store pending rotation
        self._pending_rotation_from_handle = angle

        # Update slider display immediately (visual feedback)
        self.editing_controls.set_slider_value("rotation", angle, silent=True)

        # Start throttle timer for processor update
        if not self._rotation_handle_throttle_timer.isActive():
            self._rotation_handle_throttle_timer.start()

    def _apply_pending_rotation(self):
        """Apply the pending rotation value to processor."""
        if self._pending_rotation_from_handle is None:
            return

        angle = self._pending_rotation_from_handle

        # Update processor (this triggers image re-render)
        self.image_processor.set_processing_params(rotation=angle)
        self._request_update_from_view()
        self.save_timer.start(1000)

        # Schedule undo state
        current_settings = self.image_processor.get_current_settings()
        self.settings_manager.schedule_undo_state(
            f"Rotate to {angle:.1f}°", current_settings
        )

        self._pending_rotation_from_handle = None
