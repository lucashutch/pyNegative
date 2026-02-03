from pathlib import Path
import numpy as np
from PIL import Image
from functools import partial
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt

from .loaders import RawLoader
from .widgets import ZoomControls, ToastWidget, ZoomableGraphicsView, StarRatingWidget
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

        # Auto-save timer
        self.save_timer = QtCore.QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._auto_save_sidecar)

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

        # View replaced with ZoomableGraphicsView
        self.view = ZoomableGraphicsView()

        # Layout for canvas + zoom controls + carousel
        self.canvas_container = QtWidgets.QGridLayout(self.canvas_frame)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.canvas_container.addWidget(self.view, 0, 0)

        # Zoom Controls (Bottom Right overlay)
        self.zoom_ctrl = ZoomControls()
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

        self.canvas_container.addWidget(
            self.zoom_ctrl, 0, 0, Qt.AlignBottom | Qt.AlignRight
        )
        self.zoom_ctrl.setContentsMargins(0, 0, 20, 20)

        # Add carousel from carousel manager
        self.carousel_widget = self.carousel_manager.get_widget()
        self.canvas_container.addWidget(self.carousel_widget, 1, 0)
        self.carousel_widget.installEventFilter(self)

        # Toast widget for notifications
        self.toast = ToastWidget(self.canvas_frame)

        # Performance metric label
        self.perf_label = QtWidgets.QLabel(self.canvas_frame)
        self.perf_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 128); color: white; padding: 4px; border-radius: 4px;"
        )
        self.canvas_container.addWidget(
            self.perf_label, 0, 0, Qt.AlignBottom | Qt.AlignLeft
        )
        self.perf_label.setContentsMargins(10, 0, 0, 10)
        self.perf_label.hide()

        # Preview Rating Widget (Bottom Left overlay)
        self.preview_rating_widget = PreviewStarRatingWidget(self.canvas_frame)
        self.preview_rating_widget.setObjectName("PreviewRatingWidget")
        self.preview_rating_widget.setStyleSheet("""
            QWidget#PreviewRatingWidget {
                background-color: rgba(0, 0, 0, 0.5);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self.canvas_container.addWidget(
            self.preview_rating_widget, 0, 0, Qt.AlignBottom | Qt.AlignLeft
        )
        self.preview_rating_widget.setContentsMargins(20, 0, 0, 20)
        self.preview_rating_widget.hide()

    def _setup_connections(self):
        """Setup signal/slot connections between components."""
        # Editing controls -> Image processor
        self.editing_controls.settingChanged.connect(self._on_setting_changed)
        self.editing_controls.ratingChanged.connect(self._on_rating_changed)
        self.editing_controls.autoWbRequested.connect(self._on_auto_wb_requested)
        self.editing_controls.saveRequested.connect(self.save_file)
        self.editing_controls.presetApplied.connect(self._on_preset_applied)

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
        # Auto-fit image if in fitting mode
        if (
            hasattr(self, "image_processor")
            and self.image_processor.base_img_full is not None
        ):
            if getattr(self.view, "_is_fitting", False):
                self.view.reset_zoom()
        super().resizeEvent(event)

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
        self.editing_controls.set_save_enabled(False)
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

        QtWidgets.QApplication.processEvents()

        self.editing_controls.set_save_enabled(False)  # Disable save until full load

        loader = RawLoader(path)
        loader.signals.finished.connect(self._on_raw_loaded)
        self.thread_pool.start(loader)

    def save_file(self):
        """Save the current image."""
        if not self.image_processor.base_img_full:
            return

        input_dir = self.raw_path.parent
        default_name = self.raw_path.with_suffix(".jpg").name
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window(),
            "Save",
            str(input_dir / default_name),
            "JPEG (*.jpg);;HEIF (*.heic)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )

        if path:
            path = Path(path)
            try:
                # RELOAD FULL RESOLUTION FOR SAVING
                self.show_toast("Processing Full Resolution...")
                QtWidgets.QApplication.processEvents()

                # Explicitly request full size (half_size=False)
                full_img = pynegative.open_raw(self.raw_path, half_size=False)

                # Get current settings from image processor
                settings = self.image_processor.get_current_settings()

                # Process full resolution with current settings
                img, _ = pynegative.apply_tone_map(
                    full_img,
                    temperature=settings.get("temperature", 0.0),
                    tint=settings.get("tint", 0.0),
                    exposure=settings["exposure"],
                    contrast=settings["contrast"],
                    blacks=settings["blacks"],
                    whites=settings["whites"],
                    shadows=settings["shadows"],
                    highlights=settings["highlights"],
                    saturation=settings["saturation"],
                )
                pil_img = Image.fromarray((img * 255).astype(np.uint8))

                if settings["sharpen_value"] > 0:
                    pil_img = pynegative.sharpen_image(
                        pil_img,
                        settings["sharpen_radius"],
                        settings["sharpen_percent"],
                        method="High Quality",
                    )

                if settings["de_noise"] > 0:
                    pil_img = pynegative.de_noise_image(
                        pil_img, settings["de_noise"], method="High Quality"
                    )

                pynegative.save_image(pil_img, path)

                QtWidgets.QMessageBox.information(
                    self, "Saved", f"Saved full resolution to {path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
                self.show_toast("Error saving")

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
            de_noise=self.editing_controls.val_de_noise,
        )
        self._request_update_from_view()

        # Push undo state for preset application
        self.settings_manager.push_immediate_undo_state(
            f"Apply {preset_type} preset", self.image_processor.get_current_settings()
        )

    def _on_raw_loaded(self, path, img_arr, settings):
        """Handle raw image loading completion."""
        if Path(path) != self.raw_path:
            return  # User switched images already

        if img_arr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load image")
            return

        # Apply loaded settings
        if settings:
            # Set rating
            rating = settings.get("rating", 0)
            self.editing_controls.set_rating(rating)
            self.preview_rating_widget.set_rating(rating)
            self.settings_manager.set_current_settings(
                self.image_processor.get_current_settings(), rating
            )

            # Apply settings to editing controls
            self.editing_controls.apply_settings(settings)

            # Apply settings to image processor
            self.image_processor.set_processing_params(
                temperature=settings.get("temperature", 0.0),
                tint=settings.get("tint", 0.0),
                exposure=settings.get("exposure", 0.0),
                contrast=settings.get("contrast", 1.0),
                whites=settings.get("whites", 1.0),
                blacks=settings.get("blacks", 0.0),
                highlights=settings.get("highlights", 0.0),
                shadows=settings.get("shadows", 0.0),
                saturation=settings.get("saturation", 1.0),
            )

            # Handle sharpening value
            sharpen_val = settings.get("sharpen_value")
            if sharpen_val is not None:
                self.editing_controls.set_slider_value("val_sharpen", sharpen_val)
            else:
                # Compatibility: try to infer from radius/percent
                radius = settings.get("sharpen_radius", 2.0)
                # Reverse mapping: s = (radius - 0.5) / 0.75 * 100
                inferred_val = max(0, min(100, (radius - 0.5) / 0.75 * 100))
                self.editing_controls.set_slider_value("val_sharpen", inferred_val)

            self.editing_controls.set_slider_value(
                "val_denoise", settings.get("de_noise", 0)
            )

            # Update image processor with all settings
            self.editing_controls._apply_preset("")  # Trigger the sharpening mapping
            self.image_processor.set_processing_params(
                **self.editing_controls.get_all_settings()
            )

        # Set image and trigger update
        self.image_processor.set_image(img_arr)
        self.editing_controls.set_save_enabled(True)
        self._request_update_from_view()

        # Request a fit once the UI settles
        QtCore.QTimer.singleShot(50, self.view.reset_zoom)
        QtCore.QTimer.singleShot(200, self.view.reset_zoom)

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
