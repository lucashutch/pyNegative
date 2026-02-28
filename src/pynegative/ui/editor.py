import logging
import math
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .. import core as pynegative
from .carouselmanager import CarouselManager
from .editingcontrols import EditingControls
from .editor_managers import (
    ComparisonManager,
    ContextMenuManager,
    CropManager,
    FloatingUIManager,
    ShortcutManager,
    VersionManager,
)
from .imageprocessing import ImageProcessingPipeline
from .loaders import RawLoader
from .settingsmanager import SettingsManager
from .widgets import (
    HistoryPanel,
    MetadataPanel,
    RightPanel,
    ZoomableGraphicsView,
    ZoomControls,
    PreviewStarRatingWidget,
    SelectivePasteDialog,
)

logger = logging.getLogger(__name__)


class EditorWidget(QtWidgets.QWidget):
    ratingChanged = QtCore.Signal(str, int)
    imageDoubleClicked = QtCore.Signal()

    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.raw_path = None
        self.settings = QtCore.QSettings("pyNegative", "Editor")

        # Auto-save timer for metadata
        self.save_timer = QtCore.QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._auto_save_sidecar)

        # Debounce timer for UI settings
        self._settings_save_timer = QtCore.QTimer()
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(500)
        self._settings_save_timer.timeout.connect(self._save_ui_settings)

        # Initialize components
        self._init_components()
        self._init_ui()
        self._setup_connections()
        self.shortcut_manager.setup_shortcuts()

    def _init_components(self):
        """Initialize all component instances."""
        self.image_processor = ImageProcessingPipeline(self.thread_pool, self)
        self.editing_controls = EditingControls(self)
        self.settings_manager = SettingsManager(self)
        self.carousel_manager = CarouselManager(self.thread_pool, self)

        self.comparison_manager = ComparisonManager(self)
        self.crop_manager = CropManager(self)
        self.floating_ui_manager = FloatingUIManager(self)
        self.shortcut_manager = ShortcutManager(self)
        self.context_menu_manager = ContextMenuManager(self)
        self.version_manager = VersionManager(self)

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- Left Panel ---
        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("EditorPanel")
        self.panel.setMinimumWidth(320)
        self.panel.setMaximumWidth(600)
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(8, 10, 8, 10)
        self.panel_layout.setSpacing(2)
        self.splitter.addWidget(self.panel)
        self.panel_layout.addWidget(self.editing_controls)

        # --- Canvas ---
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setObjectName("CanvasFrame")
        self.canvas_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.splitter.addWidget(self.canvas_frame)
        self.splitter.setSizes([360, 1000])

        self.canvas_container = QtWidgets.QVBoxLayout(self.canvas_frame)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.canvas_container.setSpacing(0)

        self.main_content_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.main_content_splitter.setHandleWidth(4)
        self.canvas_container.addWidget(self.main_content_splitter)

        self.inner_canvas_container = QtWidgets.QWidget()
        self.inner_canvas_layout = QtWidgets.QVBoxLayout(self.inner_canvas_container)
        self.inner_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.inner_canvas_layout.setSpacing(0)

        self.canvas_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.inner_canvas_layout.addWidget(self.canvas_splitter)
        self.main_content_splitter.addWidget(self.inner_canvas_container)

        self.view = ZoomableGraphicsView()
        self.canvas_splitter.addWidget(self.view)

        self.carousel_widget = self.carousel_manager.get_widget()
        self.canvas_splitter.addWidget(self.carousel_widget)

        self.metadata_panel = MetadataPanel()
        self.history_panel = HistoryPanel()
        self.right_panel = RightPanel(self.metadata_panel, self.history_panel)
        self.main_content_splitter.addWidget(self.right_panel)
        self.main_content_splitter.setSizes([1000, 280])

        self._metadata_panel_visible = self.settings.value(
            "metadata_panel_visible", False, type=bool
        )
        self._history_panel_visible = self.settings.value(
            "history_panel_visible", False, type=bool
        )
        if self._metadata_panel_visible:
            self.right_panel.show_info_tab()
        elif self._history_panel_visible:
            self.right_panel.show_history_tab()
        self.carousel_widget.installEventFilter(self)

        self.canvas_splitter.setStretchFactor(0, 5)
        self.canvas_splitter.setStretchFactor(1, 0)
        self.canvas_splitter.setHandleWidth(4)
        self.carousel_widget.setMinimumHeight(100)
        self.carousel_widget.setMaximumHeight(400)
        self.canvas_splitter.setCollapsible(1, False)
        self.canvas_splitter.splitterMoved.connect(self._on_carousel_splitter_moved)

        # Zoom Controls
        self.zoom_ctrl = ZoomControls()
        self.zoom_ctrl.setParent(self.canvas_frame)
        self.zoom_ctrl.zoomChanged.connect(lambda z: self.view.set_zoom(z, manual=True))
        self.zoom_ctrl.zoomInClicked.connect(self.view.zoom_in)
        self.zoom_ctrl.zoomOutClicked.connect(self.view.zoom_out)
        self.zoom_ctrl.fitClicked.connect(self.view.reset_zoom)
        self.zoom_ctrl.zoomPresetSelected.connect(
            lambda z: self.view.set_zoom(z, manual=True)
        )
        self.view.zoomChanged.connect(self.zoom_ctrl.update_zoom)

        # Preview Rating
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

        # Managers setup UI
        self.comparison_manager.setup_ui(self.canvas_frame, self.view)
        self.floating_ui_manager.setup_ui(self.canvas_frame)

        # Context menus
        self.view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(
            self.context_menu_manager.show_main_photo_context_menu
        )

        self.image_processor.set_view_reference(self.view)
        self.view.zoomChanged.connect(self._request_update_from_view)
        self.view.doubleClicked.connect(self.imageDoubleClicked.emit)

        # Initial positioning
        QtCore.QTimer.singleShot(100, self._reposition_floating_ui)
        QtCore.QTimer.singleShot(150, self._load_carousel_height)

    def _load_carousel_height(self):
        settings = QtCore.QSettings("pyNegative", "Editor")
        height = int(settings.value("carousel_height", 210))
        self.carousel_manager.set_carousel_height(height)
        total_high = self.canvas_frame.height()
        if total_high > 0:
            self.canvas_splitter.setSizes([total_high - height, height])

    def _on_carousel_splitter_moved(self, pos, index):
        if index == 1:
            self.carousel_manager.set_carousel_height(self.carousel_widget.height())
            self._reposition_floating_ui()
            self._settings_save_timer.start()

    def _save_ui_settings(self):
        height = self.carousel_widget.height()
        settings = QtCore.QSettings("pyNegative", "Editor")
        settings.setValue("carousel_height", height)

    def _setup_connections(self):
        self.editing_controls.settingChanged.connect(self._on_setting_changed)
        self.editing_controls.cropToggled.connect(self.crop_manager.toggle_crop)
        self.editing_controls.aspectRatioChanged.connect(self.view.set_aspect_ratio)
        self.editing_controls.ratingChanged.connect(self._on_rating_changed)
        self.editing_controls.autoWbRequested.connect(self._on_auto_wb_requested)
        self.editing_controls.presetApplied.connect(self._on_preset_applied)

        self.view.rotationChanged.connect(self.crop_manager.handle_rotation_changed)
        self.view.interactionFinished.connect(
            self.crop_manager.handle_interaction_finished
        )

        self.editing_controls.histogram_section.expandedChanged.connect(
            self.image_processor.set_histogram_enabled
        )
        self.image_processor.histogramUpdated.connect(
            self.editing_controls.histogram_widget.set_data
        )
        self.image_processor.previewUpdated.connect(self.view.set_pixmaps)
        self.image_processor.performanceMeasured.connect(self._on_performance_measured)

        self.image_processor.uneditedPixmapUpdated.connect(
            self._on_unedited_pixmap_updated
        )
        self.image_processor.editedPixmapUpdated.connect(
            lambda p: self.comparison_manager.update_pixmaps(edited=p)
        )

        self.settings_manager.showToast.connect(self.show_toast)
        self.settings_manager.settingsCopied.connect(self._on_settings_copied)
        self.settings_manager.settingsPasted.connect(self._on_settings_pasted)
        self.settings_manager.undoStateChanged.connect(self._on_undo_state_changed)

        self.carousel_manager.imageSelected.connect(self._on_carousel_image_selected)
        self.carousel_manager.selectionChanged.connect(
            self._on_carousel_selection_changed
        )
        self.carousel_manager.selectionChanged.connect(
            self._on_carousel_keyboard_navigation
        )
        self.carousel_manager.contextMenuRequested.connect(
            self.context_menu_manager.handle_carousel_context_menu
        )

        self.preview_rating_widget.ratingChanged.connect(
            self._on_preview_rating_changed
        )

        # History panel signals
        self.history_panel.snapshotSelected.connect(self._on_snapshot_selected)
        self.history_panel.restoreRequested.connect(self._on_snapshot_restore)
        self.history_panel.tagRequested.connect(self._on_snapshot_tag)
        self.history_panel.deleteRequested.connect(self._on_snapshot_delete)
        self.version_manager.snapshotsChanged.connect(self._refresh_history_panel)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if (
            hasattr(self, "image_processor")
            and self.image_processor.base_img_full is not None
        ):
            if self.view._is_fitting:
                self.view.reset_zoom()
        QtCore.QTimer.singleShot(0, self._reposition_floating_ui)

    def _reposition_floating_ui(self):
        self.floating_ui_manager.reposition(
            self.view,
            self.zoom_ctrl,
            self.preview_rating_widget,
            self.comparison_manager,
        )

    def clear(self):
        self.raw_path = None
        self.setWindowTitle("Editor")
        self.editing_controls.reset_sliders()
        self.editing_controls.set_rating(0)
        self.view.reset_zoom()
        self.view.set_pixmaps(QtGui.QPixmap(), 0, 0)
        self.carousel_manager.clear()
        self.settings_manager.clear_clipboard()
        self.metadata_panel.clear()
        self.history_panel.clear()
        self.version_manager.stop()

    def update_rating_for_path(self, path, rating):
        if self.raw_path and str(self.raw_path) == path:
            self.editing_controls.set_rating(rating)
            self.preview_rating_widget.set_rating(rating)

    def load_image(self, path):
        path = Path(path)
        logger.info(f"Image selection changed: {path.name}")

        # Ensure previous image settings are saved before switching
        if self.save_timer.isActive():
            self.save_timer.stop()
            self._auto_save_sidecar()

        self.raw_path = path

        # 1. Clear previous image state, but instantly show cached thumbnail
        self.image_processor.set_image(None)

        # Load and show thumbnail instantly
        try:
            # We skip the memory cache here because it requires mtime and size.
            # load_cached_thumbnail is fast enough (reads from disk cache).
            thumb_img, _ = pynegative.load_cached_thumbnail(str(path), size=400)
            if thumb_img is not None:
                # Convert PIL Image or numpy array to QPixmap
                if hasattr(thumb_img, "mode"):
                    if thumb_img.mode == "RGB":
                        img_data = np.array(thumb_img)
                        h, w, c = img_data.shape
                        qimg = QtGui.QImage(
                            img_data.data, w, h, w * c, QtGui.QImage.Format_RGB888
                        )
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        self.view.set_pixmaps(pixmap, w, h, clear_tiles=True)
                        self.view.reset_zoom()
                elif isinstance(thumb_img, np.ndarray):
                    if thumb_img.dtype == np.float32:
                        img_uint8 = (np.clip(thumb_img, 0, 1) * 255).astype(np.uint8)
                    else:
                        img_uint8 = thumb_img
                    h, w, c = img_uint8.shape
                    qimg = QtGui.QImage(
                        img_uint8.data, w, h, w * c, QtGui.QImage.Format_RGB888
                    )
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    self.view.set_pixmaps(pixmap, w, h, clear_tiles=True)
                    self.view.reset_zoom()
            else:
                self.view.set_pixmaps(None, 0, 0, clear_tiles=True)
        except Exception as e:
            logger.warning(f"Could not load thumbnail for instant preview: {e}")
            self.view.set_pixmaps(None, 0, 0, clear_tiles=True)

        self.metadata_panel.clear()

        if self._metadata_panel_visible:
            # We don't have settings yet in load_image, they come in _on_raw_loaded
            self.metadata_panel.load_for_path(path)

        self.editing_controls.set_crop_checked(False)
        self.view.set_crop_mode(False)

        # Stop version autosave for previous image
        self.version_manager.stop()

        # 2. Start RAW loader
        loader = RawLoader(path)
        loader.signals.finished.connect(self._on_raw_loaded)
        self.thread_pool.start(loader)

    def load_carousel_folder(self, folder):
        self.current_folder = Path(folder)
        self.settings_manager.clear_clipboard()
        self.carousel_manager.load_folder(folder)

    def set_carousel_images(self, image_list, current_path):
        self.carousel_manager.set_images(image_list, Path(current_path))
        self.settings_manager.clear_clipboard()

    def show_toast(self, message):
        self.floating_ui_manager.show_toast(message)

    def set_preview_mode(self, enabled):
        self.panel.setVisible(not enabled)
        self.preview_rating_widget.setVisible(enabled)

    def open(self, path, image_list=None):
        path = Path(path)
        if image_list:
            self.set_carousel_images(image_list, path)
        else:
            self.load_carousel_folder(path.parent)
            self.carousel_manager.select_image(path)

    def _on_setting_changed(self, setting_name, value):
        self.image_processor.set_processing_params(**{setting_name: value})

        if setting_name == "crop" and value is None:
            # If crop is reset to None, and we are in crop mode, reset the visual rect
            if self.editing_controls.geometry_controls.crop_btn.isChecked():
                scene_rect = self.view.sceneRect()
                if not scene_rect.isEmpty():
                    self.view.set_crop_rect(scene_rect)
                    # Also update safe bounds for current rotation
                    rotate_val = self.image_processor.get_current_settings().get(
                        "rotation", 0.0
                    )
                    self.crop_manager.update_safe_bounds(rotate_val)

        if setting_name in ["flip_h", "flip_v"]:
            current_settings = self.image_processor.get_current_settings()
            current_crop = current_settings.get("crop")
            if current_crop:
                c_left, c_top, c_right, c_bottom = current_crop
                new_crop = (
                    (1.0 - c_right, c_top, 1.0 - c_left, c_bottom)
                    if setting_name == "flip_h"
                    else (c_left, 1.0 - c_bottom, c_right, 1.0 - c_top)
                )
                self.image_processor.set_processing_params(crop=new_crop)
                if self.editing_controls.geometry_controls.crop_btn.isChecked():
                    scene_rect = self.view.sceneRect()
                    sw, sh = scene_rect.width(), scene_rect.height()
                    nl, nt, nr, nb = new_crop
                    rect = QtCore.QRectF(
                        nl * sw, nt * sh, (nr - nl) * sw, (nb - nt) * sh
                    )
                    self.view.set_crop_rect(rect)

        if (
            setting_name == "rotation"
            and self.image_processor.base_img_full is not None
        ):
            rotate_val = value
            h, w = self.image_processor.base_img_full.shape[:2]
            text = (
                self.editing_controls.geometry_controls.aspect_ratio_combo.currentText()
            )
            ratio = self.crop_manager._text_to_ratio(text)

            if self.editing_controls.geometry_controls.crop_btn.isChecked():
                self.view.set_rotation(rotate_val)
                # Only update safe bounds immediately on SLIDER move
                self.crop_manager.update_safe_bounds(rotate_val)
            else:
                # If not in crop mode, automatically update to the best fitting safe crop
                safe_crop = pynegative.calculate_max_safe_crop(
                    w, h, rotate_val, aspect_ratio=ratio
                )
                self.image_processor.set_processing_params(crop=safe_crop)
                # Still need to update safe bounds in the view for when we enter crop mode
                phi = abs(math.radians(rotate_val))
                W = w * math.cos(phi) + h * math.sin(phi)
                H = w * math.sin(phi) + math.cos(phi) * h
                c_safe_l, c_safe_t, c_safe_r, c_safe_b = safe_crop
                safe_rect = QtCore.QRectF(
                    c_safe_l * W,
                    c_safe_t * H,
                    (c_safe_r - c_safe_l) * W,
                    (c_safe_b - c_safe_t) * H,
                )
                self.view.set_crop_safe_bounds(safe_rect)

        if (
            setting_name in ["lens_name_override", "lens_camera_override"]
            and self._metadata_panel_visible
        ):
            # Refresh metadata panel to show new override
            current_settings = self.image_processor.get_current_settings()
            self.metadata_panel.load_for_path(self.raw_path, current_settings)

        self._request_update_from_view()
        self.save_timer.start(1000)
        current_settings = self.image_processor.get_current_settings()
        self.settings_manager.schedule_undo_state(
            f"Adjust {setting_name}", current_settings
        )

    def _on_rating_changed(self, rating):
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
        if self.image_processor.base_img_preview is None:
            return
        wb_settings = pynegative.calculate_auto_wb(
            self.image_processor.base_img_preview
        )
        self.editing_controls.set_slider_value(
            "val_temperature", wb_settings["temperature"]
        )
        self.editing_controls.set_slider_value("val_tint", wb_settings["tint"])
        self.image_processor.set_processing_params(**wb_settings)
        self._request_update_from_view()
        self.save_timer.start(1000)
        self.settings_manager.push_immediate_undo_state(
            "Auto White Balance", self.image_processor.get_current_settings()
        )

    def _on_preview_rating_changed(self, rating):
        self.editing_controls.set_rating(rating)
        self._on_rating_changed(rating)

    def _on_preset_applied(self, preset_type):
        self.image_processor.set_processing_params(
            sharpen_value=self.editing_controls.val_sharpen_value,
            sharpen_radius=self.editing_controls.val_sharpen_radius,
            sharpen_percent=self.editing_controls.val_sharpen_percent,
            denoise_luma=self.editing_controls.val_denoise_luma,
            denoise_chroma=self.editing_controls.val_denoise_chroma,
        )
        self._request_update_from_view()
        self.settings_manager.push_immediate_undo_state(
            f"Apply {preset_type} preset", self.image_processor.get_current_settings()
        )

    def _cycle_denoise_mode(self):
        new_method = self.editing_controls.cycle_denoise_method()
        self.show_toast(f"Denoise: {new_method}")

    def _on_raw_loaded(self, path, img_arr, settings):
        if Path(path) != self.raw_path:
            return
        if img_arr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load image")
            return

        # 1. Update processor first
        self.image_processor.set_image(img_arr)
        h, w = img_arr.shape[:2]

        # 2. Apply settings
        self.editing_controls.reset_sliders(silent=True)
        self.editing_controls.set_rating(0)
        self.preview_rating_widget.set_rating(0)
        if settings:
            rating = settings.get("rating", 0)
            self.editing_controls.set_rating(rating)
            self.preview_rating_widget.set_rating(rating)
            self.settings_manager.set_current_settings(
                self.image_processor.get_current_settings(), rating
            )
            self.editing_controls.apply_settings(settings)
            all_params = self.editing_controls.get_all_settings()

            # Update Lens Controls info
            from ..io import lens_resolver

            source, resolved = lens_resolver.resolve_lens_profile(path)
            self.editing_controls.lens_controls.set_lens_info(source, resolved or {})
            self.image_processor.set_lens_info(resolved)

            loaded_crop = settings.get("crop")
            rotate_val = settings.get("rotation", 0.0)
            if loaded_crop is None and abs(rotate_val) > 0.1:
                loaded_crop = pynegative.calculate_max_safe_crop(w, h, rotate_val)
            all_params["crop"] = loaded_crop
            self.image_processor.set_processing_params(**all_params)

        # 3. Explicitly update the view with the correct dimensions *before* resetting zoom.
        # Use existing pixmap (thumbnail) if available, but update the reference dimensions.
        current_pixmap = self.view._bg_item.pixmap()
        self.view.set_pixmaps(current_pixmap, w, h)

        # 4. Reset zoom to get correct viewport math for the new image size
        self.view.reset_zoom()

        # 5. Then request update
        self._request_update_from_view()

        if self.comparison_manager.enabled:
            self.comparison_manager.comparison_overlay.setUneditedPixmap(
                self.image_processor.get_unedited_pixmap(2048)
            )
        if self._metadata_panel_visible:
            self.metadata_panel.load_for_path(self.raw_path, settings)

        # Load history panel if visible
        if self._history_panel_visible:
            self._refresh_history_panel()

        # Start version autosave timer for this image
        self.version_manager.start()

    def _load_metadata(self):
        """Load metadata into the panel (called by MainWindow on toggle)."""
        if self.raw_path:
            current_settings = self.image_processor.get_current_settings()
            self.metadata_panel.load_for_path(self.raw_path, current_settings)

    def _refresh_history_panel(self):
        """Reload the history panel from disk snapshots."""
        snapshots = self.version_manager.load_snapshots()
        self.history_panel.set_snapshots(snapshots)

    # ------------------------------------------------------------------
    # Snapshot preview / restore
    # ------------------------------------------------------------------

    _stashed_settings: dict | None = None
    _stashed_rating: int = 0

    def _on_snapshot_selected(self, snapshot_id: str):
        """Preview a snapshot (or cancel preview on empty id)."""
        if not snapshot_id:
            # Cancel — revert to stashed state
            self._revert_preview()
            return

        snap = self.history_panel.get_snapshot_by_id(snapshot_id)
        if snap is None:
            return

        # Stash current working state on first preview
        if self._stashed_settings is None:
            self._stashed_settings = self.image_processor.get_current_settings()
            self._stashed_rating = self.editing_controls.star_rating_widget.rating()

        # Apply snapshot settings to the editor
        settings = snap["settings"]
        self.editing_controls.apply_settings(settings)
        self.image_processor.set_processing_params(**settings)
        self._request_update_from_view()

        rating = settings.get("rating", 0)
        self.editing_controls.set_rating(rating)

        self.history_panel.set_previewing(True)
        self.show_toast("Previewing snapshot")

    def _on_snapshot_restore(self, snapshot_id: str):
        """Commit the previewed snapshot as the current working state."""
        snap = self.history_panel.get_snapshot_by_id(snapshot_id)
        if snap is None:
            return

        settings = snap["settings"]
        rating = settings.get("rating", 0)

        # Push an undo state so the user can revert
        self.settings_manager.push_immediate_undo_state("Restore snapshot", settings)

        # Persist as the current working sidecar
        self.settings_manager.auto_save_sidecar(self.raw_path, settings, rating)

        # Clear stash — this is now the current state
        self._stashed_settings = None
        self._stashed_rating = 0
        self.history_panel.set_previewing(False)
        self.show_toast("Snapshot restored")

    def _revert_preview(self):
        """Revert the editor to the stashed pre-preview state."""
        if self._stashed_settings is None:
            return
        self.editing_controls.apply_settings(self._stashed_settings)
        self.image_processor.set_processing_params(**self._stashed_settings)
        self._request_update_from_view()
        self.editing_controls.set_rating(self._stashed_rating)
        self._stashed_settings = None
        self._stashed_rating = 0
        self.history_panel.set_previewing(False)

    # ------------------------------------------------------------------
    # Snapshot tagging / deletion
    # ------------------------------------------------------------------

    def _on_snapshot_tag(self, snapshot_id: str):
        """Tag or untag a snapshot (toggle). Prompts for a label if tagging."""
        if not self.raw_path:
            return
        snap = self.history_panel.get_snapshot_by_id(snapshot_id)
        if snap is None:
            return

        if snap.get("is_tagged"):
            # Untag
            pynegative.update_snapshot(
                self.raw_path, snapshot_id, is_tagged=False, label=None
            )
            self.show_toast("Version untagged")
        else:
            # Tag — prompt for a name
            import time

            default_label = pynegative.format_snapshot_timestamp(time.time())
            label, ok = QtWidgets.QInputDialog.getText(
                self, "Tag Version", "Version name:", text=default_label
            )
            if not ok:
                return
            pynegative.update_snapshot(
                self.raw_path,
                snapshot_id,
                is_tagged=True,
                label=label if label else default_label,
            )
            self.show_toast(f"Version tagged: {label or default_label}")

        self._refresh_history_panel()

    def _on_snapshot_delete(self, snapshot_id: str):
        """Delete a snapshot from the history."""
        if not self.raw_path:
            return
        pynegative.delete_snapshot(self.raw_path, snapshot_id)
        self._refresh_history_panel()
        self.show_toast("Snapshot deleted")

    def _request_update_from_view(self):
        if (
            self.view is not None
            and hasattr(self, "image_processor")
            and self.image_processor.base_img_full is not None
        ):
            self.image_processor.request_update()

    def _auto_save_sidecar(self):
        if not self.raw_path:
            return

        # Source of truth for settings is the EditingControls widget
        settings = self.editing_controls.get_all_settings()

        # Add crop from view/processor (special case as it's not a slider)
        if self.editing_controls.geometry_controls.crop_btn.isChecked():
            rect = self.view.get_crop_rect()
            scene_rect = self.view.sceneRect()
            w, h = scene_rect.width(), scene_rect.height()
            if w > 0 and h > 0:
                c_left, c_top, c_right, c_bottom = (
                    max(0.0, min(1.0, rect.left() / w)),
                    max(0.0, min(1.0, rect.top() / h)),
                    max(0.0, min(1.0, rect.right() / w)),
                    max(0.0, min(1.0, rect.bottom() / h)),
                )
                settings["crop"] = (c_left, c_top, c_right, c_bottom)
        else:
            # Preserve existing crop if not currently editing?
            # Actually image_processor has the active crop.
            settings["crop"] = self.image_processor.get_current_settings().get("crop")

        self.settings_manager.auto_save_sidecar(
            self.raw_path, settings, self.editing_controls.star_rating_widget.rating()
        )

    def _on_carousel_image_selected(self, path):
        if Path(path) != self.raw_path:
            self.load_image(path)

    def _on_carousel_selection_changed(self, selected_paths):
        pass

    def _on_carousel_keyboard_navigation(self, selected_paths):
        current_path = self.carousel_manager.get_current_path()
        if current_path and current_path != self.raw_path:
            self.load_image(str(current_path))

    def _on_settings_copied(self, source_path, settings):
        pass

    def _on_settings_pasted(self, target_paths, settings):
        pass

    def _on_undo_state_changed(self):
        pass

    def show_selective_paste_dialog(self):
        """Show the selective paste dialog and apply selected settings."""
        if not self.settings_manager.has_clipboard_content():
            self.show_toast("Clipboard is empty")
            return

        dialog = SelectivePasteDialog(self)
        dialog.setWindowTitle("Paste Settings Selective")
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            selected_keys = dialog.get_selected_keys()
            if not selected_keys:
                return

            selected_paths = self.carousel_manager.get_selected_paths()
            if selected_paths:
                self.settings_manager.paste_settings_to_selected(
                    selected_paths,
                    current_settings_callback=self.editing_controls.apply_settings,
                    keys_to_include=selected_keys,
                )
            else:
                self.settings_manager.paste_settings_to_current(
                    self.editing_controls.apply_settings, keys_to_include=selected_keys
                )
            self._request_update_from_view()

    def show_selective_copy_dialog(self):
        """Show the selective copy dialog and copy selected settings."""
        if not self.raw_path:
            return

        dialog = SelectivePasteDialog(self)
        dialog.setWindowTitle("Copy Settings Selective")
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            selected_keys = dialog.get_selected_keys()
            if not selected_keys:
                return

            self.settings_manager.copy_settings_selective(
                self.image_processor.get_current_settings(), selected_keys
            )

    def _on_unedited_pixmap_updated(self, pixmap):
        self.comparison_manager.update_pixmaps(unedited=pixmap)
        # Fast feedback: if the view is currently empty, show the unedited version immediately
        if self.image_processor.base_img_full is not None:
            h, w = self.image_processor.base_img_full.shape[:2]
            # Only set background if it's currently null (prevents flicker when high-quality comes in)
            if self.view._bg_item.pixmap().isNull():
                self.view.set_pixmaps(pixmap, w, h)
                self.view.reset_zoom()

    @QtCore.Slot(float)
    def _on_performance_measured(self, elapsed_ms):
        self.floating_ui_manager.set_perf_text(f"{elapsed_ms:.1f} ms")

    def _toggle_performance_overlay(self):
        is_visible = self.floating_ui_manager.toggle_perf_visibility()
        self.show_toast(f"Performance Overlay {'On' if is_visible else 'Off'}")

    def _set_rating_by_number(self, rating):
        self.preview_rating_widget.set_rating(rating)
        self.editing_controls.set_rating(rating)
        self._on_rating_changed(rating)

    def _set_rating_shortcut(self, rating):
        """Compatibility alias for tests."""
        self._set_rating_by_number(rating)

    def _navigate_previous(self):
        if self.isVisible():
            self.carousel_manager.select_previous()

    def _navigate_next(self):
        if self.isVisible():
            self.carousel_manager.select_next()
