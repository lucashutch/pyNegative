from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from .widgets import ComboBox, ToastWidget, CollapsibleSection
from .exportgallerymanager import ExportGalleryManager
from .exportsettingsmanager import ExportSettingsManager
from .exportprocessor import ExportJob
from .renamesettingsmanager import RenameSettingsManager
from .renamepreviewdialog import RenamePreviewDialog


class ExportWidget(QtWidgets.QWidget):
    """Main export widget coordinating gallery, settings, and processing."""

    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None

        # Initialize components
        self.gallery_manager = ExportGalleryManager(thread_pool, self)
        self.settings_manager = ExportSettingsManager(self)
        self.export_job = ExportJob(thread_pool, self)
        self.rename_settings_manager = RenameSettingsManager(self)

        # Toast for export completion (longer duration: 8 seconds)
        self.export_toast = ToastWidget(self, duration=8000)

        # Rename preview dialog (created on demand)
        self._rename_preview_dialog = RenamePreviewDialog(self)
        self._pending_rename_mapping = None

        self._init_ui()
        self._connect_components()
        self._load_initial_settings()

    def _init_ui(self):
        """Initialize the user interface."""
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Splitter for resizable panels
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left side: Gallery
        self._setup_gallery_panel()

        # Right side: Settings
        self._setup_settings_panel()

        # Set initial splitter sizes
        self.splitter.setSizes([1000, 340])

    def _setup_gallery_panel(self):
        """Setup the gallery selection panel."""
        gallery_container = QtWidgets.QWidget()
        gallery_layout = QtWidgets.QVBoxLayout(gallery_container)
        gallery_layout.setContentsMargins(10, 10, 10, 10)
        gallery_layout.setSpacing(10)
        self.splitter.addWidget(gallery_container)

        # Add gallery widget from manager
        gallery_layout.addWidget(self.gallery_manager.get_widget())

        # Selection counter
        self.selection_label = QtWidgets.QLabel("0 items selected")
        gallery_layout.addWidget(self.selection_label)

    def _setup_settings_panel(self):
        """Setup the export settings panel."""
        settings_container = QtWidgets.QFrame()
        settings_container.setObjectName("EditorPanel")
        settings_container.setMinimumWidth(320)
        settings_container.setMaximumWidth(600)
        self.splitter.addWidget(settings_container)

        # Scroll area for settings
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        outer_layout = QtWidgets.QVBoxLayout(settings_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Content widget
        content_widget = QtWidgets.QWidget()
        self.settings_layout = QtWidgets.QVBoxLayout(content_widget)
        self.settings_layout.setContentsMargins(10, 10, 10, 10)
        self.settings_layout.setSpacing(10)
        scroll.setWidget(content_widget)

        # Dynamic margin adjustment
        self._setup_scrollbar_margin_adjustment(scroll)

        # Export controls
        self._setup_export_controls()

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.settings_layout.addWidget(self.progress_bar)

    def _setup_scrollbar_margin_adjustment(self, scroll):
        """Setup dynamic margin adjustment for scrollbar."""

        def _update_margins():
            try:
                vbar = scroll.verticalScrollBar()
                is_visible = vbar.isVisible()
                right_margin = 26 if is_visible else 10
                self.settings_layout.setContentsMargins(10, 10, right_margin, 10)
            except Exception:
                pass

        scroll.verticalScrollBar().rangeChanged.connect(_update_margins)
        scroll.verticalScrollBar().valueChanged.connect(_update_margins)
        QtCore.QTimer.singleShot(100, _update_margins)

    def _setup_export_controls(self):
        """Setup export settings controls."""
        # Preset selector
        self.preset_label = QtWidgets.QLabel("Preset")
        self.settings_layout.addWidget(self.preset_label)
        self.preset_combo = ComboBox()
        self.preset_combo.setObjectName("ExportComboBox")
        self.settings_layout.addWidget(self.preset_combo)

        self.save_preset_button = QtWidgets.QPushButton("Save Preset")
        self.save_preset_button.setObjectName("SavePresetButton")
        self.settings_layout.addWidget(self.save_preset_button)

        # Format selector
        self.format_label = QtWidgets.QLabel("Format")
        self.settings_layout.addWidget(self.format_label)
        self.format_combo = ComboBox()
        self.format_combo.setObjectName("ExportComboBox")
        self.format_combo.addItems(self.settings_manager.get_supported_formats())
        self.settings_layout.addWidget(self.format_combo)

        # Format-specific settings
        self._setup_format_settings()

        # Size settings
        self._setup_size_settings()

        # Destination settings
        self._setup_destination_settings()

        # Rename settings
        self._setup_rename_settings()

        # Open folder on complete checkbox
        self.open_folder_checkbox = QtWidgets.QCheckBox("Open folder when complete")
        self.open_folder_checkbox.setChecked(False)
        self.settings_layout.addWidget(self.open_folder_checkbox)

        self.settings_layout.addStretch()

        # Export button
        self.export_button = QtWidgets.QPushButton("Export")
        self.settings_layout.addWidget(self.export_button)

    def _setup_format_settings(self):
        """Setup format-specific setting groups."""
        # JPEG settings
        self.jpeg_settings = QtWidgets.QGroupBox("JPEG Settings")
        self.jpeg_layout = QtWidgets.QFormLayout(self.jpeg_settings)
        self.jpeg_quality = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_layout.addRow("Quality", self.jpeg_quality)
        self.settings_layout.addWidget(self.jpeg_settings)

        # HEIF settings
        self.heif_settings = QtWidgets.QGroupBox("HEIF Settings")
        self.heif_layout = QtWidgets.QFormLayout(self.heif_settings)
        self.heif_quality = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.heif_quality.setRange(1, 100)
        self.heif_layout.addRow("Quality", self.heif_quality)
        self.heif_bit_depth = QtWidgets.QComboBox()
        self.heif_bit_depth.setObjectName("ExportComboBox")
        self.heif_bit_depth.addItems(["8-bit", "10-bit", "12-bit"])
        self.heif_layout.addRow("Bit Depth", self.heif_bit_depth)
        self.settings_layout.addWidget(self.heif_settings)

    def _setup_size_settings(self):
        """Setup size constraint settings in a collapsible section."""
        # Create collapsible section (collapsed by default for clean UI)
        self.size_section = CollapsibleSection("SIZE", expanded=False)

        # Form layout for size settings
        size_form_widget = QtWidgets.QWidget()
        size_form_layout = QtWidgets.QFormLayout(size_form_widget)
        size_form_layout.setSpacing(10)
        size_form_layout.setContentsMargins(0, 0, 0, 0)

        self.max_width = QtWidgets.QLineEdit()
        self.max_width.setObjectName("ExportLineEdit")
        self.max_width.setPlaceholderText("Width (px)")
        self.max_width.setValidator(QtGui.QIntValidator(1, 100000))
        size_form_layout.addRow("Max Width", self.max_width)

        self.max_height = QtWidgets.QLineEdit()
        self.max_height.setObjectName("ExportLineEdit")
        self.max_height.setPlaceholderText("Height (px)")
        self.max_height.setValidator(QtGui.QIntValidator(1, 100000))
        size_form_layout.addRow("Max Height", self.max_height)

        self.size_section.add_widget(size_form_widget)
        self.settings_layout.addWidget(self.size_section)

    def _setup_destination_settings(self):
        """Setup destination folder settings."""
        self.destination_group = QtWidgets.QGroupBox("Destination")
        self.destination_layout = QtWidgets.QVBoxLayout(self.destination_group)

        # Destination path label (shows shortened path)
        self.dest_path_label = QtWidgets.QLabel("No folder loaded")
        self.dest_path_label.setStyleSheet("color: #666;")
        self.dest_path_label.setWordWrap(True)
        self.destination_layout.addWidget(self.dest_path_label)

        # Change button
        self.change_dest_button = QtWidgets.QPushButton("Change...")
        self.change_dest_button.clicked.connect(self._choose_export_destination)
        self.destination_layout.addWidget(self.change_dest_button)

        self.settings_layout.addWidget(self.destination_group)

        # Store the actual export destination path
        self._export_destination = None

    def _setup_rename_settings(self):
        """Setup rename settings controls in a collapsible section."""
        # Create collapsible section (collapsed by default for clean UI)
        self.rename_section = CollapsibleSection("RENAME", expanded=False)

        # Form layout for settings
        rename_form_widget = QtWidgets.QWidget()
        rename_form_layout = QtWidgets.QFormLayout(rename_form_widget)
        rename_form_layout.setSpacing(10)
        rename_form_layout.setContentsMargins(0, 0, 0, 0)

        # Pattern selector
        self.rename_pattern_combo = ComboBox()
        self.rename_pattern_combo.setObjectName("ExportComboBox")
        self.rename_pattern_combo.addItems(
            self.rename_settings_manager.get_pattern_names()
        )
        # Set default to "Prefix + Sequence"
        default_pattern_idx = self.rename_pattern_combo.findText("Prefix + Sequence")
        if default_pattern_idx >= 0:
            self.rename_pattern_combo.setCurrentIndex(default_pattern_idx)
        rename_form_layout.addRow("Pattern", self.rename_pattern_combo)

        # Prefix input
        self.rename_prefix_input = QtWidgets.QLineEdit()
        self.rename_prefix_input.setObjectName("ExportLineEdit")
        self.rename_prefix_input.setPlaceholderText("e.g., Vacation")
        rename_form_layout.addRow("Prefix", self.rename_prefix_input)

        # Start number input
        self.rename_start_seq = QtWidgets.QSpinBox()
        self.rename_start_seq.setObjectName("ExportLineEdit")
        self.rename_start_seq.setRange(1, 9999)
        self.rename_start_seq.setValue(1)
        rename_form_layout.addRow("Start #", self.rename_start_seq)

        # Preview button
        self.rename_preview_button = QtWidgets.QPushButton("Preview Names...")
        self.rename_preview_button.clicked.connect(self._show_rename_preview)
        rename_form_layout.addRow(self.rename_preview_button)

        # Conflict warning label
        self.rename_conflict_label = QtWidgets.QLabel()
        self.rename_conflict_label.setStyleSheet("color: #e74c3c;")
        self.rename_conflict_label.hide()
        rename_form_layout.addRow(self.rename_conflict_label)

        self.rename_section.add_widget(rename_form_widget)

        # Initially disable the form when rename is not enabled
        rename_form_widget.setEnabled(False)
        self.rename_form_widget = rename_form_widget

        self.settings_layout.addWidget(self.rename_section)

        # Connect panel expansion to auto-enable/disable
        self.rename_section.header.clicked.connect(self._on_rename_panel_expanded)

    def _connect_components(self):
        """Connect signals between components."""
        # Gallery manager signals
        self.gallery_manager.selectionCountChanged.connect(
            self._on_selection_count_changed
        )

        # Settings manager signals
        self.settings_manager.presetApplied.connect(self._on_preset_applied)

        # Export job signals
        self.export_job.signals.progress.connect(self.progress_bar.setValue)
        self.export_job.signals.batchCompleted.connect(self._on_export_completed)
        self.export_job.signals.error.connect(self._on_export_error)

        # UI control signals
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.save_preset_button.clicked.connect(self._save_preset)
        self.export_button.clicked.connect(self.start_export)

        # Update settings manager when controls change
        self.format_combo.currentTextChanged.connect(
            lambda text: self.settings_manager.update_setting("format", text)
        )
        self.jpeg_quality.valueChanged.connect(
            lambda val: self.settings_manager.update_setting("jpeg_quality", val)
        )
        self.heif_quality.valueChanged.connect(
            lambda val: self.settings_manager.update_setting("heif_quality", val)
        )
        self.heif_bit_depth.currentTextChanged.connect(
            lambda text: self.settings_manager.update_setting("heif_bit_depth", text)
        )
        self.max_width.textChanged.connect(
            lambda text: self.settings_manager.update_setting("max_width", text)
        )
        self.max_height.textChanged.connect(
            lambda text: self.settings_manager.update_setting("max_height", text)
        )

        # Rename settings connections
        self.rename_pattern_combo.currentTextChanged.connect(
            lambda text: self.rename_settings_manager.update_setting("pattern", text)
        )
        self.rename_prefix_input.textChanged.connect(
            lambda text: self.rename_settings_manager.update_setting("prefix", text)
        )
        self.rename_start_seq.valueChanged.connect(
            lambda val: self.rename_settings_manager.update_setting("start_seq", val)
        )

        # Open folder checkbox connection
        self.open_folder_checkbox.stateChanged.connect(
            lambda state: self.settings_manager.update_setting(
                "open_folder_on_complete", state == QtCore.Qt.CheckState.Checked.value
            )
        )

    def _load_initial_settings(self):
        """Load initial settings and presets."""
        # Load presets
        presets = self.settings_manager.load_presets()
        self.preset_combo.clear()
        self.preset_combo.addItems(presets)

        # Set default to Archival
        archival_index = self.preset_combo.findText("Archival")
        if archival_index != -1:
            self.preset_combo.setCurrentIndex(archival_index)

        # Initial format display
        self._on_format_changed(self.format_combo.currentIndex())

        # Load open folder on complete setting
        current_settings = self.settings_manager.get_current_settings()
        self.open_folder_checkbox.setChecked(
            current_settings.get("open_folder_on_complete", False)
        )

    # Slot handlers

    def _on_selection_count_changed(self, count):
        """Update selection counter label."""
        self.selection_label.setText(f"{count} items selected")

    def _on_preset_applied(self, preset_name):
        """Update UI when preset is applied."""
        settings = self.settings_manager.get_current_settings()

        self.format_combo.setCurrentText(settings["format"])
        self.jpeg_quality.setValue(settings["jpeg_quality"])
        self.heif_quality.setValue(settings["heif_quality"])
        self.heif_bit_depth.setCurrentText(settings["heif_bit_depth"])
        self.max_width.setText(str(settings["max_width"]))
        self.max_height.setText(str(settings["max_height"]))

        self._on_format_changed(self.format_combo.currentIndex())

    def _apply_preset(self, index):
        """Apply selected preset."""
        preset_name = self.preset_combo.itemText(index)
        if preset_name == "Custom":
            return
        self.settings_manager.apply_preset(preset_name)

    def _save_preset(self):
        """Save current settings as a preset."""
        preset_name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Preset", "Preset Name:"
        )
        if ok and preset_name:
            self.settings_manager.save_preset(preset_name)
            self._load_initial_settings()
            self.preset_combo.setCurrentText(preset_name)

    def _on_format_changed(self, index):
        """Show/hide format-specific settings."""
        format = self.format_combo.itemText(index)
        self.jpeg_settings.setVisible(format == "JPEG")
        self.heif_settings.setVisible(format == "HEIF")

    def _format_destination_path(self, path):
        """Format path to show only last 3 components (parent/parent/exported)."""
        if not path:
            return "No folder loaded"

        path = Path(path)
        parts = path.parts

        # Show last 3 parts if path has enough components
        if len(parts) >= 3:
            return f".../{parts[-3]}/{parts[-2]}/{parts[-1]}"
        elif len(parts) >= 2:
            return f".../{parts[-2]}/{parts[-1]}"
        else:
            return str(path)

    def _update_destination_display(self):
        """Update the destination path label."""
        if self._export_destination:
            display_path = self._format_destination_path(self._export_destination)
            self.dest_path_label.setText(display_path)
            self.dest_path_label.setStyleSheet("")
            self.dest_path_label.setToolTip(str(self._export_destination))
        else:
            self.dest_path_label.setText("No folder loaded")
            self.dest_path_label.setStyleSheet("color: #666;")
            self.dest_path_label.setToolTip("")

    def _choose_export_destination(self):
        """Open folder dialog for custom export destination."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Export Destination",
            str(self._export_destination) if self._export_destination else "",
        )
        if folder:
            self._export_destination = Path(folder)
            self._update_destination_display()

    def _open_export_folder(self):
        """Open the export destination folder in system file explorer."""
        if not self._export_destination:
            return

        path = str(self._export_destination)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_export_completed(self, success_count, skipped_count, total_count):
        """Handle export batch completion."""
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        msg = f"Export finished: {success_count} of {total_count} files exported successfully."
        if skipped_count > 0:
            msg += f" ({skipped_count} skipped due to conflicts)"

        # NEW: Open folder if setting is enabled
        if self.settings_manager.get_current_settings().get(
            "open_folder_on_complete", False
        ):
            self._open_export_folder()

        self.export_toast.show_message(msg)

    def _on_export_error(self, error):
        """Handle export error."""
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(
            self, "Export error", f"An error occurred during export:\n{error}"
        )

    def _on_rename_panel_expanded(self, expanded):
        """Auto-enable/disable renaming when panel is expanded/collapsed."""
        self.rename_settings_manager.set_enabled(expanded)
        self.settings_manager.set_rename_enabled(expanded)

        # Enable/disable the form based on panel expansion state
        self.rename_form_widget.setEnabled(expanded)

        # Clear any previous rename mapping when collapsed
        if not expanded:
            self._pending_rename_mapping = None
            self.rename_conflict_label.hide()

    def _show_rename_preview(self):
        """Show the rename preview dialog and return True if confirmed."""
        # Get selected files from gallery
        files = self.gallery_manager.get_selected_paths()
        if not files:
            return False

        # Get current rename settings from UI controls
        pattern_name = self.rename_pattern_combo.currentText()
        prefix = self.rename_prefix_input.text()
        start_seq = self.rename_start_seq.value()
        destination = (
            self._export_destination if self._export_destination else Path.home()
        )
        format_ext = self._get_format_extension()

        # Generate preview data
        files_as_paths = [Path(f) for f in files]
        preview_data = self.rename_settings_manager.generate_preview(
            files_as_paths, pattern_name, prefix, start_seq, destination, format_ext
        )

        # Set up the preview dialog with data
        self._rename_preview_dialog.set_preview_data(preview_data)

        # Show the dialog and wait for user response
        if self._rename_preview_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # User clicked OK, store the rename mapping for use during export
            self._pending_rename_mapping = (
                self._rename_preview_dialog.get_rename_mapping(files_as_paths)
            )
            return True
        else:
            self._pending_rename_mapping = None
            return False

    def _get_format_extension(self):
        """Get the file extension for the current export format."""
        format_name = self.format_combo.currentText()
        return "jpg" if format_name == "JPEG" else "heic"

    # Public API

    def load_folder(self, folder):
        """Load images from a folder."""
        self.current_folder = Path(folder)

        # Set default export destination to <folder>/exported
        self._export_destination = self.current_folder / "exported"
        self._update_destination_display()

        # Get filter settings from main window
        main_window = self.window()
        filter_mode = main_window.filter_combo.currentText()
        filter_rating = main_window.filter_rating_widget.rating()

        self.gallery_manager.load_folder(folder, filter_mode, filter_rating)

    def set_images(self, image_list):
        """Set specific images for export."""
        # Get filter settings from main window
        main_window = self.window()
        filter_mode = main_window.filter_combo.currentText()
        filter_rating = main_window.filter_rating_widget.rating()

        self.gallery_manager.set_images(image_list, filter_mode, filter_rating)

    def apply_filter_from_main(self):
        """Reapply filter when main window filter changes."""
        if self.current_folder:
            self.load_folder(self.current_folder)

    def start_export(self):
        """Start the export process."""
        # Get selected files
        files = self.gallery_manager.get_selected_paths()
        if not files:
            QtWidgets.QMessageBox.warning(
                self, "No files selected", "Please select at least one file to export."
            )
            return

        # Get destination
        if not self._export_destination:
            QtWidgets.QMessageBox.warning(
                self, "No destination set", "Please load a folder first."
            )
            return

        # Create destination directory if it doesn't exist
        self._export_destination.mkdir(exist_ok=True)
        destination = self._export_destination

        # Check if rename is enabled - if so, require preview confirmation
        rename_settings = self.rename_settings_manager.get_current_settings()
        rename_mapping = None

        if rename_settings.get("enabled", False):
            # Validate rename settings first
            errors = self.rename_settings_manager.validate_settings()
            if errors:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Rename Settings", "\n".join(errors)
                )
                return

            # Show preview dialog
            preview_confirmed = self._show_rename_preview()
            if not preview_confirmed:
                return  # User cancelled the preview

            rename_mapping = self._pending_rename_mapping

        # Update UI
        self.export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start export with rename mapping if available
        settings = self.settings_manager.get_current_settings()
        self.export_job.start_export(files, settings, str(destination), rename_mapping)

    def get_supported_formats(self):
        """Get list of supported export formats."""
        return self.settings_manager.get_supported_formats()
