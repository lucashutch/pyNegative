from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from .widgets import ComboBox, ToastWidget
from .exportgallerymanager import ExportGalleryManager
from .exportsettingsmanager import ExportSettingsManager
from .exportprocessor import ExportJob


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

        # Toast for export completion (longer duration: 8 seconds)
        self.export_toast = ToastWidget(self, duration=8000)

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
        self.settings_layout.addWidget(self.heif_settings)

        # DNG settings
        self.dng_settings = QtWidgets.QGroupBox("DNG Settings")
        self.dng_layout = QtWidgets.QFormLayout(self.dng_settings)
        self.dng_compression = QtWidgets.QComboBox()
        self.dng_compression.setObjectName("ExportComboBox")
        self.dng_compression.addItems(["None", "Lossy"])
        self.dng_layout.addRow("Compression", self.dng_compression)
        self.settings_layout.addWidget(self.dng_settings)

    def _setup_size_settings(self):
        """Setup size constraint settings."""
        self.size_group = QtWidgets.QGroupBox("Size")
        self.size_group.setCheckable(True)
        self.size_group.setChecked(False)
        self.size_layout = QtWidgets.QFormLayout(self.size_group)

        self.max_width = QtWidgets.QLineEdit()
        self.max_width.setObjectName("ExportLineEdit")
        self.max_width.setPlaceholderText("Width (px)")
        self.max_width.setValidator(QtGui.QIntValidator(1, 100000))
        self.size_layout.addRow("Max Width", self.max_width)

        self.max_height = QtWidgets.QLineEdit()
        self.max_height.setObjectName("ExportLineEdit")
        self.max_height.setPlaceholderText("Height (px)")
        self.max_height.setValidator(QtGui.QIntValidator(1, 100000))
        self.size_layout.addRow("Max Height", self.max_height)
        self.settings_layout.addWidget(self.size_group)

    def _setup_destination_settings(self):
        """Setup destination folder settings."""
        self.destination_group = QtWidgets.QGroupBox("Destination")
        self.destination_layout = QtWidgets.QVBoxLayout(self.destination_group)

        self.use_default_dest = QtWidgets.QRadioButton(
            "Save in same folder as original"
        )
        self.destination_layout.addWidget(self.use_default_dest)

        self.use_custom_dest = QtWidgets.QRadioButton("Save to custom folder")
        self.destination_layout.addWidget(self.use_custom_dest)

        # Custom destination container
        self.custom_dest_container = QtWidgets.QWidget()
        self.custom_dest_layout = QtWidgets.QHBoxLayout(self.custom_dest_container)
        self.custom_dest_layout.setContentsMargins(20, 0, 0, 0)

        self.custom_dest_button = QtWidgets.QPushButton("Choose...")
        self.custom_dest_layout.addWidget(self.custom_dest_button)

        self.custom_dest_label = QtWidgets.QLabel("No folder selected")
        self.custom_dest_label.setStyleSheet("color: #666;")
        self.custom_dest_label.setWordWrap(True)
        self.custom_dest_layout.addWidget(self.custom_dest_label)
        self.custom_dest_layout.addStretch()

        self.destination_layout.addWidget(self.custom_dest_container)
        self.settings_layout.addWidget(self.destination_group)

    def _connect_components(self):
        """Connect signals between components."""
        # Gallery manager signals
        self.gallery_manager.selectionCountChanged.connect(
            self._on_selection_count_changed
        )

        # Settings manager signals
        self.settings_manager.presetApplied.connect(self._on_preset_applied)
        self.settings_manager.destinationChanged.connect(self._on_destination_changed)

        # Export job signals
        self.export_job.signals.progress.connect(self.progress_bar.setValue)
        self.export_job.signals.batchCompleted.connect(self._on_export_completed)
        self.export_job.signals.error.connect(self._on_export_error)

        # UI control signals
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.save_preset_button.clicked.connect(self._save_preset)
        self.export_button.clicked.connect(self.start_export)
        self.use_default_dest.toggled.connect(self._on_destination_mode_changed)
        self.custom_dest_button.clicked.connect(self._choose_custom_destination)

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
        self.dng_compression.currentTextChanged.connect(
            lambda text: self.settings_manager.update_setting("dng_compression", text)
        )
        self.max_width.textChanged.connect(
            lambda text: self.settings_manager.update_setting("max_width", text)
        )
        self.max_height.textChanged.connect(
            lambda text: self.settings_manager.update_setting("max_height", text)
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

        # Load destination settings
        dest_settings = self.settings_manager.load_destination_settings()
        self.use_default_dest.setChecked(dest_settings["use_default_destination"])

        custom_dest = dest_settings["custom_destination"]
        if custom_dest:
            self.custom_dest_label.setText(custom_dest)
            self.custom_dest_label.setStyleSheet("")

        # Initial format display
        self._on_format_changed(self.format_combo.currentIndex())

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
        self.dng_compression.setCurrentText(settings["dng_compression"])
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
        self.dng_settings.setVisible(format == "DNG")

    def _on_destination_mode_changed(self, use_default):
        """Handle destination mode change."""
        self.custom_dest_button.setEnabled(not use_default)
        self.settings_manager.save_destination_settings(use_default)

    def _choose_custom_destination(self):
        """Open folder dialog for custom destination."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Export Destination"
        )
        if folder:
            self.settings_manager.set_custom_destination(folder)

    def _on_destination_changed(self, folder):
        """Update UI when destination changes."""
        self.custom_dest_label.setText(folder)
        self.custom_dest_label.setStyleSheet("")

    def _on_export_completed(self, success_count, total_count):
        """Handle export batch completion."""
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.export_toast.show_message(
            f"Export finished: {success_count} of {total_count} files exported successfully."
        )

    def _on_export_error(self, error):
        """Handle export error."""
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(
            self, "Export error", f"An error occurred during export:\n{error}"
        )

    # Public API

    def load_folder(self, folder):
        """Load images from a folder."""
        self.current_folder = Path(folder)

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
        use_default = self.use_default_dest.isChecked()
        destination = self.settings_manager.get_destination(
            use_default=use_default, gallery_folder=self.current_folder
        )

        if not destination:
            destination = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Destination Folder"
            )
            if not destination:
                return
            self.settings_manager.set_custom_destination(destination)

        # Update UI
        self.export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start export
        settings = self.settings_manager.get_current_settings()
        self.export_job.start_export(files, settings, destination)

    def get_supported_formats(self):
        """Get list of supported export formats."""
        return self.settings_manager.get_supported_formats()
