from PySide6 import QtWidgets, QtCore, QtGui
from .widgets import GalleryItemDelegate, GalleryListWidget, ComboBox
from .loaders import ThumbnailLoader
from .. import core as pynegative
from pathlib import Path
import os
from PIL import Image


class ExporterSignals(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int)
    error = QtCore.Signal(str)


class Exporter(QtCore.QRunnable):
    def __init__(self, signals, files, settings, destination_folder):
        super().__init__()
        self.signals = signals
        self.files = files
        self.settings = settings
        self.destination_folder = destination_folder

    def run(self):
        count = len(self.files)
        for i, file in enumerate(self.files):
            try:
                full_img = pynegative.open_raw(str(file), half_size=False)
                sidecar_settings = pynegative.load_sidecar(str(file)) or {}

                img, _ = pynegative.apply_tone_map(
                    full_img,
                    exposure=sidecar_settings.get("exposure", 0.0),
                    contrast=sidecar_settings.get("contrast", 1.0),
                    blacks=sidecar_settings.get("blacks", 0.0),
                    whites=sidecar_settings.get("whites", 1.0),
                    shadows=sidecar_settings.get("shadows", 0.0),
                    highlights=sidecar_settings.get("highlights", 0.0),
                    saturation=sidecar_settings.get("saturation", 1.0),
                )
                pil_img = Image.fromarray((img * 255).astype("uint8"))

                max_w = self.settings.get("max_width")
                max_h = self.settings.get("max_height")
                if max_w and max_h:
                    pil_img.thumbnail((int(max_w), int(max_h)))

                file_name = Path(file).stem
                format = self.settings.get("format")
                if format == "JPEG":
                    dest_path = os.path.join(
                        self.destination_folder, f"{file_name}.jpg"
                    )
                    pil_img.save(
                        dest_path, quality=self.settings.get("jpeg_quality", 90)
                    )
                elif format == "HEIF":
                    dest_path = os.path.join(
                        self.destination_folder, f"{file_name}.heic"
                    )
                    pil_img.save(
                        dest_path,
                        format="HEIF",
                        quality=self.settings.get("heif_quality", 90),
                    )
                elif format == "DNG":
                    # DNG support is not implemented yet
                    pass

                self.signals.progress.emit(int(100 * (i + 1) / count))
            except Exception as e:
                self.signals.error.emit(f"Failed to export {file}: {e}")
                return
        self.signals.finished.emit()


class ExportWidget(QtWidgets.QWidget):
    def __init__(self, thread_pool):
        super().__init__()
        self.thread_pool = thread_pool
        self.current_folder = None
        self.settings = QtCore.QSettings("pyNegative", "Export")
        self._init_ui()

    def _init_ui(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Left side: Gallery
        self.gallery_container = QtWidgets.QWidget()
        self.gallery_layout = QtWidgets.QVBoxLayout(self.gallery_container)
        self.gallery_layout.setContentsMargins(10, 10, 10, 10)
        self.gallery_layout.setSpacing(10)
        self.main_layout.addWidget(self.gallery_container, 3)

        self.list_widget = GalleryListWidget()
        self.list_widget.setObjectName("ExportGrid")
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(180, 180))
        self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setItemDelegate(GalleryItemDelegate(self.list_widget))
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.gallery_layout.addWidget(self.list_widget)

        self.selection_label = QtWidgets.QLabel("0 items selected")
        self.gallery_layout.addWidget(self.selection_label)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)

        # Right side: Settings
        self.settings_container = QtWidgets.QWidget()
        self.settings_container.setFixedWidth(300)
        self.settings_layout = QtWidgets.QVBoxLayout(self.settings_container)
        self.settings_layout.setContentsMargins(10, 10, 10, 10)
        self.settings_layout.setSpacing(10)
        self.main_layout.addWidget(self.settings_container, 1)

        # Format
        self.format_label = QtWidgets.QLabel("Format")
        self.settings_layout.addWidget(self.format_label)
        self.format_combo = ComboBox()
        self.format_combo.setObjectName("ExportComboBox")
        self.format_combo.addItems(["JPEG", "HEIF", "DNG"])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.settings_layout.addWidget(self.format_combo)

        # Preset
        self.preset_label = QtWidgets.QLabel("Preset")
        self.settings_layout.addWidget(self.preset_label)
        self.preset_combo = ComboBox()
        self.preset_combo.setObjectName("ExportComboBox")
        self.settings_layout.addWidget(self.preset_combo)

        self.save_preset_button = QtWidgets.QPushButton("Save Preset")
        self.save_preset_button.setObjectName("SavePresetButton")
        self.save_preset_button.clicked.connect(self.save_preset)
        self.settings_layout.addWidget(self.save_preset_button)

        self.preset_combo.currentIndexChanged.connect(self.apply_preset)

        self.load_presets()

        # Settings
        self.jpeg_settings = QtWidgets.QGroupBox("JPEG Settings")
        self.jpeg_layout = QtWidgets.QFormLayout(self.jpeg_settings)
        self.jpeg_quality = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_quality.setValue(90)
        self.jpeg_layout.addRow("Quality", self.jpeg_quality)
        self.settings_layout.addWidget(self.jpeg_settings)

        self.heif_settings = QtWidgets.QGroupBox("HEIF Settings")
        self.heif_layout = QtWidgets.QFormLayout(self.heif_settings)
        self.heif_quality = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.heif_quality.setRange(1, 100)
        self.heif_quality.setValue(90)
        self.heif_layout.addRow("Quality", self.heif_quality)
        self.settings_layout.addWidget(self.heif_settings)

        self.dng_settings = QtWidgets.QGroupBox("DNG Settings")
        self.dng_layout = QtWidgets.QFormLayout(self.dng_settings)
        self.dng_compression = QtWidgets.QComboBox()
        self.dng_compression.setObjectName("ExportComboBox")
        self.dng_compression.addItems(["None", "Lossy"])
        self.dng_layout.addRow("Compression", self.dng_compression)
        self.settings_layout.addWidget(self.dng_settings)

        # Size
        self.size_group = QtWidgets.QGroupBox("Size")
        self.size_layout = QtWidgets.QFormLayout(self.size_group)
        self.max_width = QtWidgets.QLineEdit()
        self.max_width.setObjectName("ExportLineEdit")
        self.max_height = QtWidgets.QLineEdit()
        self.max_height.setObjectName("ExportLineEdit")
        self.size_layout.addRow("Max Width", self.max_width)
        self.size_layout.addRow("Max Height", self.max_height)
        self.settings_layout.addWidget(self.size_group)

        self.settings_layout.addStretch()

        # Export Button
        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.clicked.connect(self.start_export)
        self.settings_layout.addWidget(self.export_button)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.settings_layout.addWidget(self.progress_bar)

        self._on_format_changed(0)

    def _on_format_changed(self, index):
        format = self.format_combo.itemText(index)
        self.jpeg_settings.setVisible(format == "JPEG")
        self.heif_settings.setVisible(format == "HEIF")
        self.dng_settings.setVisible(format == "DNG")

    def _on_selection_changed(self):
        count = len(self.list_widget.selectedItems())
        self.selection_label.setText(f"{count} items selected")

    def load_folder(self, folder):
        self.current_folder = Path(folder)
        self.list_widget.clear()

        files = [
            f
            for f in self.current_folder.iterdir()
            if f.is_file() and f.suffix.lower() in pynegative.SUPPORTED_EXTS
        ]

        main_window = self.window()
        filter_mode = main_window.filter_combo.currentText()
        filter_rating = main_window.filter_rating_widget.rating()

        for path in files:
            sidecar_settings = pynegative.load_sidecar(str(path))
            rating = sidecar_settings.get("rating", 0) if sidecar_settings else 0

            if filter_rating > 0:
                if filter_mode == "Match" and rating != filter_rating:
                    continue
                if filter_mode == "Less" and rating >= filter_rating:
                    continue
                if filter_mode == "Greater" and rating <= filter_rating:
                    continue

            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            item.setData(QtCore.Qt.UserRole + 1, rating)
            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.list_widget.addItem(item)

            loader = ThumbnailLoader(str(path))
            loader.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(loader)

    def _on_thumbnail_loaded(self, path, pixmap):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(QtCore.Qt.UserRole) == path:
                if pixmap:
                    item.setIcon(QtGui.QIcon(pixmap))
                break

    def apply_filter_from_main(self):
        if self.current_folder:
            self.load_folder(self.current_folder)

    def start_export(self):
        files = [
            item.data(QtCore.Qt.UserRole) for item in self.list_widget.selectedItems()
        ]
        if not files:
            QtWidgets.QMessageBox.warning(
                self, "No files selected", "Please select at least one file to export."
            )
            return

        destination_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Destination Folder"
        )
        if not destination_folder:
            return

        settings = {
            "format": self.format_combo.currentText(),
            "jpeg_quality": self.jpeg_quality.value(),
            "heif_quality": self.heif_quality.value(),
            "dng_compression": self.dng_compression.currentText(),
            "max_width": self.max_width.text(),
            "max_height": self.max_height.text(),
        }

        self.export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        signals = ExporterSignals()
        signals.progress.connect(self.progress_bar.setValue)
        signals.finished.connect(self.on_export_finished)
        signals.error.connect(self.on_export_error)

        exporter = Exporter(signals, files, settings, destination_folder)
        self.thread_pool.start(exporter)

    def on_export_finished(self):
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.information(
            self, "Export finished", "All files have been exported successfully."
        )

    def on_export_error(self, error):
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(
            self, "Export error", f"An error occurred during export:\n{error}"
        )

    def load_presets(self):
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom")
        self.preset_combo.addItem("Web")
        self.preset_combo.addItem("Photo Print")
        self.preset_combo.addItem("Archival")
        self.preset_combo.addItem("Large Format Print")
        self.settings.beginGroup("presets")
        for preset_name in self.settings.childKeys():
            self.preset_combo.addItem(preset_name)
        self.settings.endGroup()

    def save_preset(self):
        preset_name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Preset", "Preset Name:"
        )
        if ok and preset_name:
            self.settings.beginGroup("presets")
            self.settings.setValue(
                preset_name,
                {
                    "format": self.format_combo.currentText(),
                    "jpeg_quality": self.jpeg_quality.value(),
                    "heif_quality": self.heif_quality.value(),
                    "dng_compression": self.dng_compression.currentText(),
                    "max_width": self.max_width.text(),
                    "max_height": self.max_height.text(),
                },
            )
            self.settings.endGroup()
            self.load_presets()
            self.preset_combo.setCurrentText(preset_name)

    def apply_preset(self, index):
        preset_name = self.preset_combo.itemText(index)
        if preset_name == "Custom":
            return

        if preset_name == "Web":
            self.format_combo.setCurrentText("JPEG")
            self.jpeg_quality.setValue(80)
            self.max_width.setText("1920")
            self.max_height.setText("1080")
            return

        if preset_name == "Photo Print":
            self.format_combo.setCurrentText("JPEG")
            self.jpeg_quality.setValue(95)
            self.max_width.setText("3600")
            self.max_height.setText("2400")
            return

        if preset_name == "Archival":
            self.format_combo.setCurrentText("HEIF")
            self.heif_quality.setValue(95)
            self.max_width.setText("")
            self.max_height.setText("")
            return

        if preset_name == "Large Format Print":
            self.format_combo.setCurrentText("JPEG")
            self.jpeg_quality.setValue(100)
            self.max_width.setText("10800")
            self.max_height.setText("7200")
            return

        self.settings.beginGroup("presets")
        preset = self.settings.value(preset_name)
        self.settings.endGroup()

        if preset:
            self.format_combo.setCurrentText(preset.get("format", "JPEG"))
            self.jpeg_quality.setValue(preset.get("jpeg_quality", 90))
            self.heif_quality.setValue(preset.get("heif_quality", 90))
            self.dng_compression.setCurrentText(preset.get("dng_compression", "None"))
            self.max_width.setText(preset.get("max_width", ""))
            self.max_height.setText(preset.get("max_height", ""))
