from pathlib import Path
import os
from PIL import Image
from PySide6 import QtCore
from .. import core as pynegative


class ExportProcessorSignals(QtCore.QObject):
    """Signals for export processor."""

    progress = QtCore.Signal(int)
    fileProcessed = QtCore.Signal(str)
    batchCompleted = QtCore.Signal(int, int)  # success_count, total_count
    error = QtCore.Signal(str)


class ExportProcessor(QtCore.QRunnable):
    """Handles export processing in a background thread."""

    def __init__(self, signals, files, settings, destination_folder):
        super().__init__()
        self.signals = signals
        self.files = files
        self.settings = settings
        self.destination_folder = destination_folder
        self._cancelled = False

    def run(self):
        """Execute the export batch."""
        count = len(self.files)
        success_count = 0

        for i, file in enumerate(self.files):
            if self._cancelled:
                break

            try:
                self._export_file(file)
                success_count += 1
                self.signals.fileProcessed.emit(str(file))
                self.signals.progress.emit(int(100 * (i + 1) / count))
            except Exception as e:
                self.signals.error.emit(f"Failed to export {file}: {e}")
                break

        if not self._cancelled:
            self.signals.batchCompleted.emit(success_count, count)

    def _export_file(self, file):
        """Export a single file."""
        file_path = Path(file)
        file_name = file_path.stem

        # Load full resolution image
        full_img = pynegative.open_raw(str(file_path), half_size=False)

        # Get sidecar settings
        sidecar_settings = pynegative.load_sidecar(str(file_path)) or {}

        # Process image with tone mapping
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

        # Apply size constraints if specified
        max_w = self.settings.get("max_width")
        max_h = self.settings.get("max_height")
        if max_w and max_h:
            pil_img.thumbnail((int(max_w), int(max_h)))

        # Save in specified format
        format = self.settings.get("format")
        if format == "JPEG":
            self._save_jpeg(pil_img, file_name)
        elif format == "HEIF":
            self._save_heif(pil_img, file_name)
        elif format == "DNG":
            # DNG support is not implemented yet
            pass

    def _save_jpeg(self, pil_img, file_name):
        """Save image as JPEG."""
        quality = self.settings.get("jpeg_quality", 90)
        dest_path = os.path.join(self.destination_folder, f"{file_name}.jpg")
        pil_img.save(dest_path, quality=quality)

    def _save_heif(self, pil_img, file_name):
        """Save image as HEIF."""
        quality = self.settings.get("heif_quality", 90)
        dest_path = os.path.join(self.destination_folder, f"{file_name}.heic")
        pil_img.save(dest_path, format="HEIF", quality=quality)

    def cancel(self):
        """Cancel the export batch."""
        self._cancelled = True

    @staticmethod
    def get_supported_formats():
        """Get list of supported export formats."""
        return ["JPEG", "HEIF"]

    @staticmethod
    def validate_export_settings(settings):
        """Validate export settings before starting."""
        errors = []

        format = settings.get("format")
        supported_formats = ExportProcessor.get_supported_formats()
        if format not in supported_formats:
            errors.append(f"Unsupported format: {format}")

        # Validate dimensions if provided
        max_width = settings.get("max_width")
        max_height = settings.get("max_height")

        if max_width:
            try:
                int(max_width)
                if int(max_width) <= 0:
                    errors.append("Max width must be positive")
            except ValueError:
                errors.append("Max width must be a number")

        if max_height:
            try:
                int(max_height)
                if int(max_height) <= 0:
                    errors.append("Max height must be positive")
            except ValueError:
                errors.append("Max height must be a number")

        return errors


class ExportJob(QtCore.QObject):
    """High-level export job coordinator."""

    def __init__(self, thread_pool, parent=None):
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.signals = ExportProcessorSignals()
        self._current_processor = None

    def start_export(self, files, settings, destination_folder):
        """Start a new export batch."""
        # Validate settings first
        errors = ExportProcessor.validate_export_settings(settings)
        if errors:
            for error in errors:
                self.signals.error.emit(error)
            return False

        # Create and start processor
        processor = ExportProcessor(self.signals, files, settings, destination_folder)
        self._current_processor = processor
        self.thread_pool.start(processor)
        return True

    def cancel_current_export(self):
        """Cancel the currently running export."""
        if self._current_processor:
            self._current_processor.cancel()

    def is_exporting(self):
        """Check if an export is currently running."""
        return self._current_processor is not None
