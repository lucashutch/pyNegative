from pathlib import Path
from PIL import Image
import numpy as np
from PySide6 import QtCore
import pillow_heif
from .. import core as pynegative


class ExportProcessorSignals(QtCore.QObject):
    """Signals for export processor."""

    progress = QtCore.Signal(int)
    fileProcessed = QtCore.Signal(str)
    batchCompleted = QtCore.Signal(
        int, int, int
    )  # success_count, skipped_count, total_count
    error = QtCore.Signal(str)
    fileSkipped = QtCore.Signal(str, str, str)  # file_path, target_name, reason


class ExportProcessor(QtCore.QRunnable):
    """Handles export processing in a background thread."""

    def __init__(
        self, signals, files, settings, destination_folder, rename_mapping=None
    ):
        super().__init__()
        self.signals = signals
        self.files = files
        self.settings = settings
        self.destination_folder = destination_folder
        self.rename_mapping = rename_mapping or {}
        self._cancelled = False

    def run(self):
        """Execute the export batch."""
        count = len(self.files)
        success_count = 0
        skipped_count = 0

        for i, file in enumerate(self.files):
            if self._cancelled:
                break

            try:
                result = self._export_file(file)
                if result == "skipped":
                    skipped_count += 1
                else:
                    success_count += 1
                    self.signals.fileProcessed.emit(str(file))
                self.signals.progress.emit(int(100 * (i + 1) / count))
            except Exception as e:
                self.signals.error.emit(f"Failed to export {file}: {e}")
                break

        if not self._cancelled:
            self.signals.batchCompleted.emit(success_count, skipped_count, count)

    def _export_file(self, file):
        """Export a single file.

        Returns:
            "success" if exported successfully, "skipped" if file exists and was skipped.
        """
        file_path = Path(file)

        # Check if we have a rename mapping for this file
        if file_path in self.rename_mapping:
            target_name = self.rename_mapping[file_path]
            # Remove extension from target_name if present, we'll add it based on format
            target_stem = Path(target_name).stem
        else:
            target_stem = file_path.stem

        file_name = target_stem

        # Determine bit depth for processing
        heif_bit_depth_str = self.settings.get("heif_bit_depth", "8-bit")
        output_bps = 8
        if self.settings.get("format") == "HEIF" and heif_bit_depth_str in (
            "10-bit",
            "12-bit",
        ):
            output_bps = 16

        # Load full resolution image
        full_img = pynegative.open_raw(
            str(file_path), half_size=False, output_bps=output_bps
        )

        # 2. Get sidecar settings
        sidecar_settings = pynegative.load_sidecar(str(file_path)) or {}

        from ..io.lens_resolver import resolve_lens_for_file

        lens_info = resolve_lens_for_file(file_path)

        # Resolve vignette params for preprocess
        vig_k1, vig_k2, vig_k3 = 0.0, 0.0, 0.0
        if sidecar_settings.get("lens_enabled", True):
            vignette = sidecar_settings.get("lens_vignette", 0.0)
            vig_k1 = vignette
            if lens_info and "vignetting" in lens_info and lens_info["vignetting"]:
                vig = lens_info["vignetting"]
                if vig.get("model") == "pa":
                    vig_k1 += vig.get("k1", 0.0)
                    vig_k2 = vig.get("k2", 0.0)
                    vig_k3 = vig.get("k3", 0.0)

        h_full, w_full = full_img.shape[:2]
        vig_cx, vig_cy = w_full / 2.0, h_full / 2.0

        # 3. Apply Linear Stages
        # 3.1 Preprocess (WB + Exposure + Vignette) - BEFORE denoise
        img = pynegative.apply_preprocess(
            full_img,
            temperature=sidecar_settings.get("temperature", 0.0),
            tint=sidecar_settings.get("tint", 0.0),
            exposure=sidecar_settings.get("exposure", 0.0),
            vignette_k1=vig_k1,
            vignette_k2=vig_k2,
            vignette_k3=vig_k3,
            vignette_cx=vig_cx,
            vignette_cy=vig_cy,
            full_width=float(w_full),
            full_height=float(h_full),
        )

        # 3.2 Denoise (Linear, after exposure)
        luma_str = sidecar_settings.get("denoise_luma")
        chroma_str = sidecar_settings.get("denoise_chroma")
        legacy_str = sidecar_settings.get("de_noise", 0)
        l_str = float(luma_str if luma_str is not None else legacy_str)
        c_str = float(chroma_str if chroma_str is not None else legacy_str)

        if l_str > 0 or c_str > 0:
            img = pynegative.de_noise_image(
                img,
                luma_strength=l_str,
                chroma_strength=c_str,
                method=sidecar_settings.get("denoise_method", "NLMeans (Numba Fast+)"),
                zoom=1.0,
            )

        # 3.3 Lens Correction (Linear, distortion only)
        img = pynegative.apply_lens_correction(
            img, sidecar_settings, lens_info, scale=1.0, skip_vignette=True
        )

        # 3.4 Dehaze (Linear)
        dehaze_val = sidecar_settings.get("de_haze", 0)
        if dehaze_val > 0:
            img, _ = pynegative.de_haze_image(img, dehaze_val / 50.0, zoom=1.0)

        # 3.5 Sharpening (Linear)
        sharpen_val = sidecar_settings.get("sharpen_value", 0)
        if sharpen_val > 0:
            img = pynegative.sharpen_image(
                img,
                sidecar_settings.get("sharpen_radius", 0.5),
                sidecar_settings.get("sharpen_percent", 0.0),
                method="High Quality",
            )

        # 4. Apply Non-Linear Stages
        # 4.1 Tone Mapping (Contrast, Levels, Saturation, Gamma)
        img, _ = pynegative.apply_tone_map(
            img,
            contrast=sidecar_settings.get("contrast", 0.0),
            blacks=sidecar_settings.get("blacks", 0.0),
            whites=sidecar_settings.get("whites", 0.0),
            shadows=sidecar_settings.get("shadows", 0.0),
            highlights=sidecar_settings.get("highlights", 0.0),
            saturation=sidecar_settings.get("saturation", 0.0),
        )

        # 4.2 Defringe
        img = pynegative.apply_defringe(img, sidecar_settings)

        # 4.3 Geometry (Flip, Rotate, Crop)
        img = pynegative.apply_geometry(
            img,
            rotate=sidecar_settings.get("rotation", 0.0),
            crop=sidecar_settings.get("crop"),
            flip_h=sidecar_settings.get("flip_h", False),
            flip_v=sidecar_settings.get("flip_v", False),
        )

        # Convert to PIL Image for final steps
        if output_bps == 16:
            pil_img = Image.fromarray(
                (np.clip(img, 0, 1) * 65535).astype("uint16"), "RGB"
            )
        else:
            pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8"))

        # Apply size constraints if specified
        max_w = self.settings.get("max_width")
        max_h = self.settings.get("max_height")
        if max_w and max_h:
            pil_img.thumbnail((int(max_w), int(max_h)))

        # Save in specified format
        format = self.settings.get("format")
        if format == "JPEG":
            return self._save_jpeg(pil_img, file_name)
        elif format == "HEIF":
            return self._save_heif(pil_img, file_name)

        return "success"

    def _save_jpeg(self, pil_img, file_name):
        """Save image as JPEG.

        Returns:
            "success" if saved, "skipped" if file already exists.
        """
        quality = self.settings.get("jpeg_quality", 90)
        dest_path = Path(self.destination_folder) / f"{file_name}.jpg"

        # Check if file already exists
        if dest_path.exists():
            self.signals.fileSkipped.emit(
                str(dest_path), f"{file_name}.jpg", "File already exists"
            )
            return "skipped"

        pil_img.save(str(dest_path), quality=quality)
        return "success"

    def _save_heif(self, pil_img, file_name):
        """Save image as HEIF.

        Returns:
            "success" if saved, "skipped" if file already exists.
        """
        quality = self.settings.get("heif_quality", 90)
        bit_depth_str = self.settings.get("heif_bit_depth", "8-bit")
        dest_path = Path(self.destination_folder) / f"{file_name}.heic"

        # Check if file already exists
        if dest_path.exists():
            self.signals.fileSkipped.emit(
                str(dest_path), f"{file_name}.heic", "File already exists"
            )
            return "skipped"

        if bit_depth_str == "12-bit":
            original_setting = pillow_heif.options.SAVE_HDR_TO_12_BIT
            pillow_heif.options.SAVE_HDR_TO_12_BIT = True
            try:
                pil_img.save(str(dest_path), format="HEIF", quality=quality)
            finally:
                pillow_heif.options.SAVE_HDR_TO_12_BIT = original_setting
        elif bit_depth_str == "10-bit":
            # 16-bit images are saved as 10-bit by default
            pil_img.save(str(dest_path), format="HEIF", quality=quality)
        else:
            # 8-bit
            pil_img.save(str(dest_path), format="HEIF", quality=quality)

        return "success"

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

    def start_export(self, files, settings, destination_folder, rename_mapping=None):
        """Start a new export batch."""
        # Validate settings first
        errors = ExportProcessor.validate_export_settings(settings)
        if errors:
            for error in errors:
                self.signals.error.emit(error)
            return False

        # Create and start processor
        processor = ExportProcessor(
            self.signals, files, settings, destination_folder, rename_mapping
        )
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
