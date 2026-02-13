"""Reusable Metadata Panel widget for displaying EXIF data from RAW files."""

from pathlib import Path
from PySide6 import QtWidgets, QtCore
import exifread

from ... import core as pynegative


class MetadataPanel(QtWidgets.QWidget):
    """A panel that displays EXIF metadata for a given RAW image file.

    Can be used standalone or embedded into any layout. Call load_for_path()
    to display metadata for a file, or show_empty() to show placeholder dashes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setObjectName("MetadataPanel")
        self.setVisible(False)

        self._current_path = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Title
        title = QtWidgets.QLabel("Image Metadata")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Scrollable area for metadata fields
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QFormLayout(self._content)
        self._content_layout.setSpacing(6)
        scroll_area.setWidget(self._content)

        layout.addWidget(scroll_area)

    # --- Public API ---

    @property
    def content_layout(self):
        """Access to the form layout (for backward compat with editor)."""
        return self._content_layout

    def load_for_path(self, raw_path):
        """Load and display EXIF metadata for the given file path."""
        if raw_path is None:
            self.show_empty()
            return

        self._current_path = str(raw_path)
        self._clear()

        # Show loading state
        loading_label = QtWidgets.QLabel("Loading metadata...")
        self._content_layout.addRow(loading_label)

        try:
            exif_data = self._extract_exif_data(self._current_path)
            self._clear()
            self._populate(exif_data)
        except Exception as e:
            self._clear()
            error_label = QtWidgets.QLabel(f"Error loading metadata: {str(e)}")
            error_label.setWordWrap(True)
            self._content_layout.addRow(error_label)

    def show_empty(self):
        """Show the panel with empty placeholder values (dashes)."""
        self._current_path = None
        self._clear()
        self._populate({})

    def clear(self):
        """Clear all metadata fields."""
        self._current_path = None
        self._clear()

    # --- Internal ---

    def _clear(self):
        """Remove all widgets from the content layout."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @staticmethod
    def _extract_exif_data(raw_path):
        """Extract EXIF data from RAW file using rawpy and custom BMFF parser."""
        exif = {}

        # File info
        p = Path(raw_path)
        exif["file_name"] = p.name
        exif["file_location"] = str(p.parent)
        ext = p.suffix.lower()

        try:
            import rawpy
            from ...io import lens_metadata, bmff_metadata

            # 1. Try rawpy for basic RAW handling and WB
            with rawpy.imread(str(raw_path)) as raw:
                # Use our robust lens info extractor
                lens_info = lens_metadata.extract_lens_info(raw_path)

                exif["camera_make"] = lens_info.get("camera_make")
                exif["camera_model"] = lens_info.get("camera_model")
                exif["lens_model"] = lens_info.get("lens_model")

                if hasattr(raw, "camera_whitebalance"):
                    wb = raw.camera_whitebalance
                    if wb is not None and len(wb) > 0:
                        exif["_raw_wb"] = wb

                if hasattr(raw, "sizes"):
                    sizes = raw.sizes
                    if hasattr(sizes, "raw_width") and hasattr(sizes, "raw_height"):
                        exif["width"] = sizes.raw_width
                        exif["height"] = sizes.raw_height

            # 2. For CR3 files, use our manual BMFF extractor for exposure data
            if ext == ".cr3":
                bmff_tags = bmff_metadata.extract_bmff_metadata(raw_path)
                bmff_info = bmff_metadata.get_exposure_info(bmff_tags)

                # Fill in exposure fields that rawpy misses
                if "iso" in bmff_info:
                    exif["iso"] = bmff_info["iso"]
                if "shutter_speed" in bmff_info:
                    exif["shutter_speed"] = bmff_info["shutter_speed"]
                if "aperture" in bmff_info:
                    exif["aperture"] = bmff_info["aperture"]
                if "focal_length" in bmff_info:
                    exif["focal_length"] = bmff_info["focal_length"]

                # Additional lens info
                exif["focal_length_35mm"] = bmff_tags.get("EXIF FocalLengthIn35mmFilm")
                exif["max_aperture"] = bmff_tags.get("EXIF MaxApertureValue")

                # Resolution Source (pyNegative specific)
                from ...io import lens_resolver

                source, resolved = lens_resolver.resolve_lens_profile(raw_path)
                if resolved:
                    exif["lens_model_resolved"] = resolved.get("name")
                    exif["lens_source"] = source.name
                else:
                    exif["lens_source"] = "NONE"

                # Other metadata
                exif["date_taken"] = bmff_tags.get(
                    "EXIF DateTimeOriginal"
                ) or bmff_tags.get("Image DateTime")
                exif["artist"] = bmff_tags.get("Image Artist")
                exif["copyright"] = bmff_tags.get("Image Copyright")

            # 3. For other files or if still missing data, try thumbnail/standard EXIF
            if not exif.get("iso"):
                try:
                    with rawpy.imread(str(raw_path)) as raw:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            from io import BytesIO

                            tags = exifread.process_file(
                                BytesIO(thumb.data), details=False
                            )

                            # Primary exposure fields
                            if not exif.get("iso"):
                                exif["iso"] = tags.get("EXIF ISOSpeedRatings")
                            if not exif.get("shutter_speed"):
                                exif["shutter_speed"] = tags.get("EXIF ExposureTime")
                            if not exif.get("aperture"):
                                exif["aperture"] = tags.get("EXIF FNumber")
                            if not exif.get("focal_length"):
                                exif["focal_length"] = tags.get("EXIF FocalLength")

                            # Date/time
                            if not exif.get("date_taken"):
                                exif["date_taken"] = (
                                    tags.get("EXIF DateTimeOriginal")
                                    or tags.get("Image DateTime")
                                    or tags.get("EXIF DateTimeDigitized")
                                )

                            # Resolution Source (pyNegative specific)
                            from ...io import lens_resolver

                            source, resolved = lens_resolver.resolve_lens_profile(
                                raw_path
                            )
                            if resolved:
                                exif["lens_model_resolved"] = resolved.get("name")
                                exif["lens_source"] = source.name
                            else:
                                exif["lens_source"] = "NONE"

                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Metadata extraction failed for {raw_path}: {e}")

        # Fallback for date
        if not exif.get("date_taken"):
            date_taken = pynegative.get_exif_capture_date(raw_path)
            if date_taken:
                exif["date_taken"] = date_taken

        # File size
        try:
            exif["file_size"] = p.stat().st_size
        except Exception:
            pass

        return exif

    def _add_section_separator(self):
        """Add a horizontal separator line to the content layout."""
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self._content_layout.addRow(separator)

    def _add_field(self, label, value):
        """Add a labeled field to the content layout."""
        value_label = QtWidgets.QLabel(value)
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._content_layout.addRow(QtWidgets.QLabel(f"<b>{label}:</b>"), value_label)

    def _populate(self, exif_data):
        """Populate metadata panel with EXIF data."""

        def fv(value):
            """Format a value, returning dash for None."""
            if value is None:
                return "—"
            return str(value)

        def format_date(value):
            """Format date as YYYY.MM.DD - HH:MM:SS."""
            if value is None:
                return "—"
            try:
                s = str(value).strip()
                # EXIF dates are typically "YYYY:MM:DD HH:MM:SS"
                if ":" in s and len(s) >= 19:
                    date_part = s[:10].replace(":", ".")
                    time_part = s[11:19]
                    return f"{date_part} - {time_part}"
                return s
            except Exception:
                return str(value)

        def format_shutter_speed(value):
            if value is None:
                return "—"
            try:
                # Handle Ratio objects
                if hasattr(value, "num") and hasattr(value, "den"):
                    num, den = value.num, value.den
                    if den == 0:
                        return "—"
                    if num == 1:
                        return f"1/{den}s"
                    elif den == 1:
                        return f"{num}s"
                    else:
                        decimal = num / den
                        return f"{decimal:.2f}s"

                # Handle string fractions
                s_val = str(value)
                if "/" in s_val:
                    parts = s_val.split("/")
                    if len(parts) == 2:
                        try:
                            num = float(parts[0])
                            den = float(parts[1])
                            if den != 0:
                                if num == 1:
                                    return f"1/{int(den)}s"
                                elif den == 1:
                                    return f"{int(num)}s"
                                else:
                                    return f"{num / den:.2f}s"
                        except ValueError:
                            pass

                return f"{value}s" if value != "—" else "—"
            except Exception:
                return str(value)

        def format_aperture(value):
            if value is None:
                return "—"
            try:
                # Handle Ratio objects from exifread
                if hasattr(value, "num") and hasattr(value, "den"):
                    if value.den == 0:
                        return "—"
                    f_stop = value.num / value.den
                    return f"f/{f_stop:.1f}"

                # Handle string fractions like "9/5"
                s_val = str(value)
                if "/" in s_val:
                    parts = s_val.split("/")
                    if len(parts) == 2:
                        try:
                            num = float(parts[0])
                            den = float(parts[1])
                            if den != 0:
                                return f"f/{num / den:.1f}"
                        except ValueError:
                            pass

                return f"f/{value}"
            except Exception:
                return str(value)

        def format_focal_length(value):
            if value is None:
                return "—"
            try:
                # Handle Ratio objects
                if hasattr(value, "num") and hasattr(value, "den"):
                    if value.den == 0:
                        return "—"
                    mm = value.num / value.den
                    return f"{mm:.0f}mm"

                # Handle string fractions
                s_val = str(value)
                if "/" in s_val:
                    parts = s_val.split("/")
                    if len(parts) == 2:
                        try:
                            num = float(parts[0])
                            den = float(parts[1])
                            if den != 0:
                                return f"{num / den:.0f}mm"
                        except ValueError:
                            pass

                return f"{value}mm"
            except Exception:
                return str(value)

        def format_file_size(bytes_size):
            if bytes_size is None:
                return "—"
            try:
                mb = bytes_size / (1024 * 1024)
                return f"{mb:.2f} MB"
            except Exception:
                return "—"

        # === File Info (always at top) ===
        self._add_field("File Name", fv(exif_data.get("file_name")))
        location = exif_data.get("file_location")
        if location:
            loc_label = QtWidgets.QLabel(str(location))
            loc_label.setWordWrap(True)
            loc_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            loc_label.setStyleSheet("color: #888; font-size: 11px;")
            self._content_layout.addRow(loc_label)

        self._add_section_separator()

        # === Primary Exposure ===
        self._add_field("ISO", fv(exif_data.get("iso")))
        self._add_field(
            "Shutter Speed",
            format_shutter_speed(exif_data.get("shutter_speed")),
        )
        self._add_field("Aperture", format_aperture(exif_data.get("aperture")))
        self._add_field(
            "Focal Length",
            format_focal_length(exif_data.get("focal_length")),
        )
        if exif_data.get("focal_length_35mm"):
            self._add_field(
                "35mm Equiv.",
                f"{fv(exif_data.get('focal_length_35mm'))}mm",
            )

        self._add_section_separator()

        # === Camera & Lens ===
        if exif_data.get("camera_make") or exif_data.get("camera_model"):
            camera = (
                f"{fv(exif_data.get('camera_make'))} "
                f"{fv(exif_data.get('camera_model'))}"
            )
            self._add_field("Camera", camera.strip())

        # Always show Lens field if we have camera info, even if lens is unknown
        lens_model = exif_data.get("lens_model")
        if lens_model:
            self._add_field("Lens", fv(lens_model))
        elif exif_data.get("camera_make") or exif_data.get("camera_model"):
            self._add_field("Lens", "Unknown / Not detected")

        if exif_data.get("lens_source"):
            source = exif_data.get("lens_source")
            resolved = exif_data.get("lens_model_resolved")

            source_text = {
                "EMBEDDED": "Embedded RAW",
                "LENSFUN_DB": "Lensfun Match",
                "MANUAL": "Manual Override",
                "NONE": "Manual Mode",
            }.get(source, source)

            if source == "NONE":
                self._add_field("Lens Profile", f"No Profile Found ({source_text})")
            else:
                self._add_field(
                    "Lens Profile",
                    f"{fv(resolved)} ({source_text})",
                )

        if exif_data.get("max_aperture"):
            try:
                val = exif_data["max_aperture"]
                if hasattr(val, "num") and hasattr(val, "den"):
                    f_stop = val.num / val.den
                    # Max aperture is in APEX units, convert:  f = sqrt(2^apex)
                    import math

                    f_number = math.sqrt(2**f_stop)
                    self._add_field("Max Aperture", f"f/{f_number:.1f}")
                else:
                    self._add_field("Max Aperture", fv(val))
            except Exception:
                self._add_field("Max Aperture", fv(exif_data.get("max_aperture")))

        # === Date ===
        if exif_data.get("date_taken"):
            self._add_section_separator()
            self._add_field("Date Taken", format_date(exif_data.get("date_taken")))

        # === Exposure Details ===
        has_exposure_details = any(
            exif_data.get(k)
            for k in [
                "exposure_compensation",
                "metering_mode",
                "exposure_program",
                "exposure_mode",
            ]
        )
        if has_exposure_details:
            self._add_section_separator()
            if exif_data.get("exposure_compensation"):
                self._add_field(
                    "Exp. Comp.",
                    fv(exif_data.get("exposure_compensation")),
                )
            if exif_data.get("metering_mode"):
                self._add_field("Metering", fv(exif_data.get("metering_mode")))
            if exif_data.get("exposure_program"):
                self._add_field("Program", fv(exif_data.get("exposure_program")))
            if exif_data.get("exposure_mode"):
                self._add_field("Exp. Mode", fv(exif_data.get("exposure_mode")))

        # === Color & Processing ===
        has_color = any(
            exif_data.get(k)
            for k in [
                "white_balance",
                "color_space",
                "flash",
                "scene_capture_type",
            ]
        )
        if has_color:
            self._add_section_separator()
            if exif_data.get("white_balance"):
                self._add_field(
                    "White Balance",
                    fv(exif_data.get("white_balance")),
                )
            if exif_data.get("color_space"):
                self._add_field("Color Space", fv(exif_data.get("color_space")))
            if exif_data.get("flash"):
                self._add_field("Flash", fv(exif_data.get("flash")))
            if exif_data.get("scene_capture_type"):
                self._add_field(
                    "Scene Type",
                    fv(exif_data.get("scene_capture_type")),
                )

        # === Image Processing (contrast, etc.) ===
        has_processing = any(
            exif_data.get(k)
            for k in ["contrast", "saturation", "sharpness", "digital_zoom"]
        )
        if has_processing:
            self._add_section_separator()
            if exif_data.get("contrast"):
                self._add_field("Contrast", fv(exif_data.get("contrast")))
            if exif_data.get("saturation"):
                self._add_field("Saturation", fv(exif_data.get("saturation")))
            if exif_data.get("sharpness"):
                self._add_field("Sharpness", fv(exif_data.get("sharpness")))
            if exif_data.get("digital_zoom"):
                self._add_field(
                    "Digital Zoom",
                    fv(exif_data.get("digital_zoom")),
                )

        # === GPS ===
        if exif_data.get("gps_latitude") or exif_data.get("gps_longitude"):
            self._add_section_separator()
            if exif_data.get("gps_latitude"):
                self._add_field("Latitude", fv(exif_data.get("gps_latitude")))
            if exif_data.get("gps_longitude"):
                self._add_field("Longitude", fv(exif_data.get("gps_longitude")))

        # === Image Info ===
        self._add_section_separator()
        if exif_data.get("width") and exif_data.get("height"):
            dimensions = f"{exif_data['width']} × {exif_data['height']}"
            self._add_field("Dimensions", dimensions)

        if exif_data.get("file_size"):
            self._add_field(
                "File Size",
                format_file_size(exif_data.get("file_size")),
            )

        if exif_data.get("orientation"):
            self._add_field("Orientation", fv(exif_data.get("orientation")))

        # === Author / Copyright ===
        has_author = any(
            exif_data.get(k)
            for k in ["artist", "copyright", "image_description", "software"]
        )
        if has_author:
            self._add_section_separator()
            if exif_data.get("artist"):
                self._add_field("Artist", fv(exif_data.get("artist")))
            if exif_data.get("copyright"):
                self._add_field("Copyright", fv(exif_data.get("copyright")))
            if exif_data.get("image_description"):
                self._add_field(
                    "Description",
                    fv(exif_data.get("image_description")),
                )
            if exif_data.get("software"):
                self._add_field("Software", fv(exif_data.get("software")))
