from PySide6 import QtWidgets, QtCore
from .base import BaseControlWidget
from ..widgets.collapsiblesection import CollapsibleSection
from ...io import lens_db_xml, lens_resolver


class LensControls(BaseControlWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        self.section = CollapsibleSection("LENS", True, self)
        main_layout.addWidget(self.section)

        # Status and Auto Detect Row
        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-size: 10px; color: #aaa;")
        status_row.addWidget(self.status_label, 1)

        self.auto_detect_btn = QtWidgets.QPushButton("Auto")
        self.auto_detect_btn.setFixedSize(30, 14)
        self.auto_detect_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                color: #bbb;
                font-size: 9px;
                padding: 0px;
                margin: 0px;
                min-height: 0px;
                line-height: 12px;
            }
            QPushButton:hover {
                background-color: rgba(139, 92, 246, 0.2);
                border: 1px solid #8b5cf6;
                color: #eee;
            }
        """)
        self.auto_detect_btn.clicked.connect(self._on_auto_detect)
        status_row.addWidget(self.auto_detect_btn)

        self.section.add_layout(status_row)

        combo_style = """
            QComboBox {
                font-size: 11px;
                padding: 0px 2px;
                min-height: 18px;
            }
            QComboBox QAbstractItemView {
                font-size: 11px;
            }
        """

        # Camera Selection
        cam_row = QtWidgets.QHBoxLayout()
        cam_label = QtWidgets.QLabel("Camera:")
        cam_label.setStyleSheet("font-size: 11px;")
        cam_row.addWidget(cam_label)

        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.setFixedHeight(20)
        self.camera_combo.setStyleSheet(combo_style)
        self.camera_combo.setEditable(True)
        self.camera_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        if self.camera_combo.completer():
            self.camera_combo.completer().setCompletionMode(
                QtWidgets.QCompleter.CompletionMode.PopupCompletion
            )
            self.camera_combo.completer().setFilterMode(
                QtCore.Qt.MatchFlag.MatchContains
            )
        cam_row.addWidget(self.camera_combo)
        self.section.add_layout(cam_row)

        # Lens Selection
        lens_row = QtWidgets.QHBoxLayout()
        lens_label = QtWidgets.QLabel("Lens:")
        lens_label.setStyleSheet("font-size: 11px;")
        lens_row.addWidget(lens_label)

        self.lens_combo = QtWidgets.QComboBox()
        self.lens_combo.setFixedHeight(20)
        self.lens_combo.setStyleSheet(combo_style)
        self.lens_combo.setEditable(True)
        self.lens_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        if self.lens_combo.completer():
            self.lens_combo.completer().setCompletionMode(
                QtWidgets.QCompleter.CompletionMode.PopupCompletion
            )
            self.lens_combo.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        lens_row.addWidget(self.lens_combo)
        self.section.add_layout(lens_row)

        # Manual Sliders (Placeholders for Phase 2+)
        self.distortion_slider = self._add_slider(
            "Distortion", -0.5, 0.5, 0.0, "lens_distortion", 0.01, self.section
        )
        self.vignette_slider = self._add_slider(
            "Vignette", 0.0, 1.0, 0.0, "lens_vignette", 0.01, self.section
        )
        self.ca_slider = self._add_slider(
            "CA Correction", 0.0, 1.0, 0.0, "lens_ca", 0.01, self.section
        )

        # Wire up signals
        self.camera_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.lens_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.section.resetClicked.connect(self.reset_section)

        self._populate_combos()

    def reset_section(self):
        self.camera_combo.setCurrentIndex(0)
        self.lens_combo.setCurrentIndex(0)
        self.set_slider_value("lens_distortion", 0.0)
        self.set_slider_value("lens_vignette", 0.0)
        self.set_slider_value("lens_ca", 0.0)

        self.settingChanged.emit("lens_camera_override", None)
        self.settingChanged.emit("lens_name_override", None)
        self.settingChanged.emit("lens_distortion", 0.0)
        self.settingChanged.emit("lens_vignette", 0.0)
        self.settingChanged.emit("lens_ca", 0.0)
        self.settingChanged.emit("lens_auto_detect", True)

    def _populate_combos(self):
        db = lens_db_xml.get_instance()
        if not db.loaded:
            self.camera_combo.addItems(["Auto"])
            self.lens_combo.addItems(["Auto"])
            return

        self.camera_combo.blockSignals(True)
        self.lens_combo.blockSignals(True)

        self.camera_combo.clear()
        cameras = sorted(list(set(f"{c['maker']} {c['model']}" for c in db.cameras)))
        self.camera_combo.addItems(["Auto"] + cameras)

        self.lens_combo.clear()
        lenses = db.get_all_lens_names()
        self.lens_combo.addItems(["Auto"] + lenses)

        self.camera_combo.blockSignals(False)
        self.lens_combo.blockSignals(False)

    def set_lens_info(self, source, info):
        # Only update status label if we are not manually overriding
        # Check current combos; if they are not Auto, we are in override mode
        is_override = (
            self.camera_combo.currentIndex() > 0 or self.lens_combo.currentIndex() > 0
        )

        if not is_override:
            # Update status label - only show source, not the name
            if source == lens_resolver.ProfileSource.EMBEDDED:
                self.status_label.setText("Source: Embedded RAW")
                self.status_label.setStyleSheet(
                    "font-size: 10px; color: #4CAF50;"
                )  # Green
            elif source == lens_resolver.ProfileSource.LENSFUN_DB:
                self.status_label.setText("Source: Lensfun Match")
                self.status_label.setStyleSheet(
                    "font-size: 10px; color: #2196F3;"
                )  # Blue
            elif source == lens_resolver.ProfileSource.MANUAL:
                self.status_label.setText("Source: Manual Selection")
                self.status_label.setStyleSheet(
                    "font-size: 10px; color: #FF9800;"
                )  # Orange
            else:
                if not lens_db_xml.is_database_available():
                    self.status_label.setText("Manual Mode (DB Missing)")
                    self.status_label.setStyleSheet(
                        "font-size: 10px; color: #F44336;"
                    )  # Red
                else:
                    self.status_label.setText("No Lens Detected")
                    self.status_label.setStyleSheet("font-size: 10px; color: #aaa;")

        # Update combo box "Auto" labels to show detected info
        exif = info.get("exif", {})
        make = exif.get("camera_make", "").strip()
        model = exif.get("camera_model", "").strip()

        # Avoid duplicate maker if it's already in the model string
        if make.lower() in model.lower():
            det_cam = model
        else:
            det_cam = f"{make} {model}".strip()

        # Fix for duplicate maker in lens name
        det_lens = info.get("name") or ""
        # If lens data is from lensfun, it might already have the maker in it
        if "lens_data" in info and det_lens:
            lens_maker = info["lens_data"].get("maker", "").strip()
            if lens_maker.lower() in det_lens.lower():
                # Already includes maker correctly
                pass
            else:
                det_lens = f"{lens_maker} {det_lens}".strip()

        # Final sanity check for duplicate maker in det_lens (e.g. "Canon Canon EF")
        parts = det_lens.split()
        if len(parts) >= 2 and parts[0].lower() == parts[1].lower():
            det_lens = " ".join(parts[1:])

        self.camera_combo.blockSignals(True)
        self.lens_combo.blockSignals(True)

        self.camera_combo.setItemText(0, f"Auto ({det_cam or 'None'})")
        self.lens_combo.setItemText(0, f"Auto ({det_lens or 'None'})")

        self.camera_combo.blockSignals(False)
        self.lens_combo.blockSignals(False)

        # Update status if overrides are present
        if self.camera_combo.currentIndex() > 0 or self.lens_combo.currentIndex() > 0:
            self.status_label.setText("Source: Manual Override")
            self.status_label.setStyleSheet(
                "font-size: 10px; color: #FF9800;"
            )  # Orange

    def _on_selection_changed(self):
        # Emit settings for manual override
        cam_idx = self.camera_combo.currentIndex()
        lens_idx = self.lens_combo.currentIndex()

        cam = self.camera_combo.currentText() if cam_idx > 0 else None
        lens = self.lens_combo.currentText() if lens_idx > 0 else None

        self.settingChanged.emit("lens_camera_override", cam)
        self.settingChanged.emit("lens_name_override", lens)

        # Update status immediately
        if cam_idx > 0 or lens_idx > 0:
            self.status_label.setText("Source: Manual Override")
            self.status_label.setStyleSheet(
                "font-size: 10px; color: #FF9800;"
            )  # Orange

    def _on_auto_detect(self):
        self.camera_combo.setCurrentIndex(0)
        self.lens_combo.setCurrentIndex(0)
        self.settingChanged.emit("lens_auto_detect", True)
