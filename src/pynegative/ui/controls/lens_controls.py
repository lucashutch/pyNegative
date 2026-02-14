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

        self.enable_check = QtWidgets.QCheckBox("Enable Correction")
        self.enable_check.setStyleSheet("font-size: 10px; color: #aaa;")
        self.enable_check.setChecked(True)
        self.enable_check.stateChanged.connect(self._on_selection_changed)
        status_row.insertWidget(1, self.enable_check)

        self.autocrop_check = QtWidgets.QCheckBox("Auto Crop")
        self.autocrop_check.setStyleSheet("font-size: 10px; color: #aaa;")
        self.autocrop_check.setChecked(True)
        self.autocrop_check.stateChanged.connect(self._on_selection_changed)
        status_row.insertWidget(2, self.autocrop_check)

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

        # Improve Completer
        cam_completer = QtWidgets.QCompleter()
        cam_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        cam_completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        cam_completer.setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion
        )
        cam_completer.setModel(self.camera_combo.model())
        self.camera_combo.setCompleter(cam_completer)

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

        # Improve Completer
        completer = QtWidgets.QCompleter()
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        # Link to the combo box's model
        completer.setModel(self.lens_combo.model())
        self.lens_combo.setCompleter(completer)

        lens_row.addWidget(self.lens_combo)
        self.section.add_layout(lens_row)

        # Manual Sliders (Placeholders for Phase 2+)
        self.distortion_slider = self._add_slider(
            "Distortion", -0.5, 0.5, 0.0, "lens_distortion", 0.01, self.section
        )
        self.vignette_slider = self._add_slider(
            "Vignette", -1.0, 1.0, 0.0, "lens_vignette", 0.01, self.section
        )
        self.ca_slider = self._add_slider(
            "CA Correction", 0.0, 2.0, 1.0, "lens_ca", 0.01, self.section
        )

        # Defringe Section
        self.section.content_layout.addSpacing(10)
        defringe_lbl = QtWidgets.QLabel("Defringe")
        defringe_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #ccc;")
        self.section.add_widget(defringe_lbl)
        self._add_slider(
            "Purple Amount", 0.0, 1.0, 0.0, "defringe_purple", 0.01, self.section
        )
        self._add_slider(
            "Green Amount", 0.0, 1.0, 0.0, "defringe_green", 0.01, self.section
        )
        self._add_slider(
            "Edge Threshold", 0.0, 0.5, 0.05, "defringe_edge", 0.01, self.section
        )
        self._add_slider(
            "Defringe Radius", 0.0, 5.0, 1.0, "defringe_radius", 1.0, self.section
        )

        # Wire up signals
        self.camera_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.lens_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.section.resetClicked.connect(self.reset_section)

        self._populate_combos()

    def reset_section(self):
        self.camera_combo.blockSignals(True)
        self.lens_combo.blockSignals(True)
        self.enable_check.blockSignals(True)
        self.autocrop_check.blockSignals(True)

        self.camera_combo.setCurrentIndex(0)
        self.lens_combo.setCurrentIndex(0)
        self.enable_check.setChecked(True)
        self.autocrop_check.setChecked(True)

        self.camera_combo.blockSignals(False)
        self.lens_combo.blockSignals(False)
        self.enable_check.blockSignals(False)
        self.autocrop_check.blockSignals(False)

        self.set_slider_value("lens_distortion", 0.0)
        self.set_slider_value("lens_vignette", 0.0)
        self.set_slider_value("lens_ca", 1.0)
        self.set_slider_value("defringe_purple", 0.0)
        self.set_slider_value("defringe_green", 0.0)
        self.set_slider_value("defringe_edge", 0.05)
        self.set_slider_value("defringe_radius", 1.0)

        self.settingChanged.emit("lens_camera_override", None)
        self.settingChanged.emit("lens_name_override", None)
        self.settingChanged.emit("lens_enabled", True)
        self.settingChanged.emit("lens_autocrop", True)
        self.settingChanged.emit("lens_distortion", 0.0)
        self.settingChanged.emit("lens_vignette", 0.0)
        self.settingChanged.emit("lens_ca", 1.0)
        self.settingChanged.emit("defringe_purple", 0.0)
        self.settingChanged.emit("defringe_green", 0.0)
        self.settingChanged.emit("defringe_edge", 0.05)
        self.settingChanged.emit("defringe_radius", 1.0)
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

        # Build camera names with deduplication
        cam_names = set()
        for c in db.cameras:
            maker = c["maker"].strip()
            model = c["model"].strip()
            if model.lower().startswith(maker.lower()):
                cam_names.add(model)
            else:
                cam_names.add(f"{maker} {model}")

        cameras = sorted(list(cam_names))
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
        if make and model and model.lower().startswith(make.lower()):
            det_cam = model
        else:
            det_cam = f"{make} {model}".strip()

        # Clean lens name is now provided by lens_resolver
        det_lens = info.get("name") or ""

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
        self.settingChanged.emit("lens_enabled", self.enable_check.isChecked())
        self.settingChanged.emit("lens_autocrop", self.autocrop_check.isChecked())

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
