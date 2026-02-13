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

        # Status Label
        self.status_label = QtWidgets.QLabel("No lens detected")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "font-size: 11px; color: #aaa; margin-bottom: 4px;"
        )
        self.section.add_widget(self.status_label)

        # Camera Selection
        cam_row = QtWidgets.QHBoxLayout()
        cam_row.addWidget(QtWidgets.QLabel("Camera:"))
        self.camera_combo = QtWidgets.QComboBox()
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
        lens_row.addWidget(QtWidgets.QLabel("Lens:"))
        self.lens_combo = QtWidgets.QComboBox()
        self.lens_combo.setEditable(True)
        self.lens_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        if self.lens_combo.completer():
            self.lens_combo.completer().setCompletionMode(
                QtWidgets.QCompleter.CompletionMode.PopupCompletion
            )
            self.lens_combo.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        lens_row.addWidget(self.lens_combo)
        self.section.add_layout(lens_row)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.auto_detect_btn = QtWidgets.QPushButton("Auto Detect")
        self.auto_detect_btn.clicked.connect(self._on_auto_detect)
        btn_row.addWidget(self.auto_detect_btn)

        self.update_db_btn = QtWidgets.QPushButton("Update DB")
        self.update_db_btn.clicked.connect(self._on_update_db)
        self.update_db_btn.setToolTip("Download latest lens database from GitHub")
        btn_row.addWidget(self.update_db_btn)
        self.section.add_layout(btn_row)

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
        self.camera_combo.setCurrentText("Auto")
        self.lens_combo.setCurrentText("Auto")
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
        if source == lens_resolver.ProfileSource.EMBEDDED:
            self.status_label.setText(f"Using embedded profile: {info['name']}")
            self.status_label.setStyleSheet("font-size: 11px; color: #4CAF50;")  # Green
        elif source == lens_resolver.ProfileSource.LENSFUN_DB:
            self.status_label.setText(f"Matched lensfun profile: {info['name']}")
            self.status_label.setStyleSheet("font-size: 11px; color: #2196F3;")  # Blue
        elif source == lens_resolver.ProfileSource.MANUAL:
            self.status_label.setText(
                f"Manual mode - profile not found for {info.get('name', 'lens')}"
            )
            self.status_label.setStyleSheet(
                "font-size: 11px; color: #FF9800;"
            )  # Orange
        else:
            if not lens_db_xml.is_database_available():
                self.status_label.setText("Manual mode - database not found")
                self.status_label.setStyleSheet(
                    "font-size: 11px; color: #F44336;"
                )  # Red
            else:
                self.status_label.setText("No lens detected")
                self.status_label.setStyleSheet("font-size: 11px; color: #aaa;")

    def _on_selection_changed(self):
        # Emit settings for manual override
        cam = self.camera_combo.currentText()
        lens = self.lens_combo.currentText()
        self.settingChanged.emit("lens_camera_override", cam if cam != "Auto" else None)
        self.settingChanged.emit("lens_name_override", lens if lens != "Auto" else None)

    def _on_auto_detect(self):
        self.camera_combo.setCurrentText("Auto")
        self.lens_combo.setCurrentText("Auto")
        self.settingChanged.emit("lens_auto_detect", True)

    def _on_update_db(self):
        # This would ideally call the download script
        import subprocess
        import sys
        from pathlib import Path

        script_path = (
            Path(__file__).parents[4] / "scripts" / "download_lens_database.py"
        )
        db_path = Path.home() / ".local" / "share" / "pyNegative" / "data" / "lensfun"

        try:
            self.status_label.setText("Updating database...")
            subprocess.Popen(
                [sys.executable, str(script_path), "--output-dir", str(db_path)]
            )
            # We don't wait for it to finish here for simplicity in UI,
            # but ideally we'd show progress and reload on finish
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to start update: {e}"
            )
