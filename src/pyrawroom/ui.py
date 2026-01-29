import sys
import os
import json
import numpy as np
from PIL import Image, ImageQt

from PySide6 import QtWidgets, QtGui, QtCore

from . import core as pyrawroom # Assuming this module is available and contains the core logic

class PyRawEditorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Tone Editor")
        self.setGeometry(100, 100, 1300, 900) # x, y, width, height

        self.raw_path = None
        self.base_img_full = None
        self.base_img_preview = None
        self.current_qpixmap = None # To hold the currently displayed QPixmap

        self._init_ui()

    def _init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # --- Left Panel ---
        self.panel = QtWidgets.QFrame()
        self.panel.setContentsMargins(10, 10, 10, 10) # padding
        self.panel.setFixedWidth(350)
        self.panel.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        main_layout.addWidget(self.panel)

        # --- Canvas Frame (Right Side) ---
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas_frame)

        self.canvas_label = QtWidgets.QLabel()
        self.canvas_label.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas_label.setStyleSheet("background-color: #2b2b2b;")
        self.canvas_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding) # Ensure it expands

        canvas_frame_layout = QtWidgets.QVBoxLayout(self.canvas_frame)
        canvas_frame_layout.addWidget(self.canvas_label)

        # Connect resize event for canvas_label to update preview scaling
        self.canvas_label.installEventFilter(self) # Install event filter on self for canvas_label

        self._setup_file_operations()
        self._setup_tone_controls()
        self._setup_sharpening_controls()
        self._setup_save_button()

    def eventFilter(self, watched, event):
        if watched == self.canvas_label and event.type() == QtCore.QEvent.Type.Resize:
            self.update_preview() # Re-scale image on resize
        return super().eventFilter(watched, event)

    def _setup_file_operations(self):
        file_frame = QtWidgets.QGroupBox("File Operations")
        file_layout = QtWidgets.QVBoxLayout(file_frame)

        open_btn = QtWidgets.QPushButton("Open RAW File")
        open_btn.clicked.connect(self.browse_raw_file)
        file_layout.addWidget(open_btn)

        load_edit_btn = QtWidgets.QPushButton("Load Edit (.json)")
        load_edit_btn.clicked.connect(self.load_json_edit)
        file_layout.addWidget(load_edit_btn)

        self.lbl_info = QtWidgets.QLabel("No file loaded")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setMaximumWidth(330) # Adjusted for panel width
        file_layout.addWidget(self.lbl_info)

        self.panel_layout.addWidget(file_frame)
        self.panel_layout.addSpacing(10) # ttk.Separator equivalent

    def _setup_tone_controls(self):
        self._add_separator()
        self.val_exposure = 0.0
        self.val_whites = 1.0
        self.val_blacks = 0.0
        self._add_slider("Exposure", -4.0, 4.0, self.val_exposure, "val_exposure", 0.01)
        self._add_slider("Contrast (Whites)", 0.5, 2.0, self.val_whites, "val_whites", 0.01)
        self._add_slider("Blacks", -0.2, 0.2, self.val_blacks, "val_blacks", 0.001)

        self._add_separator()

        self.val_highlights = 0.0
        self.val_shadows = 0.0
        self._add_slider("Highlights", -1.0, 1.0, self.val_highlights, "val_highlights", 0.01)
        self._add_slider("Shadows", -1.0, 1.0, self.val_shadows, "val_shadows", 0.01)

    def _setup_sharpening_controls(self):
        self._add_separator()

        self.var_sharpen_enabled = False
        self.sharpen_checkbox = QtWidgets.QCheckBox("Enable Sharpening")
        self.sharpen_checkbox.setChecked(self.var_sharpen_enabled)
        self.sharpen_checkbox.stateChanged.connect(lambda state: self._update_sharpen_state(state))
        self.panel_layout.addWidget(self.sharpen_checkbox)

        self.val_radius = 2.0
        self.val_percent = 150
        self._add_slider("Sharpen Radius", 0.5, 5.0, self.val_radius, "val_radius", 0.01)
        self._add_slider("Sharpen Amount", 0, 300, self.val_percent, "val_percent", 1)

    def _setup_save_button(self):
        self._add_separator(extra_spacing=20)
        self.btn_save = QtWidgets.QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setEnabled(False) # Initial state
        self.panel_layout.addWidget(self.btn_save)

        # Add a stretch to push everything to the top
        self.panel_layout.addStretch()

    def _add_separator(self, extra_spacing=10):
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.panel_layout.addWidget(separator)
        if extra_spacing > 0:
            self.panel_layout.addSpacing(extra_spacing)

    def _add_slider(self, label_text, min_val, max_val, default, var_name, step_size, is_int=False):
        frame = QtWidgets.QFrame()
        frame_layout = QtWidgets.QVBoxLayout(frame)

        label = QtWidgets.QLabel(label_text)
        frame_layout.addWidget(label)

        slider_widget = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # QSlider works with integers, so we scale float values
        if is_int:
            slider_widget.setRange(int(min_val), int(max_val))
            slider_widget.setValue(int(default))
            slider_widget.setSingleStep(int(step_size))
        else:
            # Use a multiplier to handle floating point steps
            multiplier = 1000 # Max precision for 3 decimal places
            slider_widget.setRange(int(min_val * multiplier), int(max_val * multiplier))
            slider_widget.setValue(int(default * multiplier))
            slider_widget.setSingleStep(int(step_size * multiplier))

        # Value display label
        value_label = QtWidgets.QLabel(f"{default:.2f}" if not is_int else str(default))
        value_label.setAlignment(QtCore.Qt.AlignRight)

        # Link slider value to label and internal variable
        def _update_slider_display(value):
            actual_value = value / multiplier if not is_int else value
            if is_int:
                value_label.setText(str(actual_value))
            else:
                value_label.setText(f"{actual_value:.2f}")
            setattr(self, var_name, actual_value) # Update internal variable
            self.request_update() # Request image update

        slider_widget.valueChanged.connect(_update_slider_display)

        # Store references to widgets for potential later access
        setattr(self, f"{var_name}_slider", slider_widget)
        setattr(self, f"{var_name}_value_label", value_label)

        frame_layout.addWidget(slider_widget)
        frame_layout.addWidget(value_label)
        self.panel_layout.addWidget(frame)

    # --- Methods (from original Tkinter app, adapted for PySide6) ---
    def _update_sharpen_state(self, state):
        self.var_sharpen_enabled = (state == QtCore.Qt.Checked)
        self.request_update()

    def browse_raw_file(self):
        # PySide6 QFileDialog
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilter(f"RAW files ({' '.join(f'*{ext}' for ext in pyrawroom.SUPPORTED_EXTS)})")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.load_raw_image(selected_files[0])

    def load_raw_image(self, path):
        self.lbl_info.setText(f"Loading: {os.path.basename(path)}...")
        QtWidgets.QApplication.processEvents() # Allow UI to update

        try:
            self.base_img_full = pyrawroom.open_raw(path)
            self.raw_path = path

            h, w, _ = self.base_img_full.shape
            scale = 1000 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            temp_pil = Image.fromarray((self.base_img_full * 255).astype(np.uint8))
            temp_pil = temp_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            self.base_img_preview = np.array(temp_pil).astype(np.float32) / 255.0

            self.lbl_info.setText(f"Loaded: {os.path.basename(path)}")
            self.btn_save.setEnabled(True)
            self.request_update()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
            self.lbl_info.setText("Error loading file")

    def load_json_edit(self):
        json_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Edit (.json)", "", "JSON Edit (*.json)")
        if not json_path:
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            original_path = data.get("original_file", "")
            final_raw_path = original_path

            if not os.path.exists(final_raw_path):
                json_dir = os.path.dirname(json_path)
                raw_filename = os.path.basename(original_path)
                potential_path = os.path.join(json_dir, raw_filename)

                if os.path.exists(potential_path):
                    final_raw_path = potential_path
                else:
                    QtWidgets.QMessageBox.critical(
                        self, "Error", f"Could not find original RAW file:\n{original_path}"
                    )
                    return

            self.load_raw_image(final_raw_path)

            settings = data.get("settings", {})
            # Update internal variables and slider positions
            self._set_slider_value("val_exposure", settings.get("exposure", 0.0))
            self._set_slider_value("val_whites", settings.get("whites", 1.0))
            self._set_slider_value("val_blacks", settings.get("blacks", 0.0))
            self._set_slider_value("val_highlights", settings.get("highlights", 0.0))
            self._set_slider_value("val_shadows", settings.get("shadows", 0.0))

            self.sharpen_checkbox.setChecked(settings.get("sharpen_enabled", False))
            self._set_slider_value("val_radius", settings.get("sharpen_radius", 2.0))
            self._set_slider_value("val_percent", settings.get("sharpen_percent", 150))

            self.request_update()
            self.lbl_info.setText(f"Loaded Edit: {os.path.basename(json_path)}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load JSON edit:\n{e}")

    def _set_slider_value(self, var_name, value):
        # This helper function sets the slider's value and updates associated label/internal var
        slider = getattr(self, f"{var_name}_slider")
        is_int = isinstance(getattr(self, var_name), int) # Check if the internal var is int
        multiplier = 1000 if not is_int else 1
        slider.setValue(int(value * multiplier))
        # The valueChanged signal will handle updating the label and internal var

    def request_update(self):
        if self.base_img_preview is None:
            return

        self.update_preview()

    def process_image(self, img_arr):
        img, _ = pyrawroom.apply_tone_map(
            img_arr,
            exposure=self.val_exposure,
            blacks=self.val_blacks,
            whites=self.val_whites,
            shadows=self.val_shadows,
            highlights=self.val_highlights,
        )

        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        if self.var_sharpen_enabled:
            pil_img = pyrawroom.sharpen_image(
                pil_img,
                radius=self.val_radius,
                percent=self.val_percent
            )
        return pil_img

    def update_preview(self):
        if self.base_img_preview is None:
            self.canvas_label.clear()
            return

        res_pil_img = self.process_image(self.base_img_preview)

        # Convert PIL Image to QPixmap
        q_image = ImageQt.ImageQt(res_pil_img)
        pixmap = QtGui.QPixmap.fromImage(q_image)

        # Scale pixmap to fit canvas_label
        if not self.canvas_label.size().isEmpty():
            scaled_pixmap = pixmap.scaled(
                self.canvas_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.canvas_label.setPixmap(scaled_pixmap)
            self.current_qpixmap = pixmap # Store original pixmap for potential full-res scaling later
        else:
            # If canvas_label has no size yet (e.g., initial startup), clear it
            self.canvas_label.clear()


    def save_file(self):
        if self.base_img_full is None:
            return

        input_dir = os.path.dirname(self.raw_path)
        base_name = os.path.splitext(os.path.basename(self.raw_path))[0]
        default_filename = f"{base_name}.jpg"

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Result",
            os.path.join(input_dir, default_filename),
            "JPEG (*.jpg);;HEIF (*.heic)"
        )
        if not out_path:
            return

        self.lbl_info.setText("Processing full res... please wait.")
        QtWidgets.QApplication.processEvents()

        try:
            final_img = self.process_image(self.base_img_full)
            pyrawroom.save_image(final_img, out_path, quality=95)

            json_path = os.path.splitext(out_path)[0] + ".json"
            settings_data = {
                "original_file": self.raw_path,
                "settings": {
                    "exposure": self.val_exposure,
                    "whites": self.val_whites,
                    "blacks": self.val_blacks,
                    "highlights": self.val_highlights,
                    "shadows": self.val_shadows,
                    "sharpen_enabled": self.var_sharpen_enabled,
                    "sharpen_radius": self.val_radius,
                    "sharpen_percent": self.val_percent,
                },
            }
            with open(json_path, "w") as f:
                json.dump(settings_data, f, indent=4)

            QtWidgets.QMessageBox.information(
                self, "Saved", f"Saved Image & Settings to:\n{os.path.dirname(out_path)}"
            )
            self.lbl_info.setText(f"Saved: {os.path.basename(out_path)}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    editor_app = PyRawEditorApp()
    editor_app.show()
    sys.exit(app.exec())
