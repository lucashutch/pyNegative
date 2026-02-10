import math
from PySide6 import QtCore
from ... import core as pynegative


class CropManager(QtCore.QObject):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

        # Throttling for rotation handle updates
        self._rotation_handle_throttle_timer = QtCore.QTimer()
        self._rotation_handle_throttle_timer.setSingleShot(True)
        self._rotation_handle_throttle_timer.setInterval(33)  # ~30fps
        self._rotation_handle_throttle_timer.timeout.connect(
            self._apply_pending_rotation
        )
        self._pending_rotation_from_handle = None

    def toggle_crop(self, enabled):
        self.editor.view.set_crop_mode(enabled)
        current_settings = self.editor.image_processor.get_current_settings()

        if enabled:
            # Enter Crop Mode
            current_crop = current_settings.get("crop")
            rotate_val = current_settings.get("rotation", 0.0)
            self.editor.view.set_rotation(rotate_val)

            if self.editor.image_processor.base_img_full is not None:
                h, w = self.editor.image_processor.base_img_full.shape[:2]
                phi = abs(math.radians(rotate_val))
                W = w * math.cos(phi) + h * math.sin(phi)
                H = w * math.sin(phi) + h * math.cos(phi)

                if current_crop:
                    c_left, c_top, c_right, c_bottom = current_crop
                    rect = QtCore.QRectF(
                        c_left * W,
                        c_top * H,
                        (c_right - c_left) * W,
                        (c_bottom - c_top) * H,
                    )
                    self.editor.view.set_crop_rect(rect)
                else:
                    self.editor.view.set_crop_rect(QtCore.QRectF(0, 0, W, H))

                # Safe bounds
                text = self.editor.editing_controls.geometry_controls.aspect_ratio_combo.currentText()
                ratio = self._text_to_ratio(text)
                safe_crop = pynegative.calculate_max_safe_crop(
                    w, h, rotate_val, aspect_ratio=ratio
                )
                c_safe_l, c_safe_t, c_safe_r, c_safe_b = safe_crop
                safe_rect = QtCore.QRectF(
                    c_safe_l * W,
                    c_safe_t * H,
                    (c_safe_r - c_safe_l) * W,
                    (c_safe_b - c_safe_t) * H,
                )
                self.editor.view.set_crop_safe_bounds(safe_rect)

            self.editor.image_processor.set_processing_params(crop=None)
            self.editor._request_update_from_view()
            self.editor.show_toast("Crop Mode Active: Drag to crop")
            QtCore.QTimer.singleShot(100, self.editor.view.fit_crop_in_view)
        else:
            # Exit Crop Mode
            rect = self.editor.view.get_crop_rect()
            scene_rect = self.editor.view.sceneRect()
            w, h = scene_rect.width(), scene_rect.height()

            c_val = None
            if w > 0 and h > 0:
                c_left, c_top, c_right, c_bottom = (
                    rect.left() / w,
                    rect.top() / h,
                    rect.right() / w,
                    rect.bottom() / h,
                )
                c_left, c_top, c_right, c_bottom = (
                    max(0.0, min(1.0, c_left)),
                    max(0.0, min(1.0, c_top)),
                    max(0.0, min(1.0, c_right)),
                    max(0.0, min(1.0, c_bottom)),
                )

                if (
                    c_left > 0.005
                    or c_top > 0.005
                    or c_right < 0.995
                    or c_bottom < 0.995
                ):
                    c_val = (c_left, c_top, c_right, c_bottom)

            self.editor.image_processor.set_processing_params(crop=c_val)
            self.editor._request_update_from_view()
            self.editor.show_toast("Crop Applied")
            self.editor._auto_save_sidecar()

    def handle_rotation_changed(self, angle: float):
        self._pending_rotation_from_handle = angle
        self.editor.editing_controls.set_slider_value("rotation", angle, silent=True)
        if not self._rotation_handle_throttle_timer.isActive():
            self._rotation_handle_throttle_timer.start()

    def _apply_pending_rotation(self):
        if self._pending_rotation_from_handle is None:
            return
        angle = self._pending_rotation_from_handle
        self.editor.image_processor.set_processing_params(rotation=angle)
        self.editor._request_update_from_view()
        self.editor.save_timer.start(1000)

        current_settings = self.editor.image_processor.get_current_settings()
        self.editor.settings_manager.schedule_undo_state(
            f"Rotate to {angle:.1f}°", current_settings
        )
        self._pending_rotation_from_handle = None

    def _text_to_ratio(self, text):
        if text == "1:1":
            return 1.0
        if text == "4:3":
            return 4.0 / 3.0
        if text == "3:2":
            return 3.0 / 2.0
        if text == "16:9":
            return 16.0 / 9.0
        return None
