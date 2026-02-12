from PySide6 import QtWidgets, QtCore


class FloatingUIManager(QtCore.QObject):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def setup_ui(self, canvas_frame):
        self.canvas_frame = canvas_frame

        # Performance metric label
        self.perf_label = QtWidgets.QLabel(canvas_frame)
        self.perf_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 128); color: white; padding: 4px; border-radius: 4px;"
        )
        self.perf_label.hide()

        # Toast widget for notifications
        from ..widgets import ToastWidget

        self.toast = ToastWidget(canvas_frame)

    def reposition(self, view, zoom_ctrl, preview_rating_widget, comparison_manager):
        if not view:
            return

        cw = self.canvas_frame.width()
        view_pos = view.mapTo(self.canvas_frame, QtCore.QPoint(0, 0))
        vx, vy = view_pos.x(), view_pos.y()
        vw, vh = view.width(), view.height()

        # 1. Zoom controls
        if zoom_ctrl:
            zx = vx + vw - zoom_ctrl.width() - 10
            zy = vy + vh - zoom_ctrl.height() - 10
            zoom_ctrl.move(zx, zy)
            zoom_ctrl.show()
            zoom_ctrl.raise_()

        # 2. Preview rating
        if preview_rating_widget:
            prx = vx + 10
            pry = vy + vh - preview_rating_widget.height() - 5
            preview_rating_widget.move(prx, pry)

        # 3. Performance label
        if self.perf_label:
            px = vx + 20
            py = (
                vy
                + vh
                - (preview_rating_widget.height() if preview_rating_widget else 0)
                - self.perf_label.height()
                - 30
            )
            self.perf_label.move(px, py)

        # 4. Comparison Button
        if comparison_manager and hasattr(comparison_manager, "comparison_btn"):
            bx = vx + vw - comparison_manager.comparison_btn.width() - 10
            by = (
                vy
                + vh
                - (zoom_ctrl.height() if zoom_ctrl else 0)
                - comparison_manager.comparison_btn.height()
                - 15
            )
            comparison_manager.comparison_btn.move(bx, by)
            comparison_manager.comparison_btn.show()
            comparison_manager.comparison_btn.raise_()

        # 5. Comparison Overlay
        if comparison_manager and hasattr(comparison_manager, "comparison_overlay"):
            comparison_manager.comparison_overlay.setGeometry(vx, vy, vw, vh)
            comparison_manager.comparison_overlay.raise_()
            comparison_manager.update_handle_position()

        # 6. Toast Widget
        if self.toast:
            self.toast.move(
                (cw - self.toast.width()) // 2, (vh - self.toast.height()) // 2 + vy
            )

    def show_toast(self, message):
        self.toast.show_message(message)

    def set_perf_text(self, text):
        self.perf_label.setText(text)

    def toggle_perf_visibility(self):
        is_visible = not self.perf_label.isVisible()
        self.perf_label.setVisible(is_visible)
        return is_visible
