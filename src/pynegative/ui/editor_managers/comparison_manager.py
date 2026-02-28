from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from enum import Enum


class ComparisonSource(Enum):
    """What is being shown on each side of the comparison split."""

    UNEDITED = "unedited"
    CURRENT = "current"
    SNAPSHOT = "snapshot"


class ComparisonManager(QtCore.QObject):
    comparisonToggled = QtCore.Signal(bool)

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.enabled = False

        # Source tracking for left / right sides
        self._left_source = ComparisonSource.UNEDITED
        self._right_source = ComparisonSource.CURRENT
        self._left_snapshot_settings: dict | None = None
        self._right_snapshot_settings: dict | None = None

    def setup_ui(self, canvas_frame, view):
        self.canvas_frame = canvas_frame
        self.view = view

        # Comparison Toggle Button
        self.comparison_btn = QtWidgets.QToolButton(canvas_frame)
        self.comparison_btn.setFixedSize(30, 30)
        self.comparison_btn.setCheckable(True)
        self.comparison_btn.setToolTip("Compare (U)")
        self.comparison_btn.setIcon(self._create_comparison_icon())
        self.comparison_btn.setStyleSheet("""
            QToolButton {
                background-color: rgba(0, 0, 0, 128);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 4px;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: rgba(0, 0, 0, 180);
                border-color: rgba(255, 255, 255, 80);
            }
            QToolButton:checked {
                background-color: rgba(138, 43, 226, 180);
                border-color: rgba(138, 43, 226, 200);
            }
        """)
        self.comparison_btn.clicked.connect(self.toggle_comparison)

        # Comparison Handle
        from ..widgets import ComparisonHandle

        self.comparison_handle = ComparisonHandle(canvas_frame)
        self.comparison_handle.dragged.connect(self._on_handle_dragged)
        self.comparison_handle.hide()

        # Comparison Drawing Layer
        from ..widgets import ComparisonOverlay

        self.comparison_overlay = ComparisonOverlay(canvas_frame)
        self.comparison_overlay.setView(view)
        self.comparison_overlay.raise_()

    def _create_comparison_icon(self):
        pixmap = QtGui.QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        left_color = QtGui.QColor(100, 100, 100)
        right_color = QtGui.QColor(150, 150, 150)
        painter.setBrush(left_color)
        painter.setPen(Qt.NoPen)
        painter.drawRect(2, 4, 8, 16)
        painter.setBrush(right_color)
        painter.drawRect(14, 4, 8, 16)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        painter.drawLine(12, 2, 12, 22)
        painter.end()
        return QtGui.QIcon(pixmap)

    def toggle_comparison(self):
        enabled = self.comparison_btn.isChecked()
        self.enabled = enabled
        self.comparison_overlay.setComparisonActive(enabled)

        if enabled:
            self.comparison_handle.show()
            self.update_handle_position()
            # Reset to default sources on fresh enable
            self._left_source = ComparisonSource.UNEDITED
            self._right_source = ComparisonSource.CURRENT
            self._left_snapshot_settings = None
            self._right_snapshot_settings = None
            unedited_pixmap = self.editor.image_processor.get_unedited_pixmap()
            self.comparison_overlay.setUneditedPixmap(unedited_pixmap)
            if self.view._bg_item and self.view._bg_item.pixmap():
                self.comparison_overlay.setEditedPixmap(self.view._bg_item.pixmap())
            self.editor.show_toast("Comparison enabled")
        else:
            self.comparison_handle.hide()
            self.editor.show_toast("Comparison disabled")

        self.comparisonToggled.emit(enabled)

    def update_handle_position(self):
        if hasattr(self, "comparison_handle") and hasattr(self, "comparison_overlay"):
            split_pos = self.comparison_overlay._split_position
            ov_x = self.comparison_overlay.x()
            ov_w = self.comparison_overlay.width()
            split_x = ov_x + int(ov_w * split_pos)
            hx = split_x - (self.comparison_handle.width() / 2)
            hy = (self.canvas_frame.height() - self.comparison_handle.height()) / 2
            self.comparison_handle.move(hx, hy)

    def _on_handle_dragged(self, global_x):
        local_pos = self.canvas_frame.mapFromGlobal(QtCore.QPointF(global_x, 0))
        split_pos = max(0.0, min(1.0, local_pos.x() / self.canvas_frame.width()))
        self.comparison_overlay.setSplitPosition(split_pos)
        self.update_handle_position()

    def update_pixmaps(self, unedited=None, edited=None):
        if self.enabled and self.comparison_overlay:
            if unedited and self._left_source == ComparisonSource.UNEDITED:
                self.comparison_overlay.setUneditedPixmap(unedited)
            if edited and self._right_source == ComparisonSource.CURRENT:
                self.comparison_overlay.setEditedPixmap(edited)

    # ------------------------------------------------------------------
    # Snapshot comparison
    # ------------------------------------------------------------------

    def set_left_snapshot(self, settings: dict):
        """Render *settings* and use the result as the left comparison image."""
        self._left_source = ComparisonSource.SNAPSHOT
        self._left_snapshot_settings = settings
        pixmap = self.editor.image_processor.render_snapshot_pixmap(settings)
        if not pixmap.isNull():
            self.comparison_overlay.setUneditedPixmap(pixmap)
            self.editor.show_toast("Left comparison: snapshot")

    def set_right_snapshot(self, settings: dict):
        """Render *settings* and use the result as the right comparison image."""
        self._right_source = ComparisonSource.SNAPSHOT
        self._right_snapshot_settings = settings
        pixmap = self.editor.image_processor.render_snapshot_pixmap(settings)
        if not pixmap.isNull():
            self.comparison_overlay.setEditedPixmap(pixmap)
            self.editor.show_toast("Right comparison: snapshot")

    def reset_sources(self):
        """Reset both sides to the default unedited / current pair."""
        self._left_source = ComparisonSource.UNEDITED
        self._right_source = ComparisonSource.CURRENT
        self._left_snapshot_settings = None
        self._right_snapshot_settings = None
        if self.enabled:
            unedited = self.editor.image_processor.get_unedited_pixmap()
            self.comparison_overlay.setUneditedPixmap(unedited)
            if self.view._bg_item and self.view._bg_item.pixmap():
                self.comparison_overlay.setEditedPixmap(self.view._bg_item.pixmap())
