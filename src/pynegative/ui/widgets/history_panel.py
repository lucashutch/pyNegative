"""History panel widget showing edit snapshots/versions."""

import logging
from datetime import datetime

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)


class SnapshotItemWidget(QtWidgets.QWidget):
    """Custom widget for a single snapshot entry in the list."""

    def __init__(self, snapshot: dict, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Icon column
        icon_label = QtWidgets.QLabel()
        if snapshot.get("is_tagged"):
            icon_label.setText("★")
            icon_label.setStyleSheet("color: #FFD700; font-size: 14px;")
            icon_label.setToolTip("Tagged version")
        elif snapshot.get("is_auto"):
            icon_label.setText("⏱")
            icon_label.setStyleSheet("color: #888; font-size: 12px;")
            icon_label.setToolTip("Auto-save")
        else:
            icon_label.setText("💾")
            icon_label.setStyleSheet("font-size: 12px;")
            icon_label.setToolTip("Manual save")
        icon_label.setFixedWidth(20)
        layout.addWidget(icon_label)

        # Text column
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        timestamp = snapshot.get("timestamp", 0)
        time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        label = snapshot.get("label")
        if label:
            name_label = QtWidgets.QLabel(label)
            name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
            text_layout.addWidget(name_label)

        time_label = QtWidgets.QLabel(time_str)
        time_label.setStyleSheet("color: #aaa; font-size: 11px;")
        text_layout.addWidget(time_label)

        # Type indicator
        if snapshot.get("is_auto"):
            type_text = "Auto-save"
        elif snapshot.get("is_tagged"):
            type_text = "Tagged version"
        else:
            type_text = "Manual save"
        type_label = QtWidgets.QLabel(type_text)
        type_label.setStyleSheet("color: #777; font-size: 10px;")
        text_layout.addWidget(type_label)

        layout.addLayout(text_layout, 1)


class HistoryPanel(QtWidgets.QWidget):
    """Panel displaying the snapshot/version history for an image."""

    snapshotSelected = QtCore.Signal(str)  # snapshot_id
    restoreRequested = QtCore.Signal(str)  # snapshot_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HistoryPanel")

        self._snapshots: list[dict] = []
        self._selected_id: str | None = None
        self._previewing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Title
        title = QtWidgets.QLabel("Edit History")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Action bar
        action_bar = QtWidgets.QHBoxLayout()
        action_bar.setSpacing(4)

        self.restore_btn = QtWidgets.QPushButton("Restore")
        self.restore_btn.setEnabled(False)
        self.restore_btn.setToolTip("Restore the selected snapshot as current edit")
        self.restore_btn.clicked.connect(self._on_restore_clicked)
        action_bar.addWidget(self.restore_btn)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Cancel preview and return to current edit")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        action_bar.addWidget(self.cancel_btn)

        action_bar.addStretch()
        layout.addLayout(action_bar)

        # Snapshot list
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.currentItemChanged.connect(self._on_item_changed)
        self.list_widget.setSpacing(2)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.list_widget)

        # Empty state
        self._empty_label = QtWidgets.QLabel(
            "No history yet.\nSnapshots are created automatically\n"
            "every 60 seconds, or use Ctrl+S."
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #777; font-size: 11px;")
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        # Start in empty state: show label, hide list
        self.list_widget.hide()

    def set_snapshots(self, snapshots: list[dict]):
        """Populate the panel with snapshot entries (newest first)."""
        self._snapshots = snapshots
        self._selected_id = None
        self._previewing = False
        self.restore_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.list_widget.clear()

        if not snapshots:
            self._empty_label.show()
            self.list_widget.hide()
            return

        self._empty_label.hide()
        self.list_widget.show()

        for snap in snapshots:
            item_widget = SnapshotItemWidget(snap)
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, snap["id"])
            item.setSizeHint(item_widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, item_widget)

    def set_previewing(self, previewing: bool):
        """Update button states for preview mode."""
        self._previewing = previewing
        self.restore_btn.setEnabled(previewing)
        self.cancel_btn.setEnabled(previewing)

    def get_snapshot_by_id(self, snapshot_id: str) -> dict | None:
        """Look up a snapshot by id from the current list."""
        for snap in self._snapshots:
            if snap.get("id") == snapshot_id:
                return snap
        return None

    def clear(self):
        """Clear the panel."""
        self._snapshots = []
        self._selected_id = None
        self._previewing = False
        self.list_widget.clear()
        self.restore_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self._empty_label.show()
        self.list_widget.hide()

    def _on_item_changed(self, current, previous):
        if current is None:
            self._selected_id = None
            return
        snapshot_id = current.data(Qt.UserRole)
        self._selected_id = snapshot_id
        self.snapshotSelected.emit(snapshot_id)

    def _on_restore_clicked(self):
        if self._selected_id:
            self.restoreRequested.emit(self._selected_id)

    def _on_cancel_clicked(self):
        self._previewing = False
        self.restore_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.list_widget.clearSelection()
        # Emit empty string to signal cancellation
        self.snapshotSelected.emit("")
