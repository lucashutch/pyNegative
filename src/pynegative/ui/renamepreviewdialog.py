from pathlib import Path
from typing import List, Optional, Tuple
from PySide6 import QtWidgets, QtGui


class RenamePreviewDialog(QtWidgets.QDialog):
    """Modal dialog showing preview of file rename operations."""

    def __init__(self, parent=None):
        """Initialize the dialog UI.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Rename Preview")
        self.setModal(True)
        self.resize(700, 500)

        self._preview_data = []
        self._confirmed = False

        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title_label = QtWidgets.QLabel("Export Rename Preview")
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Description
        desc_label = QtWidgets.QLabel(
            "Review how your files will be renamed before export:"
        )
        desc_label.setStyleSheet("color: #666;")
        layout.addWidget(desc_label)

        # Table widget
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Original Name", "New Name", "Status"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

        # Summary label
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.summary_label)

        # Conflict warning
        self.conflict_label = QtWidgets.QLabel()
        self.conflict_label.setStyleSheet("color: #e74c3c;")
        layout.addWidget(self.conflict_label)

        # Button box - only OK and Cancel buttons
        button_box = QtWidgets.QDialogButtonBox()
        button_box.addButton(QtWidgets.QDialogButtonBox.Ok)
        button_box.addButton(QtWidgets.QDialogButtonBox.Cancel)

        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)

    def set_preview_data(self, preview_data: List[Tuple[str, str, Optional[str]]]):
        """Set the preview data and populate the table.

        Args:
            preview_data: List of tuples (original_name, new_name, warning)
        """
        self._preview_data = preview_data
        self._populate_table()
        self._update_summary()

    def _populate_table(self):
        """Populate the table with preview data."""
        self.table.setRowCount(len(self._preview_data))

        for row, (original, new_name, warning) in enumerate(self._preview_data):
            # Original name
            original_item = QtWidgets.QTableWidgetItem(original)
            self.table.setItem(row, 0, original_item)

            # New name
            new_item = QtWidgets.QTableWidgetItem(new_name)
            self.table.setItem(row, 1, new_item)

            # Status
            if warning:
                status_item = QtWidgets.QTableWidgetItem(f"⚠️ {warning}")
                status_item.setForeground(QtGui.QColor("#e74c3c"))
                # Color the row red
                original_item.setBackground(QtGui.QColor("#ffebee"))
                new_item.setBackground(QtGui.QColor("#ffebee"))
                status_item.setBackground(QtGui.QColor("#ffebee"))
            else:
                status_item = QtWidgets.QTableWidgetItem("✓ Ready")
                status_item.setForeground(QtGui.QColor("#27ae60"))

            self.table.setItem(row, 2, status_item)

        self.table.resizeRowsToContents()

    def _update_summary(self):
        """Update the summary label with counts."""
        total = len(self._preview_data)
        conflicts = sum(1 for _, _, warning in self._preview_data if warning)
        ready = total - conflicts

        self.summary_label.setText(
            f"Total: {total} files | Ready: {ready} | Conflicts: {conflicts}"
        )

        if conflicts > 0:
            self.conflict_label.setText(
                f"⚠️ {conflicts} file(s) will be skipped due to conflicts. "
                "Review the table above."
            )
        else:
            self.conflict_label.clear()

    def _on_accept(self):
        """Handle accept button click."""
        self._confirmed = True
        self.accept()

    def is_confirmed(self) -> bool:
        """Return whether the user confirmed the preview."""
        return self._confirmed

    def get_rename_mapping(self, source_files: List[Path]) -> dict:
        """Create a mapping from source file to target filename.

        Only includes files without warnings.

        Args:
            source_files: List of source file paths (should match preview_data order)

        Returns:
            Dictionary mapping source Path to target filename string
        """
        mapping = {}
        for i, file_path in enumerate(source_files):
            if i < len(self._preview_data):
                _, new_name, warning = self._preview_data[i]
                if not warning and new_name:
                    mapping[file_path] = new_name
        return mapping
