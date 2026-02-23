from PySide6 import QtWidgets
from ..settings_constants import SETTINGS_GROUPS


class SelectivePasteDialog(QtWidgets.QDialog):
    """Dialog to choose which settings to paste from the clipboard."""

    # Session memory for last selected keys
    _last_selected_keys = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paste Settings Selective")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._checkboxes = {}  # key: checkbox
        self._group_checkboxes = {}  # group_name: checkbox

        self._init_ui()

        # Restore last selection if available, otherwise select all
        if SelectivePasteDialog._last_selected_keys is not None:
            self._apply_selection(SelectivePasteDialog._last_selected_keys)
        else:
            self._select_all(True)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Scroll area for many settings
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_content = QtWidgets.QWidget()

        # Grid layout for group boxes (2 columns)
        scroll_layout = QtWidgets.QGridLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        for i, (group_name, items) in enumerate(SETTINGS_GROUPS.items()):
            group_box = QtWidgets.QGroupBox(group_name)
            group_box.setCheckable(True)
            group_box.setChecked(True)

            # Inner grid layout for items (2 columns)
            group_layout = QtWidgets.QGridLayout(group_box)
            group_layout.setSpacing(4)

            self._group_checkboxes[group_name] = group_box
            group_box.toggled.connect(
                lambda checked, gn=group_name: self._on_group_toggled(gn, checked)
            )

            for j, (key, label) in enumerate(items):
                cb = QtWidgets.QCheckBox(label)
                cb.setChecked(True)

                # Arrange in 2 columns inside the group box
                group_layout.addWidget(cb, j // 2, j % 2)

                self._checkboxes[key] = cb
                cb.toggled.connect(
                    lambda checked, gn=group_name: self._on_item_toggled(gn)
                )

            # Arrange group boxes in 2 columns in the main scroll area
            scroll_layout.addWidget(group_box, i // 2, i % 2)

        # Add a stretch to the bottom of the grid
        # We can't easily add stretch to QGridLayout like QVBoxLayout
        # but we can set row stretch for the last row
        last_row = (len(SETTINGS_GROUPS) - 1) // 2
        scroll_layout.setRowStretch(last_row + 1, 1)

        # Global buttons
        btn_layout = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_none_btn = QtWidgets.QPushButton("Select None")
        select_all_btn.clicked.connect(lambda: self._select_all(True))
        select_none_btn.clicked.connect(lambda: self._select_all(False))
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(select_none_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Dialog buttons
        dialog_btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.accepted.connect(self.accept)
        dialog_btns.rejected.connect(self.reject)
        layout.addWidget(dialog_btns)

        # Style adjustment
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                margin-top: 1.2em;
                border: 1px solid #444;
                border-radius: 4px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #e5e5e5;
            }
            QGroupBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #404040;
                background-color: #242424;
            }
            QGroupBox::indicator:hover {
                border-color: #6366f1;
            }
            QGroupBox::indicator:checked {
                background-color: #6366f1;
                border-color: #6366f1;
            }
            QGroupBox::indicator:checked:hover {
                background-color: #818cf8;
                border-color: #818cf8;
            }
        """)

    def _on_group_toggled(self, group_name, checked):
        # When group is toggled, update all children
        for key, label in SETTINGS_GROUPS[group_name]:
            cb = self._checkboxes[key]
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _on_item_toggled(self, group_name):
        # If all items in a group are unchecked, uncheck group
        # If any item is checked, check group (or keep partially checked? GroupBox only has 2 states easily)
        group_cb = self._group_checkboxes[group_name]
        any_checked = any(
            self._checkboxes[key].isChecked() for key, _ in SETTINGS_GROUPS[group_name]
        )
        group_cb.blockSignals(True)
        group_cb.setChecked(any_checked)
        group_cb.blockSignals(False)

    def _select_all(self, checked):
        for group_cb in self._group_checkboxes.values():
            group_cb.setChecked(checked)
        for cb in self._checkboxes.values():
            cb.setChecked(checked)

    def _apply_selection(self, keys):
        # Start by unselecting everything
        self._select_all(False)
        for key in keys:
            if key in self._checkboxes:
                self._checkboxes[key].setChecked(True)

        # Update group checkboxes based on children
        for group_name in SETTINGS_GROUPS:
            any_checked = any(
                self._checkboxes[key].isChecked()
                for key, _ in SETTINGS_GROUPS[group_name]
            )
            self._group_checkboxes[group_name].setChecked(any_checked)

    def get_selected_keys(self):
        selected = []
        for key, cb in self._checkboxes.items():
            if cb.isChecked():
                selected.append(key)
        return selected

    def accept(self):
        # Save selection for session
        SelectivePasteDialog._last_selected_keys = self.get_selected_keys()
        super().accept()
