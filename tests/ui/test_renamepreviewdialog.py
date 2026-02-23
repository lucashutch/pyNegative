from pathlib import Path
from pynegative.ui.renamepreviewdialog import RenamePreviewDialog


def test_rename_preview_dialog(qtbot):
    dialog = RenamePreviewDialog()
    qtbot.addWidget(dialog)

    preview_data = [
        ("original1.ARW", "new1.jpg", None),
        ("original2.ARW", "new2.jpg", "Duplicate name generated"),
    ]

    dialog.set_preview_data(preview_data)

    # Check table population
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "original1.ARW"
    assert "Duplicate" in dialog.table.item(1, 2).text()

    # Check summary
    assert "Ready: 1" in dialog.summary_label.text()
    assert "Conflicts: 1" in dialog.summary_label.text()

    # Check mapping
    source_files = [Path("path/to/original1.ARW"), Path("path/to/original2.ARW")]
    mapping = dialog.get_rename_mapping(source_files)
    assert len(mapping) == 1
    assert mapping[Path("path/to/original1.ARW")] == "new1.jpg"

    # Test confirmation
    dialog._on_accept()
    assert dialog.is_confirmed() is True
