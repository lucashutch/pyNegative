"""Tests for Phase 5: Carousel comparison context menu actions."""

from unittest.mock import MagicMock, patch

from PySide6 import QtCore

from pynegative.ui.editor_managers.context_menu_manager import ContextMenuManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context_menu_manager():
    editor = MagicMock()
    editor.comparison_manager = MagicMock()
    editor.comparison_manager.enabled = False
    editor.image_processor = MagicMock()
    editor.settings_manager = MagicMock()
    editor.settings_manager.has_clipboard_content.return_value = False
    editor.editing_controls = MagicMock()
    return ContextMenuManager(editor)


# ---------------------------------------------------------------------------
# Carousel comparison actions
# ---------------------------------------------------------------------------


class TestCarouselComparisonActions:
    def test_comparison_actions_shown_when_enabled(self):
        """When comparison is enabled, the carousel context menu should include
        'Set as Left/Right Comparison Image' actions."""
        mgr = _make_context_menu_manager()
        mgr.editor.comparison_manager.enabled = True

        carousel_widget = MagicMock()
        carousel_widget.get_selected_paths.return_value = ["/img/a.dng"]
        carousel_widget.mapToGlobal.return_value = QtCore.QPoint(0, 0)

        with patch(
            "pynegative.ui.editor_managers.context_menu_manager.QtWidgets.QMenu"
        ) as MockMenu:
            menu_instance = MagicMock()
            MockMenu.return_value = menu_instance
            menu_instance.sizeHint.return_value = QtCore.QSize(200, 120)
            added_actions = []

            def track_action(text):
                action = MagicMock()
                action.text = text
                added_actions.append(text)
                return action

            menu_instance.addAction.side_effect = track_action

            mgr.handle_carousel_context_menu(
                "carousel", (QtCore.QPoint(5, 5), "/img/a.dng", carousel_widget)
            )

            assert "Set as Left Comparison Image" in added_actions
            assert "Set as Right Comparison Image" in added_actions

    def test_comparison_actions_shown_when_disabled(self):
        """Comparison actions are always shown and can auto-enable compare mode."""
        mgr = _make_context_menu_manager()
        mgr.editor.comparison_manager.enabled = False

        carousel_widget = MagicMock()
        carousel_widget.get_selected_paths.return_value = ["/img/a.dng"]
        carousel_widget.mapToGlobal.return_value = QtCore.QPoint(0, 0)

        with patch(
            "pynegative.ui.editor_managers.context_menu_manager.QtWidgets.QMenu"
        ) as MockMenu:
            menu_instance = MagicMock()
            MockMenu.return_value = menu_instance
            menu_instance.sizeHint.return_value = QtCore.QSize(200, 120)
            added_actions = []

            def track_action(text):
                action = MagicMock()
                action.text = text
                added_actions.append(text)
                return action

            menu_instance.addAction.side_effect = track_action

            mgr.handle_carousel_context_menu(
                "carousel", (QtCore.QPoint(5, 5), "/img/a.dng", carousel_widget)
            )

            assert "Set as Left Comparison Image" in added_actions
            assert "Set as Right Comparison Image" in added_actions


class TestSetCarouselComparison:
    @patch("pynegative.ui.editor_managers.context_menu_manager.pynegative")
    def test_set_left_with_sidecar(self, mock_core):
        mock_core.load_sidecar.return_value = {"exposure": 1.5, "contrast": 0.5}
        mgr = _make_context_menu_manager()
        mgr.editor.raw_path = "/img/a.dng"

        mgr._set_carousel_comparison("/img/a.dng", "left")

        mock_core.load_sidecar.assert_called_once_with("/img/a.dng")
        mgr.editor.comparison_manager.set_left_snapshot.assert_called_once_with(
            {"exposure": 1.5, "contrast": 0.5}
        )

    @patch("pynegative.ui.editor_managers.context_menu_manager.pynegative")
    def test_set_right_with_sidecar(self, mock_core):
        mock_core.load_sidecar.return_value = {"exposure": -0.5}
        mgr = _make_context_menu_manager()
        mgr.editor.raw_path = "/img/b.dng"

        mgr._set_carousel_comparison("/img/b.dng", "right")

        mock_core.load_sidecar.assert_called_once_with("/img/b.dng")
        mgr.editor.comparison_manager.set_right_snapshot.assert_called_once_with(
            {"exposure": -0.5}
        )

    @patch("pynegative.ui.editor_managers.context_menu_manager.pynegative")
    def test_set_left_no_sidecar(self, mock_core):
        """When no sidecar exists, use empty settings (defaults)."""
        mock_core.load_sidecar.return_value = None
        mgr = _make_context_menu_manager()
        mgr.editor.raw_path = "/img/c.dng"

        mgr._set_carousel_comparison("/img/c.dng", "left")

        mgr.editor.comparison_manager.set_left_snapshot.assert_called_once_with({})

    @patch("pynegative.ui.editor_managers.context_menu_manager.pynegative")
    def test_set_right_no_sidecar(self, mock_core):
        mock_core.load_sidecar.return_value = None
        mgr = _make_context_menu_manager()
        mgr.editor.raw_path = "/img/d.dng"

        mgr._set_carousel_comparison("/img/d.dng", "right")

        mgr.editor.comparison_manager.set_right_snapshot.assert_called_once_with({})

    @patch("pynegative.ui.editor_managers.context_menu_manager.pynegative")
    def test_auto_enables_comparison_before_setting_snapshot(self, mock_core):
        mock_core.load_sidecar.return_value = {"exposure": 0.25}
        mgr = _make_context_menu_manager()
        mgr.editor.comparison_manager.enabled = False
        mgr.editor.raw_path = "/img/e.dng"

        mgr._set_carousel_comparison("/img/e.dng", "left")

        mgr.editor.comparison_manager.comparison_btn.setChecked.assert_called_once_with(
            True
        )
        mgr.editor.comparison_manager.toggle_comparison.assert_called_once()

    @patch(
        "pynegative.ui.editor_managers.context_menu_manager.ContextMenuManager._load_comparison_pixmap"
    )
    @patch("pynegative.ui.editor_managers.context_menu_manager.pynegative")
    def test_set_left_cross_photo_uses_pixmap(self, mock_core, mock_load_pixmap):
        mock_core.load_sidecar.return_value = {"exposure": 1.0}
        mock_pixmap = MagicMock()
        mock_pixmap.isNull.return_value = False
        mock_load_pixmap.return_value = mock_pixmap
        mgr = _make_context_menu_manager()
        mgr.editor.raw_path = "/img/current.dng"

        mgr._set_carousel_comparison("/img/other.dng", "left")

        mgr.editor.comparison_manager.set_left_pixmap.assert_called_once()
        mgr.editor.comparison_manager.set_left_snapshot.assert_not_called()
        mock_core.load_sidecar.assert_not_called()

    @patch(
        "pynegative.ui.editor_managers.context_menu_manager.ContextMenuManager._load_comparison_pixmap"
    )
    @patch("pynegative.ui.editor_managers.context_menu_manager.pynegative")
    def test_set_right_cross_photo_uses_pixmap(self, mock_core, mock_load_pixmap):
        mock_core.load_sidecar.return_value = {"contrast": 0.2}
        mock_pixmap = MagicMock()
        mock_pixmap.isNull.return_value = False
        mock_load_pixmap.return_value = mock_pixmap
        mgr = _make_context_menu_manager()
        mgr.editor.raw_path = "/img/current.dng"

        mgr._set_carousel_comparison("/img/other2.dng", "right")

        mgr.editor.comparison_manager.set_right_pixmap.assert_called_once()
        mgr.editor.comparison_manager.set_right_snapshot.assert_not_called()
        mock_core.load_sidecar.assert_not_called()
