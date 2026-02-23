from unittest.mock import MagicMock, patch

from pynegative.ui.controls.lens_controls import LensControls


def test_lens_controls_coverage(qapp):
    controls = LensControls()

    # reset_section
    controls.reset_section()

    # _populate_combos
    with patch("pynegative.io.lens_db_xml.get_instance") as mock_get:
        mock_db = MagicMock()
        mock_db.loaded = True
        mock_db.cameras = [{"maker": "Canon", "model": "EOS 5D"}]
        mock_db.get_all_lens_names.return_value = ["Canon EF 50mm"]
        mock_get.return_value = mock_db
        controls._populate_combos()

        mock_db.loaded = False
        controls._populate_combos()

    # set_lens_info
    from pynegative.io import lens_resolver

    info = {
        "exif": {"camera_make": "Canon", "camera_model": "EOS 5D"},
        "name": "Canon EF 50mm",
    }
    controls.set_lens_info(lens_resolver.ProfileSource.EMBEDDED, info)
    controls.set_lens_info(lens_resolver.ProfileSource.LENSFUN_DB, info)
    controls.set_lens_info(lens_resolver.ProfileSource.MANUAL, info)
    controls.set_lens_info(lens_resolver.ProfileSource.NONE, info)

    # manual override status
    controls.camera_combo.setCurrentIndex(1)
    controls.set_lens_info(lens_resolver.ProfileSource.EMBEDDED, info)

    # _on_selection_changed
    controls._on_selection_changed()

    # _on_auto_detect
    controls._on_auto_detect()
