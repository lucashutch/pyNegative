from unittest.mock import patch

import pytest

from pynegative.io import lens_resolver
from pynegative.io.lens_resolver import ProfileSource


@pytest.fixture
def mock_lens_metadata():
    with patch("pynegative.io.lens_resolver.lens_metadata") as mock:
        yield mock


@pytest.fixture
def mock_lens_db():
    with patch("pynegative.io.lens_resolver.lens_db_xml") as mock:
        mock.get_instance.return_value.loaded = True
        mock.get_instance.return_value.find_lens.return_value = None
        yield mock


def test_resolve_lens_profile_manual_fallback(mock_lens_metadata, mock_lens_db):
    # Setup mock EXIF data
    exif_info = {
        "camera_make": "Canon",
        "camera_model": "EOS R50",
        "lens_model": "",  # Empty lens model -> simulates no lens detection
        "focal_length": 50.0,
        "aperture": 1.8,
    }
    mock_lens_metadata.extract_lens_info.return_value = exif_info
    mock_lens_metadata.extract_embedded_correction_params.return_value = None

    # Call resolver
    source, info = lens_resolver.resolve_lens_profile("test.cr3")

    # Verify it returns NONE but includes EXIF data
    assert source == ProfileSource.NONE
    assert info is not None
    assert info["exif"] == exif_info
    assert info["name"] is None


def test_resolve_lens_profile_manual_with_unknown_lens(
    mock_lens_metadata, mock_lens_db
):
    # Setup mock EXIF data with an unknown lens name
    exif_info = {
        "camera_make": "Canon",
        "camera_model": "EOS R50",
        "lens_model": "Unknown Lens",
        "focal_length": 50.0,
        "aperture": 1.8,
    }
    mock_lens_metadata.extract_lens_info.return_value = exif_info
    mock_lens_metadata.extract_embedded_correction_params.return_value = None

    # DB returns None for match
    mock_lens_db.get_instance.return_value.find_lens.return_value = None

    # Call resolver
    source, info = lens_resolver.resolve_lens_profile("test.cr3")

    # Verify it returns MANUAL (because lens_model exists) and includes EXIF data
    assert source == ProfileSource.MANUAL
    assert info is not None
    assert info["exif"] == exif_info
    assert info["name"] == "Unknown Lens"
