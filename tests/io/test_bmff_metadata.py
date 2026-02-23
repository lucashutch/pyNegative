from unittest.mock import patch
from pynegative.io.bmff_metadata import extract_bmff_metadata, get_exposure_info


def test_extract_bmff_metadata(tmp_path):
    path = tmp_path / "test.cr3"
    # Create a fake file with a TIFF header "II*\0" at some offset
    content = b"random data" * 10
    content += b"II\x2a\x00" + b"fake tiff data"
    path.write_bytes(content)

    with patch("exifread.process_file") as mock_exif:
        mock_exif.return_value = {"EXIF ISOSpeedRatings": 400}

        tags = extract_bmff_metadata(path)
        assert tags["EXIF ISOSpeedRatings"] == 400


def test_get_exposure_info():
    tags = {
        "EXIF ISOSpeedRatings": 100,
        "EXIF ExposureTime": "1/500",
        "EXIF FNumber": "4",
        "EXIF FocalLength": "24",
        "Image Make": "Sony",
        "Image Model": "A7RIII",
        "EXIF LensModel": "FE 24-70mm",
    }
    info = get_exposure_info(tags)
    assert info["iso"] == 100
    assert info["shutter_speed"] == "1/500"
    assert info["aperture"] == "4"
    assert info["focal_length"] == "24"
    assert info["camera_make"] == "Sony"
    assert info["camera_model"] == "A7RIII"
    assert info["lens_model"] == "FE 24-70mm"


def test_extract_bmff_metadata_error(tmp_path):
    path = tmp_path / "error.cr3"
    path.write_bytes(b"no header here")
    # Should just return empty dict
    assert extract_bmff_metadata(path) == {}
