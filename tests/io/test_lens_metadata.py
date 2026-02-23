from unittest.mock import MagicMock, patch
from pynegative.io.lens_metadata import extract_lens_info


def test_extract_lens_info_rawpy(tmp_path):
    path = tmp_path / "test.ARW"
    path.write_text("fake")

    with patch("rawpy.imread") as mock_imread:
        mock_raw = MagicMock()
        mock_raw.lens_config.lens = b"FE 85mm F1.8"
        mock_raw.lens_config.maker = b"Sony"
        mock_raw.idata.make = b"Sony"
        mock_raw.idata.model = b"ILCE-7M3"
        mock_imread.return_value.__enter__.return_value = mock_raw

        info = extract_lens_info(path)
        assert info["lens_model"] == "Sony FE 85mm F1.8"
        assert info["camera_make"] == "Sony"
        assert info["camera_model"] == "ILCE-7M3"


def test_extract_lens_info_cr3(tmp_path):
    path = tmp_path / "test.cr3"
    path.write_text("fake")

    with patch("pynegative.io.bmff_metadata.extract_bmff_metadata") as mock_bmff:
        mock_bmff.return_value = {
            "EXIF ISOSpeedRatings": 100,
            "EXIF ExposureTime": "1/200",
            "EXIF FNumber": "2.8",
            "EXIF FocalLength": "35",
            "Image Make": "Canon",
            "Image Model": "Canon EOS R5",
            "EXIF LensModel": "RF35mm F1.8 MACRO IS STM",
        }
        with patch("rawpy.imread") as mock_imread:
            mock_imread.side_effect = Exception("failed")

            info = extract_lens_info(path)
            assert info["camera_make"] == "Canon"
            assert info["lens_model"] == "RF35mm F1.8 MACRO IS STM"
            assert info["focal_length"] == 35.0
            assert info["aperture"] == 2.8


def test_extract_lens_info_exifread(tmp_path):
    path = tmp_path / "test.jpg"
    path.write_text("fake")

    with patch("exifread.process_file") as mock_exif:
        mock_exif.return_value = {
            "Image Make": "Nikon",
            "Image Model": "Nikon D850",
            "EXIF LensModel": "50mm f/1.8",
            "EXIF FocalLength": "50",
            "EXIF FNumber": "1.8",
        }
        # mock open to avoid actual file read
        with patch("builtins.open", MagicMock()):
            with patch("rawpy.imread") as mock_imread:
                mock_imread.side_effect = Exception("failed")

                info = extract_lens_info(path)
                assert info["camera_make"] == "Nikon"
                assert info["lens_model"] == "50mm f/1.8"
                assert info["focal_length"] == 50.0


def test_to_float_helper():
    # We can test internal to_float via extract_lens_info and mocked tags
    pass  # covered by cases above
