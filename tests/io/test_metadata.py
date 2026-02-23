from unittest.mock import MagicMock, patch
from pathlib import Path
from pynegative.io.metadata import get_exif_capture_date, format_date


def test_format_date():
    assert format_date(1705320000) == "2024-01-15"


def test_get_exif_capture_date_standard_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img_path.write_text("fake image content")

    with patch("PIL.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_exif = {36867: "2023:05:20 10:00:00"}
        mock_img.getexif.return_value = mock_exif
        mock_open.return_value.__enter__.return_value = mock_img

        date = get_exif_capture_date(img_path)
        assert date == "2023-05-20"


def test_get_exif_capture_date_rawpy(tmp_path):
    raw_path = tmp_path / "test.ARW"
    raw_path.write_text("fake raw content")

    with patch("rawpy.imread") as mock_imread:
        mock_raw = MagicMock()
        mock_raw.extract_exif.return_value = b"DateTimeOriginal 2024:02:10 12:00:00"
        mock_imread.return_value.__enter__.return_value = mock_raw

        date = get_exif_capture_date(raw_path)
        assert date == "2024-02-10"


def test_get_exif_capture_date_fallback(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("fake content")

    # Mock everything before the fallback
    with patch("PIL.Image.open", side_effect=Exception("no PIL")), patch(
        "rawpy.imread"
    ) as mock_imread:
        mock_raw = MagicMock()
        mock_raw.extract_exif.side_effect = Exception("no exif")
        mock_imread.return_value.__enter__.return_value = mock_raw

        # Mocking stat to control mtime
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_mtime = 1705320000  # 2024-01-15

            date = get_exif_capture_date(file_path)
            assert date == "2024-01-15"


def test_get_exif_capture_date_error():
    assert get_exif_capture_date("non_existent_file.xyz") is None
