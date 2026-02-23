import json
from unittest.mock import MagicMock, patch
from pynegative.cli.download_lens_database import run_download, download_file


def test_download_file(tmp_path):
    output_dir = tmp_path / "lenses"
    output_dir.mkdir()

    file_info = {
        "name": "test_lens.xml",
        "download_url": "http://example.com/test_lens.xml",
    }

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        res = download_file(file_info, output_dir)
        assert res is True
        mock_retrieve.assert_called_once()

    # Test non-xml
    res = download_file({"name": "readme.txt"}, output_dir)
    assert res is None


def test_run_download(tmp_path):
    output_dir = tmp_path / "db"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            [
                {"name": "sony.xml", "download_url": "http://sony.xml"},
                {"name": "canon.xml", "download_url": "http://canon.xml"},
            ]
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with patch("urllib.request.urlretrieve"):
            count = run_download(output_dir, quiet=True)
            assert count == 2
            assert (output_dir / ".lensdb_version").exists()


def test_run_download_fallback(tmp_path):
    output_dir = tmp_path / "db_fallback"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = Exception("API Down")

        with patch("urllib.request.urlretrieve"):
            # Fallback should trigger and try to download core_files
            count = run_download(output_dir, quiet=True)
            # core_files has 6 files
            assert count == 6
