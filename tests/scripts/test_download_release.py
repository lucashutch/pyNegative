#!/usr/bin/env python3
"""Unit tests for the download_release.py script."""

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from download_release import (
    check_update_available,
    download_and_extract,
    get_latest_version,
    load_version,
    main,
    save_version,
)


def create_mock_urlopen_response(data: bytes):
    """Create a mock response object that supports context manager protocol."""
    mock_response = MagicMock()
    mock_response.read.return_value = data
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


class TestGetLatestVersion:
    """Tests for get_latest_version function."""

    def test_successful_api_call(self):
        """Test successful GitHub API call returns tag name."""
        mock_response = create_mock_urlopen_response(
            json.dumps({"tag_name": "v1.2.3"}).encode()
        )

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = get_latest_version("owner/repo", verbose=False)
            assert result == "v1.2.3"

    def test_api_returns_no_tag(self):
        """Test when API returns no tag_name, falls back to 'main'."""
        mock_response = create_mock_urlopen_response(json.dumps({}).encode())

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = get_latest_version("owner/repo", verbose=False)
            assert result == "main"

    def test_api_error_falls_back_to_main(self):
        """Test when API call fails, falls back to 'main'."""
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = get_latest_version("owner/repo", verbose=False)
            assert result == "main"

    def test_verbose_output(self, capsys):
        """Test verbose mode prints progress messages."""
        mock_response = create_mock_urlopen_response(
            json.dumps({"tag_name": "v1.0.0"}).encode()
        )

        with patch("urllib.request.urlopen", return_value=mock_response):
            get_latest_version("owner/repo", verbose=True)

        captured = capsys.readouterr()
        assert "Checking for latest release" in captured.out
        assert "Latest release found: v1.0.0" in captured.out


class TestDownloadAndExtract:
    """Tests for download_and_extract function."""

    def test_download_main_branch_success(self, tmp_path):
        """Test downloading and extracting main branch."""
        dest_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Create a mock zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pyNegative-main/README.md", "# Test")
            zf.writestr("pyNegative-main/pyproject.toml", "[project]")
        zip_buffer.seek(0)

        mock_response = create_mock_urlopen_response(zip_buffer.read())

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                result = download_and_extract("main", dest_dir, repo, verbose=False)

        assert result is True
        assert os.path.exists(dest_dir)
        assert os.path.exists(os.path.join(dest_dir, "README.md"))

    def test_download_release_success(self, tmp_path):
        """Test downloading and extracting a release."""
        dest_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Create a mock zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pyNegative-v1.0.0/README.md", "# Test")
        zip_buffer.seek(0)

        mock_response = create_mock_urlopen_response(zip_buffer.read())

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                result = download_and_extract("v1.0.0", dest_dir, repo, verbose=False)

        assert result is True

    def test_download_network_error(self, tmp_path):
        """Test handling of network error during download."""
        dest_dir = str(tmp_path / "install")
        repo = "owner/repo"

        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = download_and_extract("v1.0.0", dest_dir, repo, verbose=False)

        assert result is False

    def test_removes_old_installation(self, tmp_path):
        """Test that old installation is removed before extracting new one."""
        dest_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Create existing installation
        os.makedirs(dest_dir)
        (tmp_path / "install" / "old_file.txt").write_text("old")

        # Create a mock zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pyNegative-v1.0.0/README.md", "# New")
        zip_buffer.seek(0)

        mock_response = create_mock_urlopen_response(zip_buffer.read())

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                result = download_and_extract("v1.0.0", dest_dir, repo, verbose=False)

        assert result is True
        assert not os.path.exists(os.path.join(dest_dir, "old_file.txt"))
        assert os.path.exists(os.path.join(dest_dir, "README.md"))

    def test_empty_zip_error(self, tmp_path):
        """Test handling of empty zip file."""
        dest_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Create an empty zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED):
            pass  # No files
        zip_buffer.seek(0)

        mock_response = create_mock_urlopen_response(zip_buffer.read())

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = download_and_extract("v1.0.0", dest_dir, repo, verbose=False)

        assert result is False

    def test_generates_icons(self, tmp_path):
        """Test that icon generation is triggered when script exists."""
        dest_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Create a mock zip file that includes the icon generation script
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pyNegative-v1.0.0/README.md", "# Test")
            zf.writestr("pyNegative-v1.0.0/scripts/generate_icons.py", "# Icon gen")
        zip_buffer.seek(0)

        mock_response = create_mock_urlopen_response(zip_buffer.read())

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                download_and_extract("v1.0.0", dest_dir, repo, verbose=False)

        # Verify subprocess was called for icon generation
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert any("generate_icons.py" in str(arg) for arg in call_args)

    def test_icon_generation_failure_warning(self, tmp_path, capsys):
        """Test that icon generation failure prints warning."""
        dest_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Create a mock zip file that includes the icon generation script
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pyNegative-v1.0.0/README.md", "# Test")
            zf.writestr("pyNegative-v1.0.0/scripts/generate_icons.py", "# Icon gen")
        zip_buffer.seek(0)

        mock_response = create_mock_urlopen_response(zip_buffer.read())

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(
                    returncode=1, stderr="Icon error"
                )
                download_and_extract("v1.0.0", dest_dir, repo, verbose=False)

        captured = capsys.readouterr()
        assert "Warning: Icon generation failed" in captured.out


class TestCheckUpdateAvailable:
    """Tests for check_update_available function."""

    def test_same_version_with_pyproject(self, tmp_path):
        """Test when current == latest and pyproject.toml exists."""
        install_dir = str(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[project]")

        result = check_update_available("v1.0.0", "v1.0.0", install_dir)
        assert result is False

    def test_different_versions(self, tmp_path):
        """Test when versions differ."""
        install_dir = str(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[project]")

        result = check_update_available("v1.0.0", "v1.1.0", install_dir)
        assert result is True

    def test_no_pyproject_file(self, tmp_path):
        """Test when pyproject.toml doesn't exist."""
        install_dir = str(tmp_path)

        result = check_update_available("v1.0.0", "v1.0.0", install_dir)
        assert result is True

    def test_no_current_version(self, tmp_path):
        """Test when no current version is set (fresh install)."""
        install_dir = str(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[project]")

        result = check_update_available(None, "v1.0.0", install_dir)
        assert result is True


class TestVersionFileManagement:
    """Tests for version file save/load functions."""

    def test_save_and_load_version(self, tmp_path):
        """Test saving and loading version roundtrip."""
        install_dir = str(tmp_path)
        version = "v1.2.3"

        save_version(version, install_dir)
        loaded = load_version(install_dir)

        assert loaded == version

    def test_save_creates_directory(self, tmp_path):
        """Test that save_version creates directory if it doesn't exist."""
        install_dir = str(tmp_path / "nested" / "dir")
        version = "v1.0.0"

        save_version(version, install_dir)
        loaded = load_version(install_dir)

        assert loaded == version
        assert os.path.exists(install_dir)

    def test_load_version_file_not_exists(self, tmp_path):
        """Test loading version when file doesn't exist."""
        install_dir = str(tmp_path)

        result = load_version(install_dir)
        assert result is None

    def test_load_version_file_read_error(self, tmp_path):
        """Test loading version when file read fails."""
        install_dir = str(tmp_path)
        version_file = tmp_path / ".version"
        version_file.write_text("v1.0.0")

        # Make file unreadable (on Unix systems)
        if os.name != "nt":
            os.chmod(version_file, 0o000)
            result = load_version(install_dir)
            os.chmod(version_file, 0o644)  # Restore permissions
            assert result is None


class TestMain:
    """Tests for main function and CLI."""

    def test_main_successful_install(self, tmp_path):
        """Test main function with successful installation."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        with patch(
            "sys.argv",
            ["download_release.py", "--repo", repo, "--install-dir", install_dir],
        ):
            with patch("download_release.get_latest_version", return_value="v1.0.0"):
                with patch("download_release.download_and_extract", return_value=True):
                    result = main()

        assert result == 0
        assert load_version(install_dir) == "v1.0.0"

    def test_main_already_latest(self, tmp_path):
        """Test main function when already on latest version."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Setup existing installation
        os.makedirs(install_dir)
        (tmp_path / "install" / "pyproject.toml").write_text("[project]")
        save_version("v1.0.0", install_dir)

        with patch(
            "sys.argv",
            ["download_release.py", "--repo", repo, "--install-dir", install_dir],
        ):
            with patch("download_release.get_latest_version", return_value="v1.0.0"):
                result = main()

        assert result == 2  # Exit code 2 = already on latest

    def test_main_download_failure(self, tmp_path):
        """Test main function when download fails."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        with patch(
            "sys.argv",
            ["download_release.py", "--repo", repo, "--install-dir", install_dir],
        ):
            with patch("download_release.get_latest_version", return_value="v1.0.0"):
                with patch("download_release.download_and_extract", return_value=False):
                    result = main()

        assert result == 1  # Exit code 1 = error

    def test_main_specific_version(self, tmp_path):
        """Test main function with specific version argument."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        with patch(
            "sys.argv",
            [
                "download_release.py",
                "--repo",
                repo,
                "--install-dir",
                install_dir,
                "--version",
                "v2.0.0",
            ],
        ):
            with patch("download_release.download_and_extract", return_value=True):
                result = main()

        assert result == 0
        assert load_version(install_dir) == "v2.0.0"

    def test_main_verbose_mode(self, tmp_path, capsys):
        """Test main function with verbose flag."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        with patch(
            "sys.argv",
            [
                "download_release.py",
                "--repo",
                repo,
                "--install-dir",
                install_dir,
                "--verbose",
            ],
        ):
            with patch("download_release.get_latest_version", return_value="v1.0.0"):
                with patch("download_release.download_and_extract", return_value=True):
                    main()

        captured = capsys.readouterr()
        assert (
            "Installing v1.0.0" in captured.out
            or "Successfully installed" in captured.out
        )

    def test_main_quiet_mode(self, tmp_path, capsys):
        """Test main function with quiet flag suppresses output."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        with patch(
            "sys.argv",
            [
                "download_release.py",
                "--repo",
                repo,
                "--install-dir",
                install_dir,
                "--quiet",
            ],
        ):
            with patch("download_release.get_latest_version", return_value="v1.0.0"):
                with patch("download_release.download_and_extract", return_value=True):
                    main()

        captured = capsys.readouterr()
        # Should not have installation message in quiet mode
        assert "Installing" not in captured.out
        assert "Successfully installed" not in captured.out

    def test_main_missing_required_args(self):
        """Test main function exits when required args are missing."""
        with patch("sys.argv", ["download_release.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_install_workflow(self, tmp_path):
        """Test complete installation workflow from check to download."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Mock GitHub API response
        mock_api_response = create_mock_urlopen_response(
            json.dumps({"tag_name": "v1.5.0"}).encode()
        )

        # Mock zip download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pyNegative-v1.5.0/README.md", "# pyNegative v1.5.0")
            zf.writestr("pyNegative-v1.5.0/pyproject.toml", "[project]")
        zip_buffer.seek(0)

        mock_download_response = create_mock_urlopen_response(zip_buffer.read())

        call_count = [0]

        def mock_urlopen(req, **kwargs):
            call_count[0] += 1
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "api.github.com" in url:
                return mock_api_response
            else:
                return mock_download_response

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)

                # First call - fresh install
                with patch(
                    "sys.argv",
                    [
                        "download_release.py",
                        "--repo",
                        repo,
                        "--install-dir",
                        install_dir,
                    ],
                ):
                    result = main()

                assert result == 0
                assert load_version(install_dir) == "v1.5.0"
                assert os.path.exists(os.path.join(install_dir, "README.md"))

                # Second call - should detect already on latest
                with patch(
                    "sys.argv",
                    [
                        "download_release.py",
                        "--repo",
                        repo,
                        "--install-dir",
                        install_dir,
                    ],
                ):
                    with patch(
                        "download_release.get_latest_version", return_value="v1.5.0"
                    ):
                        result = main()

                assert result == 2  # Already on latest

    def test_update_available_workflow(self, tmp_path):
        """Test workflow when update is available."""
        install_dir = str(tmp_path / "install")
        repo = "owner/repo"

        # Setup existing v1.0.0 installation
        os.makedirs(install_dir)
        (tmp_path / "install" / "pyproject.toml").write_text("[project]")
        save_version("v1.0.0", install_dir)

        # Mock update to v1.1.0
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pyNegative-v1.1.0/README.md", "# pyNegative v1.1.0")
        zip_buffer.seek(0)

        mock_response = create_mock_urlopen_response(zip_buffer.read())

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)

                with patch(
                    "sys.argv",
                    [
                        "download_release.py",
                        "--repo",
                        repo,
                        "--install-dir",
                        install_dir,
                        "--version",
                        "v1.1.0",
                    ],
                ):
                    result = main()

                assert result == 0
                assert load_version(install_dir) == "v1.1.0"
