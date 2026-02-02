#!/usr/bin/env python3
"""Download and install pyNegative releases from GitHub.

This script handles downloading the latest release (or main branch) from GitHub,
extracting it to the specified directory, and managing version tracking.
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from typing import Optional


def get_latest_version(repo: str, verbose: bool = False) -> str:
    """Query GitHub API for the latest release tag.

    Args:
        repo: GitHub repository in format 'owner/repo'
        verbose: Whether to print progress messages

    Returns:
        The latest release tag name, or 'main' if no releases exist
    """
    if verbose:
        print("Checking for latest release...")

    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases/latest",
            headers={
                "User-Agent": "pyNegative-Installer",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            tag = data.get("tag_name", "main")
            if verbose:
                print(f"Latest release found: {tag}")
            return tag
    except Exception as e:
        if verbose:
            print(f"Could not check releases: {e}")
            print("Falling back to main branch")
        return "main"


def download_and_extract(
    version: str,
    dest_dir: str,
    repo: str,
    verbose: bool = False,
) -> bool:
    """Download and extract a release or branch from GitHub.

    Args:
        version: Version tag to download (e.g., 'v1.0.0') or 'main' for branch
        dest_dir: Destination directory for installation
        repo: GitHub repository in format 'owner/repo'
        verbose: Whether to print progress messages

    Returns:
        True if successful, False otherwise
    """
    if version == "main":
        url = f"https://github.com/{repo}/archive/refs/heads/main.zip"
        if verbose:
            print("Downloading main branch...")
    else:
        url = f"https://github.com/{repo}/archive/refs/tags/{version}.zip"
        if verbose:
            print(f"Downloading {version}...")

    try:
        # Download
        with urllib.request.urlopen(url, timeout=60) as response:
            zip_data = io.BytesIO(response.read())

        if verbose:
            print("Download complete, extracting...")

        # Extract to temp directory first
        temp_extract = dest_dir + ".extracting"
        if os.path.exists(temp_extract):
            shutil.rmtree(temp_extract)

        with zipfile.ZipFile(zip_data, "r") as zf:
            zf.extractall(temp_extract)

        # Find the extracted directory (usually like pyNegative-main/ or pyNegative-v1.0.0/)
        subdirs = [
            d
            for d in os.listdir(temp_extract)
            if os.path.isdir(os.path.join(temp_extract, d))
        ]
        if not subdirs:
            print("Error: No directory found in zip")
            return False

        source_dir = os.path.join(temp_extract, subdirs[0])

        # Remove old install if exists
        if os.path.exists(dest_dir):
            if verbose:
                print("Removing old installation...")
            shutil.rmtree(dest_dir)

        # Move to final location
        shutil.move(source_dir, dest_dir)
        shutil.rmtree(temp_extract)

        if verbose:
            print(f"Successfully extracted to {dest_dir}")

        # Generate icons if script exists
        icon_script = os.path.join(dest_dir, "scripts", "generate_icons.py")
        if os.path.exists(icon_script):
            if verbose:
                print("Generating icons...")
            try:
                result = subprocess.run(
                    ["uv", "run", "--python", "3", "python", icon_script],
                    capture_output=True,
                    text=True,
                    cwd=dest_dir,
                )
                if result.returncode == 0:
                    if verbose:
                        print("Icons generated successfully!")
                else:
                    print(f"Warning: Icon generation failed: {result.stderr}")
            except Exception as e:
                print(f"Warning: Could not generate icons: {e}")

        return True

    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def check_update_available(
    current: Optional[str],
    latest: str,
    install_dir: str,
) -> bool:
    """Check if an update is available.

    Args:
        current: Current installed version, or None if not installed
        latest: Latest available version from GitHub
        install_dir: Installation directory to check for pyproject.toml

    Returns:
        True if update is available, False if already on latest
    """
    if current == latest and os.path.exists(
        os.path.join(install_dir, "pyproject.toml")
    ):
        return False
    return True


def save_version(version: str, install_dir: str) -> None:
    """Save the installed version to a version file.

    Args:
        version: Version string to save
        install_dir: Installation directory
    """
    # Ensure install directory exists
    os.makedirs(install_dir, exist_ok=True)
    version_file = os.path.join(install_dir, ".version")
    with open(version_file, "w") as f:
        f.write(version)


def load_version(install_dir: str) -> Optional[str]:
    """Load the currently installed version from version file.

    Args:
        install_dir: Installation directory

    Returns:
        Version string, or None if not found
    """
    version_file = os.path.join(install_dir, ".version")
    if os.path.exists(version_file):
        try:
            with open(version_file, "r") as f:
                return f.read().strip()
        except Exception:
            pass
    return None


def main() -> int:
    """Main entry point for the download script.

    Returns:
        Exit code: 0=success, 1=error, 2=already on latest
    """
    parser = argparse.ArgumentParser(
        description="Download and install pyNegative releases from GitHub"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository in format 'owner/repo'",
    )
    parser.add_argument(
        "--install-dir",
        required=True,
        help="Directory to install pyNegative",
    )
    parser.add_argument(
        "--version",
        help="Specific version to install (default: latest release)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )

    args = parser.parse_args()

    # Determine output mode
    verbose = args.verbose and not args.quiet
    quiet = args.quiet

    # Get version to install
    if args.version:
        latest = args.version
        if verbose:
            print(f"Installing specified version: {latest}")
    else:
        latest = get_latest_version(args.repo, verbose=verbose)

    # Check current version
    current = load_version(args.install_dir)

    # Check if update is needed
    if not check_update_available(current, latest, args.install_dir):
        if not quiet:
            print(f"Already on latest version: {latest}")
        return 2  # Exit code 2 = already on latest

    if current and not quiet:
        print(f"Update available: {current} -> {latest}")
    elif not quiet:
        print(f"Installing {latest}...")

    # Download and extract
    if not download_and_extract(latest, args.install_dir, args.repo, verbose=verbose):
        return 1  # Exit code 1 = error

    # Save version
    save_version(latest, args.install_dir)

    if not quiet:
        print(f"Successfully installed {latest} to {args.install_dir}")

    return 0  # Exit code 0 = success


if __name__ == "__main__":
    sys.exit(main())
