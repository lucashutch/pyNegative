#!/usr/bin/env python3
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LENSFUN_DB_URL = "https://api.github.com/repos/lensfun/lensfun/contents/data/db"
RAW_URL_BASE = "https://raw.githubusercontent.com/lensfun/lensfun/master/data/db/"


def download_file(file_info, output_dir):
    name = file_info["name"]
    if not name.endswith(".xml"):
        return

    url = file_info["download_url"]
    dest = output_dir / name

    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return False


def run_download(output_dir: str | Path, quiet: bool = False):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"Downloading lensfun database to {output_dir}...")

    try:
        req = urllib.request.Request(
            LENSFUN_DB_URL,
            headers={
                "User-Agent": "pyNegative-Downloader",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        with urllib.request.urlopen(req) as response:
            files = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching file list: {e}")
        # Fallback to a core list if API fails
        core_files = [
            "slr-canon.xml",
            "slr-nikon.xml",
            "slr-sony.xml",
            "mil-sony.xml",
            "mil-olympus.xml",
            "mil-panasonic.xml",
        ]
        files = [{"name": f, "download_url": f"{RAW_URL_BASE}{f}"} for f in core_files]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda f: download_file(f, output_dir), files))

    success_count = sum(1 for r in results if r)

    if not quiet:
        print(f"Successfully downloaded {success_count} database files.")

    # Save a version file
    (output_dir / ".lensdb_version").write_text("latest")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Download lensfun database XML files")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save XML files"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    run_download(args.output_dir, args.quiet)


if __name__ == "__main__":
    main()
