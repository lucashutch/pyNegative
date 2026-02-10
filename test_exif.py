#!/usr/bin/env python3
"""Quick test script to see what EXIF tags are available in a RAW file."""

import sys
import exifread
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python test_exif.py <path_to_raw_file>")
    sys.exit(1)

raw_path = Path(sys.argv[1])

if not raw_path.exists():
    print(f"File not found: {raw_path}")
    sys.exit(1)

print(f"Reading EXIF from: {raw_path}\n")

with open(raw_path, 'rb') as f:
    tags = exifread.process_file(f, details=False)

    print(f"Total tags found: {len(tags)}\n")

    # Print all tags
    for tag, value in sorted(tags.items()):
        print(f"{tag}: {value}")

    print("\n" + "="*80)
    print("Checking specific tags we're looking for:")
    print("="*80)

    fields = {
        "ISO": "EXIF ISOSpeedRatings",
        "Shutter Speed": "EXIF ExposureTime",
        "Aperture": "EXIF FNumber",
        "Focal Length": "EXIF FocalLength",
        "Camera Make": "Image Make",
        "Camera Model": "Image Model",
        "Lens Model": "EXIF LensModel",
        "Date Taken": "EXIF DateTimeOriginal",
        "Exposure Comp": "EXIF ExposureBiasValue",
        "White Balance": "EXIF WhiteBalance",
        "Flash": "EXIF Flash",
        "Width": "EXIF ExifImageWidth",
        "Height": "EXIF ExifImageLength",
    }

    for name, tag_key in fields.items():
        value = tags.get(tag_key, "NOT FOUND")
        print(f"{name:20} ({tag_key:30}): {value}")
