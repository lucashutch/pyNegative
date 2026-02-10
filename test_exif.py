#!/usr/bin/env python3
"""Quick test script to see what EXIF tags are available in a RAW file."""

import sys
import exifread
import rawpy
from pathlib import Path
from io import BytesIO

if len(sys.argv) < 2:
    print("Usage: python test_exif.py <path_to_raw_file>")
    sys.exit(1)

raw_path = Path(sys.argv[1])

if not raw_path.exists():
    print(f"File not found: {raw_path}")
    sys.exit(1)

print(f"Reading EXIF from: {raw_path}\n")

try:
    with rawpy.imread(str(raw_path)) as raw:
        print("RAW file opened successfully with rawpy")
        print(f"Camera: {raw.color_desc if hasattr(raw, 'color_desc') else 'Unknown'}")

        # Try to extract embedded thumbnail
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                print(f"Extracted JPEG thumbnail ({len(thumb.data)} bytes)\n")

                # Parse EXIF from thumbnail
                thumb_io = BytesIO(thumb.data)
                tags = exifread.process_file(thumb_io, details=False)

                print(f"Total tags found in thumbnail: {len(tags)}\n")

                # Print all tags
                for tag, value in sorted(tags.items()):
                    print(f"{tag}: {value}")

                print("\n" + "=" * 80)
                print("Checking specific tags we're looking for:")
                print("=" * 80)

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
            else:
                print(f"Thumbnail format is not JPEG: {thumb.format}")
        except Exception as thumb_error:
            print(f"Could not extract thumbnail: {thumb_error}")

except Exception as e:
    print(f"Error opening RAW file: {e}")
    import traceback

    traceback.print_exc()
