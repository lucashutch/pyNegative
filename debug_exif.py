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

        # Get dimensions from rawpy
        if hasattr(raw, "sizes"):
            sizes = raw.sizes
            if hasattr(sizes, "raw_width") and hasattr(sizes, "raw_height"):
                print(f"RAW Dimensions: {sizes.raw_width} x {sizes.raw_height}\n")

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
                    "Date Taken (Original)": "EXIF DateTimeOriginal",
                    "Date Taken (DateTime)": "Image DateTime",
                    "Date Taken (Digitized)": "EXIF DateTimeDigitized",
                    "Exposure Comp": "EXIF ExposureBiasValue",
                    "White Balance": "EXIF WhiteBalance",
                    "Flash": "EXIF Flash",
                    "Width (Thumbnail)": "EXIF ExifImageWidth",
                    "Height (Thumbnail)": "EXIF ExifImageLength",
                }

                for name, tag_key in fields.items():
                    value = tags.get(tag_key, "NOT FOUND")
                    print(f"{name:30} ({tag_key:30}): {value}")

                # Show final date we'd use
                print("\n" + "=" * 80)
                date_taken = (
                    tags.get("EXIF DateTimeOriginal", None)
                    or tags.get("Image DateTime", None)
                    or tags.get("EXIF DateTimeDigitized", None)
                )
                print(f"Final Date Taken (with fallbacks): {date_taken}")

            else:
                print(f"Thumbnail format is not JPEG: {thumb.format}")
        except Exception as thumb_error:
            print(f"Could not extract thumbnail: {thumb_error}")

except Exception as e:
    print(f"Error opening RAW file: {e}")
    import traceback

    traceback.print_exc()
