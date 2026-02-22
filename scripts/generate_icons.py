#!/usr/bin/env python3
"""
Icon generation script for pyNegative installer.
Resizes the main icon to various sizes needed for different platforms.
"""

import sys
from pathlib import Path

from PIL import Image


def generate_icons():
    """Generate all required icon formats from the main icon."""

    # Paths
    base_dir = Path(__file__).resolve().parent.parent
    source_icon = base_dir / "pynegative_icon.png"
    icons_dir = base_dir / "scripts" / "icons"

    # Ensure icons directory exists
    icons_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source icon...")
    try:
        img = Image.open(source_icon)
    except FileNotFoundError:
        print(f"Error: Could not find source icon at {source_icon}")
        sys.exit(1)

    # Ensure image is RGBA for transparency support
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Generate 256x256 PNG (used by Linux and general purpose)
    print("Generating 256x256 PNG...")
    img_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
    png_path = icons_dir / "pynegative_256.png"
    img_256.save(str(png_path), "PNG")
    print(f"  Saved: {png_path}")

    # Generate Windows ICO file (multi-resolution)
    print("Generating Windows ICO file...")
    sizes = [16, 32, 48, 256]
    ico_images = []

    for size in sizes:
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        ico_images.append(resized)

    ico_path = icons_dir / "pynegative.ico"
    # Save with the largest image first, then the rest
    ico_images[-1].save(
        str(ico_path),
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=ico_images[:-1],
    )
    print(f"  Saved: {ico_path} (sizes: {sizes})")

    # Generate macOS ICNS file (multi-resolution)
    print("Generating macOS ICNS file...")
    mac_sizes = [16, 32, 128, 256, 512, 1024]

    # For ICNS, we need to create a set of PNGs and then combine them
    # PIL doesn't directly support ICNS, so we'll use iconutil on macOS
    # or create a directory structure that can be converted

    icns_dir = icons_dir / "pynegative.iconset"
    icns_dir.mkdir(parents=True, exist_ok=True)

    for size in mac_sizes:
        # Normal resolution
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        icon_name = f"icon_{size}x{size}.png"
        icon_path = icns_dir / icon_name
        resized.save(str(icon_path), "PNG")

        # High resolution (@2x) for sizes <= 512
        if size <= 512:
            resized_2x = img.resize((size * 2, size * 2), Image.Resampling.LANCZOS)
            icon_name_2x = f"icon_{size}x{size}@2x.png"
            icon_path_2x = icns_dir / icon_name_2x
            resized_2x.save(str(icon_path_2x), "PNG")

    print(f"  Created iconset directory: {icns_dir}")
    print("  To create .icns file on macOS, run:")
    print(f"    iconutil -c icns {icns_dir}")

    # Try to create ICNS if on macOS
    if sys.platform == "darwin":
        print("  Attempting to create ICNS file automatically...")
        try:
            import subprocess

            icns_path = icons_dir / "pynegative.icns"
            result = subprocess.run(
                ["iconutil", "-c", "icns", str(icns_dir), "-o", str(icns_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  Saved: {icns_path}")
            else:
                print(
                    "  Note: iconutil failed, iconset directory is ready for manual conversion"
                )
        except FileNotFoundError:
            print(
                "  Note: iconutil not found, iconset directory is ready for manual conversion"
            )
    else:
        print("  Note: ICNS conversion requires macOS with iconutil")

    print("\nIcon generation complete!")
    print(f"Icons saved to: {icons_dir}")


if __name__ == "__main__":
    generate_icons()
