#!/usr/bin/env python3
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pynegative
import numpy as np
from PIL import Image

def process_file_cli(input_path, output_dir, fmt, quality):
    """
    Worker function for CLI batch processing.
    """
    try:
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        # 1. Load RAW
        img = pynegative.open_raw(input_path)

        # 2. Check for sidecar or calculate auto-exposure
        settings = pynegative.load_sidecar(input_path)
        if not settings:
            settings = pynegative.calculate_auto_exposure(img)

        # 3. Apply settings
        processed, _ = pynegative.apply_tone_map(
            img,
            exposure=settings.get("exposure", 0.0),
            blacks=settings.get("blacks", 0.0),
            whites=settings.get("whites", 1.0),
            shadows=settings.get("shadows", 0.0),
            highlights=settings.get("highlights", 0.0),
            saturation=settings.get("saturation", 1.0)
        )

        # 4. Save
        pil_img = Image.fromarray((processed * 255).astype("uint8"))

        out_filename = input_path.with_suffix(f".{fmt.lower()}").name
        out_path = output_dir / out_filename

        pynegative.save_image(pil_img, out_path, quality=quality)
        return True, None
    except Exception as e:
        return False, str(e)


def run_batch(input_dir, output_dir=None, fmt="JPG", quality=90, workers=4):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory.")
        return

    # Find RAW files
    raw_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in pynegative.SUPPORTED_EXTS])

    if not raw_files:
        print(f"No RAW files found in {input_dir}")
        return

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = input_dir / "converted"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(raw_files)} files to {fmt}...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for file_path in raw_files:
            futures.append(executor.submit(process_file_cli, file_path, output_dir, fmt, quality))

        for i, future in enumerate(futures):
            success, error = future.result()
            base = raw_files[i].name
            if success:
                print(f"[{i+1}/{len(raw_files)}] {base} -> Done")
            else:
                print(f"[{i+1}/{len(raw_files)}] {base} -> Error: {error}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="pyNegative Batch Converter")
    parser.add_argument("input", help="Input directory")
    parser.add_argument("-o", "--output", help="Output directory (default: input/converted)")
    parser.add_argument("-f", "--format", default="JPG", help="Output format (JPG or HEIF)")
    parser.add_argument("-q", "--quality", type=int, default=90, help="JPEG/HEIF quality (1-100)")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Number of parallel jobs")

    args = parser.parse_args()
    run_batch(args.input, args.output, args.format, args.quality, args.jobs)

if __name__ == "__main__":
    main()
