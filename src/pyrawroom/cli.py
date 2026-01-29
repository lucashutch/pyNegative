#!/usr/bin/env python3
import rawpy
import numpy as np
import os
import shutil
import argparse
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from . import core as pyrawroom

# ---------------- Conversion ----------------
def convert_to_image(
    input_path,
    output_path,
    fmt="jpeg",
    quality=95,
    exposure=0.0,
    blacks=0.0,
    whites=1.0,
    shadows=0.0,
    highlights=0.0,
    sharpen=False,
    sharpen_radius=2,
    sharpen_percent=150,
    print_info=True,
):
    img = pyrawroom.open_raw(input_path)

    img_transformed, stats = pyrawroom.apply_tone_map(
        img,
        exposure=exposure,
        blacks=blacks,
        whites=whites,
        shadows=shadows,
        highlights=highlights
    )

    rgb_out = (img_transformed * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb_out)

    if sharpen:
        pil_img = pyrawroom.sharpen_image(pil_img, sharpen_radius, sharpen_percent)

    pyrawroom.save_image(pil_img, output_path, quality)

    if print_info:
        print(f"{os.path.basename(input_path)}:")
        print(f"  Exp: {exposure}, Blk: {blacks}, Wht: {whites}, Shd: {shadows}, Hgh: {highlights}")
        print(f"  Clipped: Shadows {stats['pct_shadows_clipped']:.2f}%, Highlights {stats['pct_highlights_clipped']:.2f}%")
        print(f"Saved: {output_path}\n")

    return stats


# ---------------- Move RAW ----------------
def move_raw_file(input_file, move_dir):
    os.makedirs(move_dir, exist_ok=True)
    dest_path = os.path.join(move_dir, os.path.basename(input_file))
    shutil.move(input_file, dest_path)
    print(f"Moved RAW to: {dest_path}")


# ---------------- Process a single file (wrapper) ----------------
def process_file_wrapper(args):
    # unpack args
    (
        input_file, out_path, fmt, quality,
        exposure, blacks, whites, shadows, highlights,
        sharpen, s_rad, s_per,
        move_raw, move_dir
    ) = args

    result = convert_to_image(
        input_file, out_path, fmt, quality,
        exposure, blacks, whites, shadows, highlights,
        sharpen, s_rad, s_per,
        print_info=False
    )

    if move_raw:
        move_raw_file(input_file, move_dir)
    return os.path.basename(input_file), result


# ---------------- Process Directory ----------------
def process_directory(
    input_dir,
    output_dir=None,
    fmt="jpeg",
    quality=95,
    move_raw=False,
    move_dir=None,
    exposure=0.0,
    blacks=0.0,
    whites=1.0,
    shadows=0.0,
    highlights=0.0,
    sharpen=False,
    sharpen_radius=2,
    sharpen_percent=150,
    max_workers=None,
):
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    raw_files = [f for f in os.listdir(input_dir) if f.lower().endswith(pyrawroom.SUPPORTED_EXTS)]
    print(f"Found {len(raw_files)} RAW file(s) in {input_dir}.")

    if not raw_files: return

    if not output_dir: output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    if move_raw and not move_dir:
        move_dir = os.path.join(input_dir, "converted")

    tasks = []
    for f in raw_files:
        input_file = os.path.join(input_dir, f)
        base_name = os.path.splitext(f)[0] + "." + fmt.lower()
        out_path = os.path.join(output_dir, base_name)

        tasks.append((
            input_file, out_path, fmt, quality,
            exposure, blacks, whites, shadows, highlights,
            sharpen, sharpen_radius, sharpen_percent,
            move_raw, move_dir
        ))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_wrapper, t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            fname, _ = future.result()
            print(f"Completed {i}/{len(raw_files)}: {fname}")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="RAW to JPEG/HEIF with Tone Mapping.")
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-f", "--format", choices=["jpeg", "heif"], default="jpeg", help="Output format")
    parser.add_argument("-q", "--quality", type=int, default=95, help="Quality (0-100)")

    # Tone Mapping
    parser.add_argument("--exposure", type=float, default=0.0, help="Exposure bias in Stops (e.g. 1.0, -2.0)")
    parser.add_argument("--whites", type=float, default=1.0, help="White point scale (Contrast). Default 1.0")
    parser.add_argument("--blacks", type=float, default=0.0, help="Black point offset. Default 0.0")
    parser.add_argument("--shadows", type=float, default=0.0, help="Shadow lift (-1.0 to 1.0)")
    parser.add_argument("--highlights", type=float, default=0.0, help="Highlight recovery (-1.0 to 1.0). Negative recovers.")

    # Sharpening
    parser.add_argument("--sharpen", action="store_true", help="Apply sharpening")
    parser.add_argument("--sharpen-radius", type=float, default=2, help="Sharpen radius")
    parser.add_argument("--sharpen-percent", type=int, default=150, help="Sharpen strength")

    # Files
    parser.add_argument("--move-raw", action="store_true", help="Move original RAWs")
    parser.add_argument("--move-dir", help="Move destination")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output,
        args.format,
        args.quality,
        move_raw=args.move_raw,
        move_dir=args.move_dir,
        exposure=args.exposure,
        blacks=args.blacks,
        whites=args.whites,
        shadows=args.shadows,
        highlights=args.highlights,
        sharpen=args.sharpen,
        sharpen_radius=args.sharpen_radius,
        sharpen_percent=args.sharpen_percent,
        max_workers=args.workers,
    )

if __name__ == "__main__":
    main()
