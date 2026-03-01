<div align="center">
  <img src="pynegative_icon.png" alt="pyNegative Logo" width="128" height="128">
  <h1>pyNegative</h1>
  <p>A modern, fast, and intuitive RAW photo editor for photographers who value speed and simplicity.</p>

  ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
  ![License](https://img.shields.io/github/license/lucashutch/pyNegative)
  ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
</div>

---

## Features

### ✨ Develop Your RAWs
Get the most out of your photos with a powerful, non-destructive editing suite.
- **Instant Previews** — See your changes in real-time as you adjust sliders.
- **Live Histogram** — High-precision histogram with Luminance, RGB, and YUV modes.
- **Fine-Tuned Control** — Precision adjustments for exposure, contrast, shadows, highlights, white balance, and more.
- **Smart Dehaze, Sharpening & Denoise** — Professional-grade Numba-accelerated algorithms to bring out detail and clarity without the noise.
- **Geometry Tools** — Crop, rotate, and flip your images with constrained aspect ratios.
- **Lens Correction** — Auto-correct distortion, vignetting, and chromatic aberration using a built-in lens database.
- **Manual Defringe** — Targeted desaturation of purple and green fringes on high-contrast edges.
- **Before/After Comparison** — Drag the split-view handle to compare your edits with the original.
- **EXIF Metadata** — View camera settings (ISO, shutter speed, aperture, lens, etc.) in a dedicated panel.
- **Undo/Redo** — Full undo/redo history for all your adjustments.
- **Safe Editing** — Your original files are never touched. Edits are saved in tiny JSON sidecar files alongside your images.

### 📁 Organize Your Shoots
Cull and rate your photos faster than ever.
- **Fluid Gallery** — Browse hundreds of photos in a responsive, resizable grid.
- **Large Preview Mode** — Toggle between grid and a full-size, zoomable preview with a double-click.
- **Quick Rating** — Rate your best shots from 1–5 stars with a click or keyboard shortcut.
- **Gallery Sorting** — Sort by filename, date taken, rating, or last edited, in ascending or descending order.
- **Smart Filtering** — Filter your gallery by star rating with match, less-than, or greater-than modes.
- **Fast Loading** — Thumbnails are cached to disk so returning to a folder is nearly instant.

### 🚀 Export with Confidence
Get your photos ready for the world with a streamlined export pipeline.
- **Batch Processing** — Export your entire filtered selection in one go.
- **Smart Destinations** — Suggested save locations based on your folder structure.
- **Flexible Formats** — High-quality JPEG, HEIF/HEIC, and more.
- **Custom Renaming** — Automatic renaming using EXIF dates and sequence numbers, with a live rename preview.
- **Export Presets** — Save and recall your favorite export configurations.

### 🖼️ Broad Format Support
- **RAW Formats**: CR2, CR3, DNG, ARW, NEF, NRW, RAF, ORF, RW2, PEF
- **Standard Formats**: JPEG, PNG, WebP, TIFF, HEIC/HEIF

---

## Installation

### Quick Install

**Windows:** Download [`install-pynegative.bat`](scripts/install-pynegative.bat) and double-click it.

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/lucashutch/pyNegative/main/scripts/install-pynegative.sh | bash
```

Note: When run non-interactively (for example via `curl | bash`), the installer will automatically update an existing pyNegative installation and prints concise status or failure messages to help diagnose issues when running in silent mode.

The installer handles all dependencies, generates icons, and creates shortcuts for you automatically. No technical knowledge required! See the [installer README](scripts/README.md) for silent mode, troubleshooting, and more.

---

## System Requirements

| | Minimum |
|---|---|
| **OS** | Windows 10+, macOS 10.12+, or a modern Linux distro |
| **Python** | 3.10+ (installed automatically by the installer) |
| **RAM** | 4 GB (8 GB recommended for large RAW files) |

---

## Keyboard Shortcuts

| Action | Shortcut |
|---|---|
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Shift+Z` |
| Copy Settings | `Ctrl+C` |
| Paste Settings | `Ctrl+V` |
| Zoom In | `Ctrl++` / `Ctrl+=` |
| Zoom Out | `Ctrl+-` |
| Fit to Screen | `Ctrl+0` |
| Toggle Comparison | `C` |
| Rate 1–5 Stars | `1`–`5` |
| Remove Rating | `0` |
| Next / Previous Image | `→` / `←` |

---

## How Edits Are Stored

pyNegative is fully non-destructive. Your original images are never modified. All adjustments are stored in lightweight JSON sidecar files inside a hidden `.pyNegative` directory next to your images:

```
/Photos/
├── IMG_0001.CR3
├── IMG_0002.CR3
└── .pyNegative/
    ├── IMG_0001.CR3.json      # Edit settings
    ├── IMG_0002.CR3.json
    └── thumbnails/            # Cached thumbnails for fast loading
        ├── IMG_0001.CR3.1738000000.400.webp
        └── IMG_0001.CR3.1738000000.400.json
```

---

## Want to Contribute?

We welcome contributions! Check out the [Contributing Guide](CONTRIBUTING.md) to get started with the source code, development setup, and code standards.

## Project Roadmap

See [TODO.md](TODO.md) for planned features and areas of improvement.

## License

This project is open source under the MIT License. See the [LICENSE](LICENSE) file for details.

### Data Licenses
The lens database used by pyNegative is sourced from the [Lensfun Project](https://lensfun.github.io/) and is licensed under **CC-BY-SA 3.0**. Attribution for individual lens profiles remains with their respective contributors in the database XML files.

### Icon Attribution
UI icons are from [Heroicons](https://heroicons.com/) by Tailwind Labs, licensed under the **MIT License**. Copyright (c) 2020 Refactoring UI Inc.
