# pyNegative

A modern, cross-platform desktop application for photographers, offering a fast and intuitive GUI to non-destructively edit, rate, and export RAW files.

## Installation

### Get the code
```bash
git clone https://github.com/lucashutch/pyNegative.git
cd pyNegative
```

### Using `uv` (Preffererd)
uv automatically creates a virtual environment and installs the dependencies.
```bash
# Install in editable mode
uv sync

# Run the UI
uv run pyneg-ui
```

### Using `pip` (Legacy)
This method is not recommended as it does not create a virtual environment. You can use the default python environment, but it is not recommended. especially on newer OSs.
```bash
# Install in editable mode
pip install -e .

# Run the UI
pyneg-ui
```

## Features

### Core Functionality
-   **Non-destructive Editing**: Edits are stored in JSON sidecar files (`.xmp`) alongside the original RAW files.
-   **RAW Image Processing**: Adjustments for exposure, contrast, blacks, whites, shadows, highlights, sharpening, and de-noise.

### UI Editor (`pyneg-ui`)
A graphical user interface (GUI) for interactively editing RAW files.
-   **Interactive Editing**: Live preview of changes as adjustments are made.
-   **Comprehensive Adjustments**: Sliders for fine-tuning exposure, contrast, blacks, whites, shadows, highlights, sharpening, and de-noise.
-   **Advanced Processing**:
    -   **Sharpening**: High-quality, edge-aware sharpening algorithms (Advanced/Standard).
    -   **Noise Reduction**: Multiple denoising methods including Non-Local Means (NLM), Bilateral, and Total Variation.
    -   **Processing Presets**: Quick "Subtle", "Medium", and "Aggressive" presets for detail enhancement.
-   **Zoom Support**: Zoom into photos for precise control during editing.
-   **Responsive Sliders**: Optimized performance for smooth slider interactions.
-   **Quick Reset**: Double-click sliders to reset them to their default values.
-   **Image Carousel**: Horizontal carousel for easy navigation through images in a folder.
-   **Integrated Rating System**: Assign star ratings (1-5) to photos directly within the editor.
-   **Dynamic Carousel Sync**: Editor's image carousel automatically updates and synchronizes with gallery filters.

### Gallery View
-   **Grid Browsing**: Browse images in a folder in a responsive grid layout.
-   **Full-size Preview Mode**: Toggle between grid layout and a high-quality, zoomable preview without leaving the gallery.
-   **Interactive Navigation**: Double-click any photo to jump into preview mode, and double-click the preview to return to the grid.
-   **Star Rating Display**: View assigned star ratings for each image thumbnail.
-   **Rating-based Filtering**: Filter displayed images by star rating (match, greater than, or less than a specified rating).

### Export Management
-   **Dedicated Export Tab**: A specialized interface for managing the export pipeline.
-   **Intelligent Destinations**: Automatically suggests export paths based on the current folder structure.
-   **Batch Exporting**: Process all currently filtered images at once with consistent settings.
-   **Flexible Formats**: Support for exporting to JPEG and high-efficiency formats.

## Development

Install Dev dependancies:
```bash
uv sync --all-groups
```

Run tests with:
```bash
uv run pytest
```

## Roadmap / TODO

Track planned features and project goals.

-   [ ] **Enable multi-threaded**: Enable multi-threaded processing of the images in the edit view and gallery view.
-   [ ] **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
-   [ ] **Live Histogram**: Real-time luminance histogram display in the UI.
-   [ ] **Batch Renaming**: Sequence-based renaming for folder exports.
-   [ ] **Readme refactor** update readme, and create a readme structure that is easy to navigate. and covers larger sections of the codebase in more details. eg tests.

## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1.  **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
2.  **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
3.  **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
