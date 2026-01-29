# pyNegative

A modern, cross-platform desktop application for photographers, offering a fast and intuitive GUI to non-destructively edit, rate, and export RAW files.

## Installation

```bash
# Clone the repository
git clone https://github.com/lucashutch/pyNegative.git
cd pyNegative

# Install in editable mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

## Features

### Core Functionality
-   **Non-destructive Editing**: Edits are stored in JSON sidecar files (`.xmp`) alongside the original RAW files.
-   **RAW Image Processing**: Adjustments for exposure, contrast, blacks, whites, shadows, highlights, sharpening, and de-noise.

### UI Editor (`pyneg-ui`)
A graphical user interface (GUI) for interactively editing RAW files.
-   **Interactive Editing**: Live preview of changes as adjustments are made.
-   **Comprehensive Adjustments**: Sliders for fine-tuning exposure, contrast, blacks, whites, shadows, highlights, sharpening, and de-noise.
-   **Zoom Support**: Zoom into photos for precise control during editing.
-   **Responsive Sliders**: Optimized performance for smooth slider interactions.
-   **Quick Reset**: Double-click sliders to reset them to their default values.
-   **Image Carousel**: Horizontal carousel for easy navigation through images in a folder.
-   **Integrated Rating System**: Assign star ratings (1-5) to photos directly within the editor.
-   **Dynamic Carousel Sync**: Editor's image carousel automatically updates and synchronizes with gallery filters.

### Gallery View
-   **Grid Browsing**: Browse images in a folder in a responsive grid layout.
-   **Star Rating Display**: View assigned star ratings for each image thumbnail.
-   **Rating-based Filtering**: Filter displayed images by star rating (match, greater than, or less than a specified rating).

## Development

Run tests with:
```bash
pytest
```

## Roadmap / TODO

Track planned features and project goals.

-   [ ] **Full size preview in gallery**: Allow the user to get a full size preview of the image in the gallery view.
-   [ ] **Enable multi-threaded**: Enable multi-threaded processing of the images in the edit view and gallery view.
-   [ ] **Rework sharpening and denoise interface and performance**: Improve the UI and performance of sharpening and de-noise functionalities.
-   [ ] **Batch Export**: Select and process multiple files at once.
-   [ ] **Sync Settings**: Copy/paste edit changes from one photo to others.
-   [ ] **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
-   [ ] **Export Tab Feature**: Add a dedicated export tab to control what and how images are exported (e.g., format, quality, resizing, watermarks).
-   [ ] **Live Histogram**: Real-time luminance histogram display in the UI.
-   [ ] **Archival Export Preset**: Reduce storage footprint (e.g., 16-bit DNG or high-quality HEIF) while retaining latitude.
-   [ ] **Batch Renaming**: Sequence-based renaming for folder exports.

## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1.  **Sidecar File Management (`src/pynegative/core.py`)**: Ensure settings (rating, tone, etc.) are correctly saved to and loaded from JSON sidecar files, including handling of defaults and corrupted files. (Ease: Easiest)
2.  **Star Rating Widget (`src/pynegative/ui/widgets/starrating.py`)**: Test the interactive UI component to verify `set_rating`, mouse events, and `ratingChanged` signal emissions. (Ease: Easy)
3.  **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
4.  **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
5.  **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
