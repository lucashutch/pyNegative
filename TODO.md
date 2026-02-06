# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Completed Features

- [x] **Unedited Photo Comparison**: Added a split-view interface (Shortcut: U) with a draggable handle to compare the original RAW image with current edits, featuring pixel-perfect alignment and zoom/pan support.

## Roadmap / TODO

- [ ] **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- [ ] **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- [ ] **User Presets**: Allow saving and applying named adjustment presets.
- [ ] **Metadata Panel**: Display EXIF data (ISO, Shutter, Aperture) in the editor.
- [ ] **General Code Cleanup**: Analyse the codebase for redundant, duplicate or unused code.

## Performance Optimisation

- [ ] **Persistent Thumbnail Cache**: Store thumbnails on disk to speed up gallery loading.
- [ ] **GPU Acceleration**: Explore OpenCL/CuPy for faster tone-mapping calculations.
- [ ] **improve gpu acceleration startup time** - it takes a few seconds for the gpu acceleration to kick in. investigate if this can be improved.

## Bugs

-- [ ] **preview double clikck** - in gallery preview double clicking the photo makes it larger, but sometimes it doesnt work. in particular if i had it enlarged, then changed the gallery filter to something taht doesnt include the photo. It will revert tot he gallery view, but when i try and double clikc another image to enlarge it, nothing happens.

## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1. **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
2. **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
3. **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
