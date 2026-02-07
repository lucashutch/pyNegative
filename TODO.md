# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- [ ] **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- [ ] **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- [ ] **User Presets**: Allow saving and applying named adjustment presets.
- [ ] **Metadata Panel**: Display EXIF data (ISO, Shutter, Aperture) in the editor.
- [ ] **Gallery sorting** - currnetly the gallery is unsorted. users should be able to sort by filename, date taken, rating, and last edited date. users should be able to sort in ascending and descending order.

## Performance Optimisation

- [ ] **Persistent Thumbnail Cache**: Store thumbnails on disk to speed up gallery loading.
- [ ] **GPU Acceleration**: Explore OpenCL/CuPy for faster tone-mapping calculations.
- [ ] **improve gpu acceleration startup time** - it takes a few seconds for the gpu acceleration to kick in. investigate if this can be improved.
- [ ] **General Code Cleanup**: Analyse the codebase for redundant, duplicate or unused code.

## Bugs

- [ ] **sharpening defaults to a value of 20** - the sharpening slider defaults to a value of 20. this is not correct. it should default to 0.
- [ ] **preview double clikck** - in gallery preview double clicking the photo makes it larger, but sometimes it doesnt work. in particular if i had it enlarged, then changed the gallery filter to something taht doesnt include the photo. It will revert tot he gallery view, but when i try and double clikc another image to enlarge it, nothing happens.

## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1. **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
2. **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
3. **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
