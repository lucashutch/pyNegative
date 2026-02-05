# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- [ ] **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- [ ] **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- [ ] **User Presets**: Allow saving and applying named adjustment presets.
- [ ] **Metadata Panel**: Display EXIF data (ISO, Shutter, Aperture) in the editor.
- [ ] **General Code Cleanup**: Analyse the codebase for redundant, duplicate or unused code.
- [ ] **image dehaze**: implement image dehazing with a slider like denoise.

## Performance Optimisation

- [ ] **Persistent Thumbnail Cache**: Store thumbnails on disk to speed up gallery loading.
- [ ] **GPU Acceleration**: Explore OpenCL/CuPy for faster tone-mapping calculations.

## Bugs

## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1. **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
2. **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
3. **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
