# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- [x] **Add support for more image formats** (RAW: .arw, .nef, .raf, etc. | Standard: .jpg, .png, .webp, .tiff)
- [ ] **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- [ ] **Live Histogram**: Real-time luminance histogram display in the UI.
- [x] **remove use of os.path** move to favour pathlib
- [ ] **General code cleanup** analyse the codebase to look for redundant, duplicate or unused code. and clean it up
- [ ] **Performance optimisation**: Analyse the codebase for potential performance optimisations. looks for quick wins in favour of larger more specific overhauls

## Bugs

- [ ] **fix test warnings**: fix test warnings in all unit tests


## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1.  **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
2.  **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
3.  **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
