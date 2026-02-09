# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- **User Presets**: Allow saving and applying named adjustment presets.
- **Metadata Panel**: Display EXIF data (ISO, Shutter, Aperture) in the editor. there should be a button to toggle the panel on and off. it should be located in the top bar. when the panel is closed, the image should take up the full width of the editor.
- **update documentation** - the documentation is out of date with the current state of the project.

## Performance Optimisation

- **Persistent Thumbnail Cache**: Store thumbnails on disk to speed up gallery loading.
- **General Code Cleanup**: Analyse the codebase for redundant, duplicate or unused code.
- **compile numba kernels on startup** - this should speed up the initialisation of the application. the kernels are cached but only compile jit whent hey are needed. should recompute them on startup if they are needed.
- **Refine NLM denoising** - separate the luma and chroma denoising into two separate functions. 

## Bugs

- **ad maximum CI test time of 2 mins** - the CI tests are occasionanly hanging. add a timeout, but also investigate why occasioanlly the CI (especailly for py3.14 hangs or takes a super long time at the installing dependancies uv sync step
-
## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1. **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
2. **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
3. **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
