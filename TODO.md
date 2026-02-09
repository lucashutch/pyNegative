# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- **User Presets**: Allow saving and applying named adjustment presets.
- **Metadata Panel**: Display EXIF data (ISO, Shutter, Aperture) in the editor. there should be a button to toggle the panel on and off. it should be located in the top bar. when the panel is closed, the image should take up the full width of the editor.

## Performance Optimisation

- **Persistent Thumbnail Cache**: Store thumbnails on disk to speed up gallery loading.
- **GPU Acceleration**: Explore OpenCL/CuPy for faster tone-mapping calculations.
- **GPU acceleration**: Investigate writing a custom gpu acceleration for opencv/opencl that perforam all of the requrired translations ins a single step to reduce time spent transferring photo data between cpu and gpu.
- **improve gpu acceleration startup time** - it takes a few seconds for the gpu acceleration to kick in. investigate if this can be improved.
- **General Code Cleanup**: Analyse the codebase for redundant, duplicate or unused code.
- **investigate vulkan acceleration** - it may be possible to use vulkan for faster cross platform image processing.
- **deep dive on program startup time**: Perform a deep dive on the time different parts of this program take to startup and contribute to the overall system start up time.

## Bugs

## Testing Improvement Areas

Based on recent project growth, the following areas would benefit from expanded unit testing:

1. **Image Adjustment Logic (`src/pynegative/core.py`)**: Test core functions like `apply_tone_map` and `sharpen_image` with known inputs to assert that the output image data is mathematically correct. (Ease: Medium)
2. **Gallery Filtering Logic (`src/pynegative/ui/gallery.py`)**: Mock a file system with sidecars to test and assert that the gallery correctly filters images based on different rating criteria. (Ease: Hard)
3. **Editor Rendering and Throttling (`src/pynegative/ui/editor.py`)**: Test the asynchronous `QTimer`-based rendering loop to ensure updates are correctly throttled and processed, preventing UI lag. (Ease: Hardest)
