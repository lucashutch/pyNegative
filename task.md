# Geometric Tools (Crop & Rotate) - Feature Implementation Status

## Overview
This document tracks the implementation of the Geometric Tools feature set for pyNegative. The goal was to provide users with non-destructive capabilities to re-compose (crop) and straighten (rotate) their RAW images directly within the editor.

## Why This Was Implemented
- **Composition Control:** Users needed a way to remove unwanted edge distractions or focus on a specific subject after the shot was taken.
- **Horizon Correction:** "Tilted" horizons are a common issue in photography. The rotation tool allows for precise straightening.
- **Non-Destructive Workflow:** All geometric changes are saved as metadata (sidecar files) and applied dynamically, preserving the original RAW data.

## Implemented Features

### 1. Visual Crop Tool
- **Interactive Overlay:** A draggable, resizable rectangle overlay on the image.
- **Rule of Thirds Grid:** Automatically appears to help with composition.
- **Handles:** Round, semi-transparent handles for resizing from corners and sides.
- **"Done" Workflow:** The tool clearly indicates when it is active, and changes are applied only when confirmed (or toggled off).
- **Auto-Initialization:** Defaults to selecting the full image if no previous crop exists.

### 2. Rotation Control
- **Slider:** Range strictly limited to ±45 degrees for horizon correction.
- **Fine-Tuning:** Added `+` and `-` buttons for 0.1-degree precision adjustments.
- **Reset:** Dedicated button to instantly return rotation to 0.
- **Conditional Visibility:** Rotation controls are hidden by default and only appear when the Crop Tool is active, reducing UI clutter.

### 3. Core Processing
- **OpenCV Optimization:** Rotations use OpenCV's linear algebra (`warpAffine`) with Nearest Neighbor interpolation during preview for high performance (60fps+ interaction), falling back to high-quality Lanczos/Bicubic for final export.
- **Pipeline Integration:** Geometric transforms are applied *after* tone mapping but *before* the final view scaling, ensuring consistent results.
- **Persistence:** Crop coordinates (normalized 0.0-1.0) and rotation angle are saved to `.json` sidecar files and restored upon reloading.

### 4. UI Improvements
- **Editable Sliders:** All editing sliders (Exposure, Contrast, etc., + Rotation) now include editable text fields for entering exact values.
- **Unit Display:** Sliders now support unit labels (e.g., "deg").

## Remaining Tasks / Future Improvements

- [ ] **rotate handles on crop tool** : add handles so a user an use their mouse to drag the handles to rotate the image
- [ ] **Flip/Mirror:** Add buttons to flip the image horizontally/vertically.
- [ ] **Auto-Straighten:** Implement computer vision logic to detect horizon lines and suggest an auto-rotation angle.
- [ ] **Arbitrary Rotation:** Allow for 90-degree rotations (Portrait/Landscape toggle) outside of the fine-tune slider.
- [ ] **Validation:** Add unit tests for corner cases in coordinate mapping (e.g., extreme zooms + rotation).
