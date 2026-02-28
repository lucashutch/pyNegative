# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- **User Presets**: Allow saving and applying named adjustment presets.

## Performance Optimisation

- **High-DPI Awareness**: Detect system screen scale factor (e.g. 200% for Retina) and adjust the Smart Resolution selection to provide appropriate pixel density.

## Bugs

- scaling issue when zooming with compare photos feature. One size gets stretched or squished instead of both being scaled equally.
- Visual artefacts when zoomed out and tone adjusted or at most resolutions with the sharpening filter enabled.
- fix the image comparison view. the left side/unedited side is too dark.
- remove any automatic adjustments and associated logic when opening a RAW file (e.g. auto WB, auto exposure) and just show the unedited image until the user makes adjustments.
- **fix vignetting**: currently when applying vignetting correction with the lens profile the corners get darker instead of brighter.
- **Crop UI Sync with Viewport Rendering**: If the UI needs to support active zooming/panning while cropping, we might need to improve the floating pixmap coordinate translation (currently viewport rendering is paused while cropping).

## Testing Improvement Areas

---

## Brainstorming: Potential Future Features

### 🎨 Advanced Editing Tools
- **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- **Tone Curves**: Full RGB and Luma curve editor for precise control over contrast and color balance.
- **Local Adjustments (Masking)**:
  - **Radial & Linear Gradients**: Apply exposure/contrast/wb to specific areas.
  - **Brush Tool**: Paint-in adjustments.
- **Spot Removal / Healing**: Basic tool to correct dust spots and skin blemishes.
- **Clipping Warnings**: Visual overlay (e.g. red/blue highlight) to show overexposed or underexposed areas.
- **Auto-Straighten Tool**: A "Spirit Level" tool to draw a line along the horizon and automatically rotate the image.
- **White Balance Picker (Eyedropper)**: Click on a neutral grey/white area in the image to instantly set the correct Temperature and Tint.
- **Hue Picker for Defringe**: Add an eyedropper to select the exact fringe color to target.

### 🗂️ Workflow & DAM (Digital Asset Management)
- **Metadata Editor**: User-editable fields for Copyright, Creator, Title, and Description.
- **Focus Peaking**: Highlight sharp edges in the viewer (e.g. with green contours) to quickly identify which shots are in focus without zooming.
