# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- **User Presets**: Allow saving and applying named adjustment presets.
- **Selective Copy/Paste (Sync Settings)**: specific dialog to choose *which* settings to paste (e.g. WB only, or everything except Crop/Geometry).

## Performance Optimisation

- **General Code Cleanup**: [DONE] Analyzed the codebase and removed redundant, duplicate, or unused code.
- **use faster nlmeans denoise on zoomed out images**: [DONE] Implemented tier-aware dynamic quality scaling (Ultra Fast for previews, True Quality for 1:1 view).
- **Zoom Controls**: Implement "Fit to Screen" button and "+" / "-" manual zoom buttons for the editor and gallery preview viewports.
- **High-DPI Awareness**: Detect system screen scale factor (e.g. 200% for Retina) and adjust the Smart Resolution selection to provide appropriate pixel density.

## Bugs

- **Add Maximum CI Test Time of 2 mins**: [DONE] Added `timeout-minutes: 2` to CI jobs. Added `--verbose` and step-level timeouts to investigate Python 3.14 hangs.
- **windows ui launches with a terminal window**: When launching the installed windows app, it opens a terminal window inthe background which launches the app, if the window is closed the app also closes. To be considered a more polished app we need to not show the terminal window.

## Testing Improvement Areas

- **All areas need more tests!**
- **Generate Coverage Report**: Generate a coverage report to see where we need to add more tests, and add a coverage badge to the README if possible.

---

## Brainstorming: Potential Future Features

### 🎨 Advanced Editing Tools
- **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- **Tone Curves**: Full RGB and Luma curve editor for precise control over contrast and color balance.
- **Local Adjustments (Masking)**:
  - **Radial & Linear Gradients**: Apply exposure/contrast/wb to specific areas.
  - **Brush Tool**: Paint-in adjustments.
- **Spot Removal / Healing**: Basic tool to correct dust spots and skin blemishes.
- **Lens Corrections**:
  - Auto-correction for geometric distortion and chromatic aberration (possibly via Lensfun).
  - Manual Vignette controls.
- **Clipping Warnings**: Visual overlay (e.g. red/blue highlight) to show overexposed or underexposed areas.
- **Auto-Straighten Tool**: A "Spirit Level" tool to draw a line along the horizon and automatically rotate the image.
- **White Balance Picker (Eyedropper)**: Click on a neutral grey/white area in the image to instantly set the correct Temperature and Tint.
- **Defringe Tool**: A specific tool to remove purple/green color fringing (Chromatic Aberration) from high-contrast edges.

### 🗂️ Workflow & DAM (Digital Asset Management)
- **Versions / Snapshots**: Create "Virtual Copies" of an image to try different edits without duplicating the file.
- **Visual History Stack**: A list of all edit steps with the ability to jump back to any point (and branch from there).
- **Metadata Editor**: User-editable fields for Copyright, Creator, Title, and Description.
- **Compare View (Side-by-Side)**: Select two different images to view side-by-side (Reference vs Active) to choose the best one.
- **Focus Peaking**: Highlight sharp edges in the viewer (e.g. with green contours) to quickly identify which shots are in focus without zooming.
