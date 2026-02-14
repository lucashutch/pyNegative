# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- **User Presets**: Allow saving and applying named adjustment presets.
- **Selective Copy/Paste (Sync Settings)**: specific dialog to choose *which* settings to paste (e.g. WB only, or everything except Crop/Geometry/lens, apply lens profile to all photos in selection etc..)

## Performance Optimisation

- **High-DPI Awareness**: Detect system screen scale factor (e.g. 200% for Retina) and adjust the Smart Resolution selection to provide appropriate pixel density.
- **implement point and dimensions classes** instead of passing around raw x ans y or h and w variables instead utilised classes for these (and other common types) so it is more clear what and how these are used.

## Bugs


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
- **Clipping Warnings**: Visual overlay (e.g. red/blue highlight) to show overexposed or underexposed areas.
- **Auto-Straighten Tool**: A "Spirit Level" tool to draw a line along the horizon and automatically rotate the image.
- **White Balance Picker (Eyedropper)**: Click on a neutral grey/white area in the image to instantly set the correct Temperature and Tint.
- **Hue Picker for Defringe**: Add an eyedropper to select the exact fringe color to target.

### 🗂️ Workflow & DAM (Digital Asset Management)
- **Versions / Snapshots**: Create "Virtual Copies" of an image to try different edits without duplicating the file.
- **Visual History Stack**: A list of all edit steps with the ability to jump back to any point (and branch from there).
- **Metadata Editor**: User-editable fields for Copyright, Creator, Title, and Description.
- **Compare View (Side-by-Side)**: Select two different images to view side-by-side (Reference vs Active) to choose the best one.
- **Focus Peaking**: Highlight sharp edges in the viewer (e.g. with green contours) to quickly identify which shots are in focus without zooming.
