# Plan: Fix Highlights vs Whites Rendering

## Problem
- Highlights and Whites sliders behave similarly, both acting as gains.
- Highlight recovery (negative slider) is ineffective and actually increases values for over-exposed pixels (>1.0).
- `rawpy` auto-brighten clips RAW data before it reaches the tone mapper.

## Changes
1. **RAW Loading (`src/pynegative/core.py`)**
    - Set `no_auto_bright=True` in `rawpy.postprocess` to prevent early clipping.
    - Set `bright=1.0` and `user_flip=0` for consistency.
2. **Auto-Exposure (`src/pynegative/core.py`)**
    - Update `calculate_auto_exposure` to use a percentile-based approach (98th percentile) to handle linear RAW data.
3. **Tone Mapping (`src/pynegative/core.py`)**
    - Remove hard clipping of luminance before mask calculation.
    - Implement compression-based highlight recovery for `highlights < 0`.
    - Keep `Whites` as a linear white-point adjustment (Levels) but ensure it distinguishes from `Highlights` by its linear nature.
    - Improve `Saturation` calculation by using updated luminance after tone adjustments.
4. **Thumbnail Fallback (`src/pynegative/core.py`)**
    - Ensure `extract_thumbnail` fallback uses `no_auto_bright=False` so thumbnails aren't too dark in the gallery.

## Verification
- [x] Create reproduction script for highlight recovery.
- [x] Verify recovery of >1.0 pixels.
- [x] Run existing unit tests (`pytest tests/core/test_tone_mapping.py`).
- [x] Verify Whites vs Highlights distinction.
