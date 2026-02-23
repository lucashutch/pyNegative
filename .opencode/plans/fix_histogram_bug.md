# Fix Histogram Bug Plan

## Problem Description
The histogram widget shows "No Data" when opened, even though an image is loaded. This happens because enabling the histogram does not trigger a re-calculation if the image processing pipeline considers the current view "up-to-date" (cached).

## Root Cause Analysis
- `ImageProcessingPipeline.set_histogram_enabled(True)` simply sets a flag and calls `request_update()`.
- `_process_pending_update()` checks if visible tiles are already in `_tile_cache`.
- If tiles are cached (status `done`), no workers are spawned.
- Histogram calculation only happens inside `ImageProcessorWorker`.
- Therefore, enabling the histogram on a static image results in no worker running, and thus no histogram calculation.

## Solution Plan
Modify `set_histogram_enabled` to invalidate the cache when enabling the histogram. This forces a re-render of the visible tiles, ensuring that at least one worker runs and calculates the histogram (via `needs_lowres=True` logic).

## Tasks
- [x] Create reproduction test case to confirm that enabling histogram does not invalidate cache.
- [x] Modify `src/pynegative/ui/imageprocessing.py`: Update `set_histogram_enabled` to increment `_current_render_state_id` and clear `_tile_cache`.
- [x] Verify fix with reproduction test.
- [x] Run existing tests to ensure no regressions.
- [x] Integrate reproduction test into `tests/ui/test_imageprocessing.py` as a permanent regression test.
- [x] Update `TODO.md` to mark the bug as fixed.

## Verification
- Run `pytest tests/repro_issue_histogram.py` -> Should pass.
- Run `pytest tests/ui/pipeline/test_worker.py` -> Should pass.
