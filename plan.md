# Performance Improvement Plan: Multi-Threading for the Edit View

This document outlines a phased approach to improve the performance and responsiveness of the image editor in pyNegative.

## Phase 1: Single-Worker Background Processing

The primary goal of this phase is to eliminate UI sluggishness during image editing by moving all heavy processing off the main GUI thread.

*   **Task List:**
    *   [ ] Create a new `ImageProcessorSignals` class inheriting from `QObject` to define signals for communicating results from the worker thread.
    *   [ ] Create a new `ImageProcessorWorker` class inheriting from `QRunnable` to perform the image processing.
    *   [ ] Move the image processing logic (tone mapping, sharpening, denoising) from `ImageProcessingPipeline.update_preview` into the `run()` method of the `ImageProcessorWorker`.
    *   [ ] Modify `ImageProcessingPipeline` to use the application's existing `QThreadPool`.
    *   [ ] In `ImageProcessingPipeline`, replace the direct call to `update_preview` with logic that creates, new `ImageProcessorWorker`.
    *   [ ] Connect the `finished` signal from the worker to a new slot in `ImageProcessingPipeline` that takes the processed pixmaps and updates the UI via the `previewUpdated` signal.
    *   [ ] Ensure the existing throttling mechanism is adapted to prevent queuing too many workers when a slider is dragged rapidly.

## Phase 2: Tile-Based Parallel CPU Rendering

Building on Phase 1, this phase will further accelerate processing by parallelizing the work across all available CPU cores.

*   **Task List:**
    *   [ ] Modify the worker creation logic in `ImageProcessingPipeline` to divide the visible image area (ROI) into a grid of smaller tiles.
    *   [ ] Implement logic to add an overlapping border to each tile to ensure seamless results from filters (e.g., sharpening).
    *   [ ] Create a specialized `ImageTileProcessorWorker(QRunnable)` for processing a single tile.
    *   [ ] Update `ImageProcessingPipeline` to dispatch a worker for each tile to the `QThreadPool`.
    *   [ ] Implement a mechanism to collect the results from all tile workers.
    *   [ ] Create a new component responsible for reassembling the processed tiles into a single pixmap for display.
    *   [ ] Update the UI with the fully reassembled image.

## Phase 3: Defer GPU Acceleration

This phase is currently deferred due to significant challenges with cross-platform compatibility and dependency management.

*   **Considerations:**
    *   **Technology:** Would require using a library like CuPy (NVIDIA-only) or a more complex, low-level graphics API like Vulkan or Metal.
    *   **Challenges:** Relying on vendor-specific technology like CUDA would violate the project's cross-platform goals. The required dependencies are large and can be difficult for end-users to install.
    *   **Future:** This could be revisited if a stable, easy-to-deploy, and truly cross-platform GPU compute framework becomes available for Python.
