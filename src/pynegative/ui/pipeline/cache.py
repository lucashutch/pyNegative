class PipelineCache:
    """Manages cached stages of the image processing pipeline."""

    def __init__(self):
        # caches[resolution_key][stage_id] = (parameters_dict, numpy_array)
        self.caches = {}
        # Effect parameters that are estimated once on the preview and synced
        self.estimated_params = {}
        # Cached background pixmap for ROI optimization
        self._cached_bg_pixmap = None
        self._cached_bg_full_w = 0
        self._cached_bg_full_h = 0

    def get(self, resolution, stage_id, current_params):
        """Returns the cached array if parameters match exactly."""
        res_cache = self.caches.get(resolution, {})
        cached_data = res_cache.get(stage_id)

        if cached_data:
            cached_params, cached_array = cached_data
            # Check if all relevant parameters for this stage match
            if all(
                current_params.get(k) == cached_params.get(k) for k in cached_params
            ):
                return cached_array
        return None

    def put(self, resolution, stage_id, params, array):
        """Stores a stage in the cache."""
        if resolution not in self.caches:
            self.caches[resolution] = {}
        self.caches[resolution][stage_id] = (params.copy(), array)

    def invalidate(self, stage_id=None):
        """Invalidates stages. If stage_id is None, invalidates everything."""
        if stage_id is None:
            self.caches = {}
            self.estimated_params = {}
            self._cached_bg_pixmap = None

    def clear(self):
        self.caches = {}
        self.estimated_params = {}
        self._cached_bg_pixmap = None

    def get_cached_bg_pixmap(self):
        """Returns cached background pixmap if it exists (no param check).

        When ROI is active, we use any cached background since it's not visible.
        """
        if self._cached_bg_pixmap is None:
            return None, 0, 0
        return self._cached_bg_pixmap, self._cached_bg_full_w, self._cached_bg_full_h

    def set_cached_bg_pixmap(self, pixmap, full_w, full_h):
        """Store the cached background pixmap."""
        self._cached_bg_pixmap = pixmap
        self._cached_bg_full_w = full_w
        self._cached_bg_full_h = full_h
