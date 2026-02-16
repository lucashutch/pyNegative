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

        # Spatial ROI cache: most recent processed large ROI per tier
        # spatial_roi_cache[tier_name] = {rect: (x1, y1, x2, y2), params: {...}, array: np.ndarray}
        self.spatial_roi_cache = {}

    def get_spatial_roi(self, tier, requested_rect, current_heavy_params):
        """
        Attempts to find a cached processed chunk that contains the requested_rect.
        requested_rect: (x1, y1, x2, y2)
        """
        cached = self.spatial_roi_cache.get(tier)
        if not cached:
            return None

        cached_rect = cached["rect"]
        cached_params = cached["params"]
        cached_array = cached["array"]

        # 1. Check if heavy parameters match exactly
        if not all(
            current_heavy_params.get(k) == cached_params.get(k) for k in cached_params
        ):
            return None

        # 2. Check if requested_rect is entirely within cached_rect
        # requested_rect is (x1, y1, x2, y2)
        cx1, cy1, cx2, cy2 = cached_rect
        rx1, ry1, rx2, ry2 = requested_rect

        if rx1 >= cx1 and ry1 >= cy1 and rx2 <= cx2 and ry2 <= cy2:
            # ROI is within cache. Extract crop from cached_array.
            # Convert global coordinates to local cache coordinates
            lx1, ly1 = rx1 - cx1, ry1 - cy1
            lx2, ly2 = lx1 + (rx2 - rx1), ly1 + (ry2 - ry1)

            return cached_array[ly1:ly2, lx1:lx2].copy()

        return None

    def put_spatial_roi(self, tier, rect, params, array):
        """Stores the most recent processed ROI for a tier."""
        self.spatial_roi_cache[tier] = {
            "rect": rect,
            "params": params.copy(),
            "array": array.copy(),
        }

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

    def invalidate(self, stage_id=None, clear_estimated=True):
        """Invalidates stages. If stage_id is None, invalidates everything."""
        if stage_id is None:
            self.caches = {}
            self.spatial_roi_cache = {}
            if clear_estimated:
                self.estimated_params = {}
            self._cached_bg_pixmap = None

    def clear(self):
        self.caches = {}
        self.spatial_roi_cache = {}
        self.estimated_params = {}
        self._cached_bg_pixmap = None
        self._cached_bg_preprocess_key = None

    def get_cached_bg_pixmap(self, preprocess_key=None):
        """Returns cached background pixmap if it exists and preprocess_key matches.

        When ROI is active, we use any cached background since it's not visible.
        """
        if self._cached_bg_pixmap is None:
            return None, 0, 0
        if (
            preprocess_key is not None
            and self._cached_bg_preprocess_key != preprocess_key
        ):
            return None, 0, 0
        return self._cached_bg_pixmap, self._cached_bg_full_w, self._cached_bg_full_h

    def set_cached_bg_pixmap(self, pixmap, full_w, full_h, preprocess_key=None):
        """Store the cached background pixmap."""
        self._cached_bg_pixmap = pixmap
        self._cached_bg_full_w = full_w
        self._cached_bg_full_h = full_h
        self._cached_bg_preprocess_key = preprocess_key
