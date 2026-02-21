import math
import cv2
import numpy as np


class GeometryResolver:
    """
    Unified geometry engine that composes all affine transformations into a single matrix.
    Pillar B: Unified Geometry Engine
    """

    def __init__(self, full_w: int, full_h: int):
        self.full_w = full_w
        self.full_h = full_h
        self.matrix = np.eye(3, dtype=np.float32)

    def reset(self):
        """Resets the transformation matrix to identity."""
        self.matrix = np.eye(3, dtype=np.float32)

    def flip(self, horizontal: bool, vertical: bool):
        """Applies flipping to the matrix."""
        if not horizontal and not vertical:
            return

        m = np.eye(3, dtype=np.float32)
        if horizontal:
            m[0, 0] = -1
            m[0, 2] = self.full_w - 1
        if vertical:
            m[1, 1] = -1
            m[1, 2] = self.full_h - 1

        self.matrix = m @ self.matrix

    def rotate(self, angle_deg: float, expand: bool = True):
        """Applies rotation around the center of the current coordinate space."""
        if abs(angle_deg) < 1e-4:
            return

        cx, cy = self.full_w / 2.0, self.full_h / 2.0

        # 1. Standard rotation matrix around origin
        angle_rad = math.radians(angle_deg)  # Matches cv2 CCW
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        r = np.array(
            [[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32
        )

        # 2. Handle expansion if requested
        tx, ty = 0, 0
        if expand:
            # Calculate new dimensions
            new_w = int(abs(self.full_h * sin_a) + abs(self.full_w * cos_a))
            new_h = int(abs(self.full_h * cos_a) + abs(self.full_w * sin_a))

            # Adjust translation to keep centered in new canvas
            tx = (new_w / 2.0) - cx
            ty = (new_h / 2.0) - cy

            self.full_w = new_w
            self.full_h = new_h

        # Compose: Translation to origin -> Rotate -> Translation back + expansion shift
        t1 = np.eye(3, dtype=np.float32)
        t1[0, 2] = -cx
        t1[1, 2] = -cy

        t2 = np.eye(3, dtype=np.float32)
        t2[0, 2] = cx + tx
        t2[1, 2] = cy + ty

        m = t2 @ r @ t1
        self.matrix = m @ self.matrix

    def get_inverse_matrix(self) -> np.ndarray:
        """Returns the inverse 3x3 matrix."""
        return np.linalg.inv(self.matrix)

    def get_output_size(self) -> tuple[int, int]:
        """Returns current bounding box dimensions."""
        return int(self.full_w), int(self.full_h)

    def crop(self, left: float, top: float, right: float, bottom: float):
        """
        Applies a crop. Coordinates are normalized (0.0-1.0) relative to
        the CURRENT bounding box of the transformed image.
        """
        if left == 0.0 and top == 0.0 and right == 1.0 and bottom == 1.0:
            return

        # Calculate translation based on normalized coordinates
        tx = left * self.full_w
        ty = top * self.full_h

        # Update dimensions
        new_w = (right - left) * self.full_w
        new_h = (bottom - top) * self.full_h

        m = np.eye(3, dtype=np.float32)
        m[0, 2] = -tx
        m[1, 2] = -ty

        self.matrix = m @ self.matrix
        self.full_w = new_w
        self.full_h = new_h

    def get_matrix_2x3(self) -> np.ndarray:
        """Returns the 2x3 affine matrix for OpenCV."""
        return self.matrix[:2, :]

    def get_fused_maps(
        self, lens_map_x: np.ndarray, lens_map_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fuses the provided lens distortion maps with the current affine transformation.
        Returns (map_x, map_y) suitable for cv2.remap().
        Pillar B: Unified Geometry Engine (Task 2.2)
        """
        out_w, out_h = self.get_output_size()
        M = self.get_matrix_2x3()

        # Warp the maps themselves. Since the maps contain source coordinates,
        # we interpolate them linearly to get the fused mapping.
        fused_x = cv2.warpAffine(
            lens_map_x,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0,
        )
        fused_y = cv2.warpAffine(
            lens_map_y,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0,
        )

        return fused_x, fused_y

    def resolve(self, rotate=0.0, crop=None, flip_h=False, flip_v=False, expand=True):
        """
        Composes all transforms into the internal matrix.
        Matches the sequential order: Flip -> Rotate -> Crop.
        """
        self.reset()
        self.flip(flip_h, flip_v)
        self.rotate(rotate, expand=expand)
        if crop:
            self.crop(*crop)
        return self.get_matrix_2x3()


def apply_geometry(img, rotate=0.0, crop=None, flip_h=False, flip_v=False):
    """
    Applies geometric transformations: Flip -> Rotation -> Crop.
    Unified via GeometryResolver.
    """
    if img is None:
        return None

    h, w = img.shape[:2]
    resolver = GeometryResolver(w, h)
    resolver.resolve(
        rotate=rotate, crop=crop, flip_h=flip_h, flip_v=flip_v, expand=True
    )

    M = resolver.get_matrix_2x3()
    out_w, out_h = resolver.get_output_size()

    # Ensure output size is at least 1x1
    out_w = max(1, int(round(out_w)))
    out_h = max(1, int(round(out_h)))

    return cv2.warpAffine(
        img,
        M,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def calculate_max_safe_crop(w, h, angle_deg, aspect_ratio=None):
    """
    Calculates the maximum normalized crop (l, t, r, b) that fits inside
    a rotated rectangle of size (w, h) rotated by angle_deg.

    If aspect_ratio is provided, the result will respect it.
    Otherwise, it uses the original image aspect ratio (w/h).

    Returns (l, t, r, b) as normalized coordinates relative to
    the EXPANDED rotated canvas.
    """
    phi = abs(math.radians(angle_deg))

    if phi < 1e-4:
        return (0.0, 0.0, 1.0, 1.0)

    if aspect_ratio is None:
        aspect_ratio = w / h

    # Formula for largest axis-aligned rectangle of aspect ratio 'AR'
    # inside a rotated rectangle of size (w, h) and angle 'phi'.

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # We need to satisfy:
    # 1. w_prime * cos + h_prime * sin <= w
    # 2. w_prime * sin + h_prime * cos <= h
    # and w_prime = h_prime * aspect_ratio

    h_prime_1 = w / (aspect_ratio * cos_phi + sin_phi)
    h_prime_2 = h / (aspect_ratio * sin_phi + cos_phi)

    h_prime = min(h_prime_1, h_prime_2)
    w_prime = h_prime * aspect_ratio

    # Expanded canvas size
    W = w * cos_phi + h * sin_phi
    H = w * sin_phi + h * cos_phi

    # Normalized dimensions relative to expanded canvas
    nw = w_prime / W
    nh = h_prime / H

    # Center it
    c_left = (1.0 - nw) / 2
    c_top = (1.0 - nh) / 2
    c_right = c_left + nw
    c_bottom = c_top + nh

    # Clamp to safe range just in case of float errors
    return (
        float(np.clip(c_left, 0.0, 1.0)),
        float(np.clip(c_top, 0.0, 1.0)),
        float(np.clip(c_right, 0.0, 1.0)),
        float(np.clip(c_bottom, 0.0, 1.0)),
    )
