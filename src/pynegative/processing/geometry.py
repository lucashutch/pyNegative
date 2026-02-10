import math
import cv2
import numpy as np


def apply_geometry(img, rotate=0.0, crop=None, flip_h=False, flip_v=False):
    """
    Applies geometric transformations: Flip -> Rotation -> Crop.

    Args:
        img: numpy array
        rotate: float (degrees, CCW. Negative values rotate clockwise)
        crop: tuple (left, top, right, bottom) as normalized coordinates (0.0-1.0).
              The crop coordinates are relative to the FLIPPED and ROTATED image.
        flip_h: bool, mirror horizontally
        flip_v: bool, mirror vertically
    """
    if img is None:
        return None

    # 1. Apply Flip
    if flip_h or flip_v:
        # flipCode: 0 for x-axis, 1 for y-axis, -1 for both
        flip_code = -1 if (flip_h and flip_v) else (1 if flip_h else 0)
        img = cv2.flip(img, flip_code)

    # 2. Apply Rotation
    if abs(rotate) > 0.01:
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        # Use INTER_CUBIC for rotation quality
        M = cv2.getRotationMatrix2D(center, rotate, 1.0)
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        img = cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    # 3. Apply Crop
    if crop is not None:
        h, w = img.shape[:2]
        c_left, c_top, c_right, c_bottom = crop
        x1, y1 = int(c_left * w), int(c_top * h)
        x2, y2 = int(c_right * w), int(c_bottom * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]

    return img


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
        float(max(0.0, min(1.0, c_left))),
        float(max(0.0, min(1.0, c_top))),
        float(max(0.0, min(1.0, c_right))),
        float(max(0.0, min(1.0, c_bottom))),
    )
