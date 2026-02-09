import logging
import numpy as np

# Use 'pynegative.core' logger so messages appear alongside other core processing logs
_logger = logging.getLogger("pynegative.core")

try:
    from numba import njit, prange
except ImportError:
    _logger.warning("Numba not found, using pure python fallback decorators")

    def njit(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    prange = range


@njit(fastmath=True, cache=True, parallel=True)
def tone_map_kernel(
    img,  # float32 array (H, W, 3) - MODIFIED IN PLACE
    exposure,  # float
    contrast,  # float
    blacks,  # float (offset)
    whites,  # float (scale)
    shadows,  # float
    highlights,  # float
    saturation,  # float
    r_mult,  # float
    g_mult,  # float
    b_mult,  # float
):
    """
    Fused kernel for tone mapping.
    Combines:
    1. White Balance
    2. Exposure
    3. Contrast
    4. Tone EQ (Shadows/Highlights)
    5. Saturation
    """
    rows, cols, _ = img.shape

    # Pre-calculate constants
    inv_denom = 1.0 / (whites - blacks) if abs(whites - blacks) > 1e-6 else 1000000.0

    # Statistics tracking (will be reduced across threads)
    clipped_shadows = 0
    clipped_highlights = 0
    pixel_sum = 0.0

    # Parallelize over pixels
    for r in prange(rows):
        for c in range(cols):
            # Load pixel
            # Ptr access might be faster, but Numba optimizes array access well
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]

            # --- 1. White Balance ---
            r_val *= r_mult
            g_val *= g_mult
            b_val *= b_mult

            # --- 2. Exposure ---
            # Power of 2 pre-calculated as passed in 'exposure' multiplier?
            # In core.py it is img *= 2**exposure.
            # We will assume 'exposure' passed here is the MULTIPLIER (2**ev)
            r_val *= exposure
            g_val *= exposure
            b_val *= exposure

            # --- 3. Contrast ---
            if contrast != 1.0:
                r_val = (r_val - 0.5) * contrast + 0.5
                g_val = (g_val - 0.5) * contrast + 0.5
                b_val = (b_val - 0.5) * contrast + 0.5

            # --- 4. Tone EQ (Blacks/Whites) ---
            # Calculate Luminance (Rec 709) for Shadows/Highlights masks
            # In core.py, this is done BEFORE Blacks/Whites are applied
            lum = 0.2126 * r_val + 0.7152 * g_val + 0.0722 * b_val

            if blacks != 0.0:
                r_val -= blacks
                g_val -= blacks
                b_val -= blacks

            if whites != 1.0:
                r_val *= inv_denom
                g_val *= inv_denom
                b_val *= inv_denom

            # --- 5. Shadows / Highlights / Saturation ---

            has_sh = (shadows != 0.0) or (highlights != 0.0)
            has_sat = saturation != 1.0

            if has_sh or has_sat:
                # Use 'lum' calculated above for S/H masks

                # Shadows/Highlights
                if has_sh:
                    # Shadows
                    if shadows != 0.0:
                        # math.pow is slow? Numba optimizes ** usually
                        # s_mask = (1.0 - clip(lum,0,1)) ** 2
                        clim = lum
                        if clim < 0.0:
                            clim = 0.0
                        if clim > 1.0:
                            clim = 1.0
                        s_mask = (1.0 - clim) * (1.0 - clim)

                        factor = 1.0 + shadows * s_mask
                        r_val *= factor
                        g_val *= factor
                        b_val *= factor

                        # Recalc lum? Approximation: old lum is close enough for highlights usually,
                        # but technically we changed the pixel.
                        # core.py does not recalc lum between shadows and highlights.

                    # Highlights
                    if highlights != 0.0:
                        h_mask_val = 0.0
                        if highlights < 0.0:
                            # Recovery
                            h_mask_val = lum if lum > 0 else 0.0
                            h_mask_val = h_mask_val * h_mask_val
                            factor = 1.0 + abs(highlights) * h_mask_val
                            r_val /= factor
                            g_val /= factor
                            b_val /= factor
                        else:
                            # Boost
                            clim = lum
                            if clim < 0.0:
                                clim = 0.0
                            if clim > 1.0:
                                clim = 1.0
                            h_mask = clim * clim
                            h_term = highlights * h_mask

                            r_val = r_val * (1.0 - h_term) + h_term
                            g_val = g_val * (1.0 - h_term) + h_term
                            b_val = b_val * (1.0 - h_term) + h_term

                # Saturation
                if has_sat:
                    # Re-calc luminance required as tone curve shifted it
                    curr_lum = 0.2126 * r_val + 0.7152 * g_val + 0.0722 * b_val
                    # Clip lum for sat calc
                    if curr_lum < 0.0:
                        curr_lum = 0.0
                    if curr_lum > 1.0:
                        curr_lum = 1.0

                    r_val = curr_lum + (r_val - curr_lum) * saturation
                    g_val = curr_lum + (g_val - curr_lum) * saturation
                    b_val = curr_lum + (b_val - curr_lum) * saturation

            # Stats Accumulation (Checking before final clip but logic in core.py checks AFTER tone adjustments)
            # core.py checks: clipped_shadows = np.sum(img < 0.0), clipped_highlights = np.sum(img > 1.0)
            # This check is on the values BEFORE the final clip to [0,1].

            # Check Red
            if r_val < 0.0:
                clipped_shadows += 1
            elif r_val > 1.0:
                clipped_highlights += 1

            # Check Green
            if g_val < 0.0:
                clipped_shadows += 1
            elif g_val > 1.0:
                clipped_highlights += 1

            # Check Blue
            if b_val < 0.0:
                clipped_shadows += 1
            elif b_val > 1.0:
                clipped_highlights += 1

            # Write back (with Clipping)
            # core.py performs np.clip(img, 0.0, 1.0) at the very end.
            if r_val < 0.0:
                r_val = 0.0
            elif r_val > 1.0:
                r_val = 1.0

            if g_val < 0.0:
                g_val = 0.0
            elif g_val > 1.0:
                g_val = 1.0

            if b_val < 0.0:
                b_val = 0.0
            elif b_val > 1.0:
                b_val = 1.0

            img[r, c, 0] = r_val
            img[r, c, 1] = g_val
            img[r, c, 2] = b_val

            pixel_sum += (r_val + g_val + b_val) / 3.0

    return clipped_shadows, clipped_highlights, pixel_sum


@njit(fastmath=True, cache=True, parallel=True)
def sharpen_kernel(img, blurred, percent):
    """
    Fused unsharp mask kernel.
    sharpened = img + (img - blurred) * (percent / 100.0)
    """
    rows, cols, _ = img.shape
    factor = percent / 100.0

    for r in prange(rows):
        for c in range(cols):
            for i in range(3):
                val = img[r, c, i]
                res = val + (val - blurred[r, c, i]) * factor
                if res < 0.0:
                    res = 0.0
                elif res > 1.0:
                    res = 1.0
                img[r, c, i] = res


@njit(fastmath=True, cache=True, parallel=True)
def bilateral_kernel_yuv(
    img_yuv, strength, sigma_color_y, sigma_space_y, sigma_color_uv, sigma_space_uv
):
    """
    Bilateral filter for YUV image.
    Fuses luma and chroma denoising with different parameters.
    """
    rows, cols, _ = img_yuv.shape
    out = np.empty_like(img_yuv)

    # Pre-calculate spatial weights for luma (3x3) and chroma (5x5)
    # Luma 3x3
    s_weights_y = np.zeros((3, 3), dtype=np.float32)
    s_inv_y = -1.0 / (2.0 * sigma_space_y * sigma_space_y)
    for i in range(-1, 2):
        for j in range(-1, 2):
            s_weights_y[i + 1, j + 1] = np.exp((i * i + j * j) * s_inv_y)

    # Chroma 5x5
    s_weights_uv = np.zeros((5, 5), dtype=np.float32)
    s_inv_uv = -1.0 / (2.0 * sigma_space_uv * sigma_space_uv)
    for i in range(-2, 3):
        for j in range(-2, 3):
            s_weights_uv[i + 2, j + 2] = np.exp((i * i + j * j) * s_inv_uv)

    c_inv_y = -1.0 / (2.0 * sigma_color_y * sigma_color_y)
    c_inv_uv = -1.0 / (2.0 * sigma_color_uv * sigma_color_uv)

    for r in prange(rows):
        for c in range(cols):
            # Process Y (Luma) - 3x3 window
            y_val = img_yuv[r, c, 0]
            sum_y = 0.0
            w_sum_y = 0.0

            for i in range(-1, 2):
                rr = r + i
                if rr < 0 or rr >= rows:
                    continue
                for j in range(-1, 2):
                    cc = c + j
                    if cc < 0 or cc >= cols:
                        continue

                    p_val = img_yuv[rr, cc, 0]
                    diff = p_val - y_val
                    w = s_weights_y[i + 1, j + 1] * np.exp(diff * diff * c_inv_y)
                    sum_y += p_val * w
                    w_sum_y += w

            out[r, c, 0] = sum_y / w_sum_y if w_sum_y > 0 else y_val

            # Process U and V (Chroma) - 5x5 window
            for ch in range(1, 3):
                val = img_yuv[r, c, ch]
                sum_ch = 0.0
                w_sum_ch = 0.0

                for i in range(-2, 3):
                    rr = r + i
                    if rr < 0 or rr >= rows:
                        continue
                    for j in range(-2, 3):
                        cc = c + j
                        if cc < 0 or cc >= cols:
                            continue

                        p_val = img_yuv[rr, cc, ch]
                        diff = p_val - val
                        w = s_weights_uv[i + 2, j + 2] * np.exp(diff * diff * c_inv_uv)
                        sum_ch += p_val * w
                        w_sum_ch += w

                out[r, c, ch] = sum_ch / w_sum_ch if w_sum_ch > 0 else val

    return out


@njit(parallel=True, fastmath=True, cache=True)
def nl_means_numba(image, h=10.0, patch_size=7, search_size=21):
    """
    Optimized Numba implementation of Non-Local Means Denoising.
    Includes early exit for patch distance calculation.
    """
    rows, cols = image.shape
    output = np.zeros_like(image)

    # Calculate offsets
    pad_p = patch_size // 2
    pad_s = search_size // 2
    patch_area_inv = 1.0 / (patch_size * patch_size)

    # Safety check: if image is too small for these windows, return original
    if rows < 2 * (pad_s + pad_p) + 1 or cols < 2 * (pad_s + pad_p) + 1:
        return image

    # Pre-calculate h squared for the weight formula
    # Add a tiny epsilon to avoid division by zero if h is extremely small
    h2 = max(h * h, 1e-10)
    # Weight threshold for early exit (e.g. weight < 0.0001)
    # -log(0.0001) * h2 + 2*h2 = distance threshold
    dist_threshold = 9.21 * h2 + 2.0 * h2

    # Iterate over every pixel in the image (Parallelized)
    for i in prange(pad_s + pad_p, rows - (pad_s + pad_p)):
        for j in range(pad_s + pad_p, cols - (pad_s + pad_p)):
            total_weight = 0.0
            weighted_sum = 0.0

            # Pre-calculate central patch values if it's 3x3
            if patch_size == 3:
                p00 = image[i - 1, j - 1]
                p01 = image[i - 1, j]
                p02 = image[i - 1, j + 1]
                p10 = image[i, j - 1]
                p11 = image[i, j]
                p12 = image[i, j + 1]
                p20 = image[i + 1, j - 1]
                p21 = image[i + 1, j]
                p22 = image[i + 1, j + 1]

                # Search window
                for r in range(i - pad_s, i + pad_s + 1):
                    for c in range(j - pad_s, j + pad_s + 1):
                        # Manually unrolled 3x3 patch comparison
                        d = (
                            (p00 - image[r - 1, c - 1]) ** 2
                            + (p01 - image[r - 1, c]) ** 2
                            + (p02 - image[r - 1, c + 1]) ** 2
                            + (p10 - image[r, c - 1]) ** 2
                            + (p11 - image[r, c]) ** 2
                            + (p12 - image[r, c + 1]) ** 2
                            + (p20 - image[r + 1, c - 1]) ** 2
                            + (p21 - image[r + 1, c]) ** 2
                            + (p22 - image[r + 1, c + 1]) ** 2
                        )

                        dist = d * patch_area_inv
                        if dist <= dist_threshold:
                            weight = np.exp(-max(dist - (2.0 * h2), 0.0) / h2)
                            weighted_sum += image[r, c] * weight
                            total_weight += weight
            else:
                # General case for other patch sizes
                for r in range(i - pad_s, i + pad_s + 1):
                    for c in range(j - pad_s, j + pad_s + 1):
                        dist = 0.0
                        skip_patch = False
                        for pr in range(-pad_p, pad_p + 1):
                            for pc in range(-pad_p, pad_p + 1):
                                diff = image[i + pr, j + pc] - image[r + pr, c + pc]
                                dist += diff * diff * patch_area_inv
                                if dist > dist_threshold:
                                    skip_patch = True
                                    break
                            if skip_patch:
                                break
                        if not skip_patch:
                            weight = np.exp(-max(dist - (2.0 * h2), 0.0) / h2)
                            weighted_sum += image[r, c] * weight
                            total_weight += weight

            # Normalize and assign
            if total_weight > 0:
                output[i, j] = weighted_sum / total_weight
            else:
                output[i, j] = image[i, j]

    return output


@njit(fastmath=True, cache=True, parallel=True)
def dark_channel_kernel(img):
    """
    Calculates the dark channel of an image: min(R, G, B) for each pixel.
    """
    rows, cols, _ = img.shape
    dark = np.empty((rows, cols), dtype=np.float32)

    for r in prange(rows):
        for c in range(cols):
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]

            # min(R, G, B)
            m = r_val
            if g_val < m:
                m = g_val
            if b_val < m:
                m = b_val
            dark[r, c] = m

    return dark


@njit(fastmath=True, cache=True, parallel=True)
def dehaze_recovery_kernel(img, transmission, atmospheric_light):
    """
    Recover radiance: J(x) = (I(x) - A) / max(t(x), t0) + A
    Transmission map 'transmission' is 2D.
    """
    rows, cols, _ = img.shape
    out = np.empty_like(img)

    for r in prange(rows):
        for c in range(cols):
            t = transmission[r, c]
            if t < 0.1:
                t = 0.1

            for i in range(3):
                val = (img[r, c, i] - atmospheric_light[i]) / t + atmospheric_light[i]
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
                out[r, c, i] = val
    return out


@njit(parallel=True, fastmath=True, cache=True)
def nl_means_numba_multichannel(
    image, h=(10.0, 10.0, 10.0), patch_size=7, search_size=21
):
    """
    Multi-channel Optimized Numba NL-Means.
    Processes all channels in a single pass to improve cache locality.

    Parameters:
    - image: (H, W, C) float array (usually C=3 for YUV/RGB).
    - h: Tuple or list of floats, one strength parameter per channel.
    """
    rows, cols, channels = image.shape
    output = np.zeros_like(image)

    # Calculate offsets
    pad_p = patch_size // 2
    pad_s = search_size // 2
    patch_area_inv = 1.0 / (patch_size * patch_size)

    # Safety check: if image is too small for these windows, return original
    if rows < 2 * (pad_s + pad_p) + 1 or cols < 2 * (pad_s + pad_p) + 1:
        return image

    # Pre-calculate h squared and thresholds for EACH channel
    h2 = np.zeros(channels, dtype=np.float64)
    dist_threshold = np.zeros(channels, dtype=np.float64)

    for c in range(channels):
        val = max(h[c], 1e-10)  # Avoid div by zero
        h2[c] = val * val
        # Threshold: -log(0.0001) * h2 + 2*h2
        dist_threshold[c] = 9.21 * h2[c] + 2.0 * h2[c]

    # Iterate over every pixel (Parallelized)
    for i in prange(pad_s + pad_p, rows - (pad_s + pad_p)):
        for j in range(pad_s + pad_p, cols - (pad_s + pad_p)):
            # Explicit local variables for accumulators
            tw0, tw1, tw2 = 0.0, 0.0, 0.0
            ws0, ws1, ws2 = 0.0, 0.0, 0.0

            if patch_size == 3:
                # Specialized 3x3 path with buffered central patch
                # Channel 0
                c0t0, c1t0, c2t0 = (
                    image[i - 1, j - 1, 0],
                    image[i - 1, j, 0],
                    image[i - 1, j + 1, 0],
                )
                c0t1, c1t1, c2t1 = (
                    image[i, j - 1, 0],
                    image[i, j, 0],
                    image[i, j + 1, 0],
                )
                c0t2, c1t2, c2t2 = (
                    image[i + 1, j - 1, 0],
                    image[i + 1, j, 0],
                    image[i + 1, j + 1, 0],
                )

                # Channel 1 (if exists)
                if channels > 1:
                    c1t0_0, c1t1_0, c1t2_0 = (
                        image[i - 1, j - 1, 1],
                        image[i - 1, j, 1],
                        image[i - 1, j + 1, 1],
                    )
                    c1t0_1, c1t1_1, c1t2_1 = (
                        image[i, j - 1, 1],
                        image[i, j, 1],
                        image[i, j + 1, 1],
                    )
                    c1t0_2, c1t1_2, c1t2_2 = (
                        image[i + 1, j - 1, 1],
                        image[i + 1, j, 1],
                        image[i + 1, j + 1, 1],
                    )

                # Channel 2 (if exists)
                if channels > 2:
                    c2t0_0, c2t1_0, c2t2_0 = (
                        image[i - 1, j - 1, 2],
                        image[i - 1, j, 2],
                        image[i - 1, j + 1, 2],
                    )
                    c2t0_1, c2t1_1, c2t2_1 = (
                        image[i, j - 1, 2],
                        image[i, j, 2],
                        image[i, j + 1, 2],
                    )
                    c2t0_2, c2t1_2, c2t2_2 = (
                        image[i + 1, j - 1, 2],
                        image[i + 1, j, 2],
                        image[i + 1, j + 1, 2],
                    )

                # Search window
                for r in range(i - pad_s, i + pad_s + 1):
                    for c_off in range(j - pad_s, j + pad_s + 1):
                        # Distance for Channel 0
                        d0 = (
                            (c0t0 - image[r - 1, c_off - 1, 0]) ** 2
                            + (c1t0 - image[r - 1, c_off, 0]) ** 2
                            + (c2t0 - image[r - 1, c_off + 1, 0]) ** 2
                            + (c0t1 - image[r, c_off - 1, 0]) ** 2
                            + (c1t1 - image[r, c_off, 0]) ** 2
                            + (c2t1 - image[r, c_off + 1, 0]) ** 2
                            + (c0t2 - image[r + 1, c_off - 1, 0]) ** 2
                            + (c1t2 - image[r + 1, c_off, 0]) ** 2
                            + (c2t2 - image[r + 1, c_off + 1, 0]) ** 2
                        )
                        d0 *= patch_area_inv

                        d1, d2 = 0.0, 0.0
                        bad_count = 1 if d0 > dist_threshold[0] else 0

                        if channels > 1:
                            d1 = (
                                (c1t0_0 - image[r - 1, c_off - 1, 1]) ** 2
                                + (c1t1_0 - image[r - 1, c_off, 1]) ** 2
                                + (c1t2_0 - image[r - 1, c_off + 1, 1]) ** 2
                                + (c1t0_1 - image[r, c_off - 1, 1]) ** 2
                                + (c1t1_1 - image[r, c_off, 1]) ** 2
                                + (c1t2_1 - image[r, c_off + 1, 1]) ** 2
                                + (c1t0_2 - image[r + 1, c_off - 1, 1]) ** 2
                                + (c1t1_2 - image[r + 1, c_off, 1]) ** 2
                                + (c1t2_2 - image[r + 1, c_off + 1, 1]) ** 2
                            )
                            d1 *= patch_area_inv
                            if d1 > dist_threshold[1]:
                                bad_count += 1
                        else:
                            bad_count += 1

                        if channels > 2:
                            d2 = (
                                (c2t0_0 - image[r - 1, c_off - 1, 2]) ** 2
                                + (c2t1_0 - image[r - 1, c_off, 2]) ** 2
                                + (c2t2_0 - image[r - 1, c_off + 1, 2]) ** 2
                                + (c2t0_1 - image[r, c_off - 1, 2]) ** 2
                                + (c2t1_1 - image[r, c_off, 2]) ** 2
                                + (c2t2_1 - image[r, c_off + 1, 2]) ** 2
                                + (c2t0_2 - image[r + 1, c_off - 1, 2]) ** 2
                                + (c2t1_2 - image[r + 1, c_off, 2]) ** 2
                                + (c2t2_2 - image[r + 1, c_off + 1, 2]) ** 2
                            )
                            d2 *= patch_area_inv
                            if d2 > dist_threshold[2]:
                                bad_count += 1
                        else:
                            bad_count += 1

                        if bad_count == 3:
                            continue

                        # Accumulate
                        w0 = np.exp(-max(d0 - (2.0 * h2[0]), 0.0) / h2[0])
                        ws0 += image[r, c_off, 0] * w0
                        tw0 += w0

                        if channels > 1:
                            w1 = np.exp(-max(d1 - (2.0 * h2[1]), 0.0) / h2[1])
                            ws1 += image[r, c_off, 1] * w1
                            tw1 += w1

                        if channels > 2:
                            w2 = np.exp(-max(d2 - (2.0 * h2[2]), 0.0) / h2[2])
                            ws2 += image[r, c_off, 2] * w2
                            tw2 += w2
            else:
                # General case for other patch sizes
                for r in range(i - pad_s, i + pad_s + 1):
                    for c_off in range(j - pad_s, j + pad_s + 1):
                        dist0, dist1, dist2 = 0.0, 0.0, 0.0
                        skip_patch = False

                        for pr in range(-pad_p, pad_p + 1):
                            for pc in range(-pad_p, pad_p + 1):
                                channels_bad_count = 0
                                # Channel 0
                                if dist0 <= dist_threshold[0]:
                                    diff = (
                                        image[i + pr, j + pc, 0]
                                        - image[r + pr, c_off + pc, 0]
                                    )
                                    dist0 += diff * diff * patch_area_inv
                                    if dist0 > dist_threshold[0]:
                                        channels_bad_count += 1
                                else:
                                    channels_bad_count += 1
                                if channels > 1:
                                    if dist1 <= dist_threshold[1]:
                                        diff = (
                                            image[i + pr, j + pc, 1]
                                            - image[r + pr, c_off + pc, 1]
                                        )
                                        dist1 += diff * diff * patch_area_inv
                                        if dist1 > dist_threshold[1]:
                                            channels_bad_count += 1
                                    else:
                                        channels_bad_count += 1
                                else:
                                    channels_bad_count += 1
                                if channels > 2:
                                    if dist2 <= dist_threshold[2]:
                                        diff = (
                                            image[i + pr, j + pc, 2]
                                            - image[r + pr, c_off + pc, 2]
                                        )
                                        dist2 += diff * diff * patch_area_inv
                                        if dist2 > dist_threshold[2]:
                                            channels_bad_count += 1
                                    else:
                                        channels_bad_count += 1
                                else:
                                    channels_bad_count += 1

                                if (
                                    channels_bad_count == channels
                                ):  # Changed from `3` to `channels` for correctness
                                    skip_patch = True
                                    break
                            if skip_patch:
                                break

                        if not skip_patch:
                            # Accumulate ... (same logic as above)
                            w0 = np.exp(-max(dist0 - (2.0 * h2[0]), 0.0) / h2[0])
                            ws0 += image[r, c_off, 0] * w0
                            tw0 += w0
                            if channels > 1:
                                w1 = np.exp(-max(dist1 - (2.0 * h2[1]), 0.0) / h2[1])
                                ws1 += image[r, c_off, 1] * w1
                                tw1 += w1
                            if channels > 2:
                                w2 = np.exp(-max(dist2 - (2.0 * h2[2]), 0.0) / h2[2])
                                ws2 += image[r, c_off, 2] * w2
                                tw2 += w2

            # Normalize and assign
            output[i, j, 0] = ws0 / tw0 if tw0 > 0 else image[i, j, 0]
            if channels > 1:
                output[i, j, 1] = ws1 / tw1 if tw1 > 0 else image[i, j, 1]
            if channels > 2:
                output[i, j, 2] = ws2 / tw2 if tw2 > 0 else image[i, j, 2]

    return output
