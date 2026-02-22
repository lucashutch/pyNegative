import numpy as np

from ._numba_base import njit, prange


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
    output = image.copy()

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
    output = image.copy()

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

                                if channels_bad_count == channels:
                                    skip_patch = True
                                    break
                            if skip_patch:
                                break

                        if not skip_patch:
                            # Accumulate
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
