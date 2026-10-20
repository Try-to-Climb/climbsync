"""Progress curve from pose keypoints.

Signal choice:
- Use the lower foot (max of left/right ankle y) as the "anchor height".
  Rationale: a climber's real progress is bounded by the lower foot — bbox
  center is dominated by torso/head posture (crouching, reaching) and
  gives false signal while standing on the ground.
- Fallback priority: ankle_y → hip_y → bbox bottom.
- Standing on ground: ankle at bottom of frame → large y → low progress.
- At top: ankle near middle → small y → high progress.

Normalization:
- robust p1/p99 of the anchor signal; no cummax (DTW handles non-monotone).

Smoothing:
- Savitzky-Golay with wider window to kill detection jitter.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# COCO indices
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def _ffill_bfill(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    mask = np.isnan(y)
    if mask.all():
        return y
    idx = np.where(~mask, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = y[idx]
    first = int(np.argmax(~mask))
    if first > 0:
        y[:first] = y[first]
    return y


def anchor_signal(
    kpts: np.ndarray,
    bbox: np.ndarray,
    frame_height: int,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """Return per-sample anchor_y in pixels (larger = lower on screen).

    kpts: (K,17,3); bbox: (K,4) xyxy.
    """
    xy = anchor_xy(kpts, bbox, conf_thresh)
    return xy[:, 1]


def anchor_xy(
    kpts: np.ndarray,
    bbox: np.ndarray,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """Return per-sample (anchor_x, anchor_y) in pixels.

    Preference: lower foot ankle (max y of L/R ankle) → single ankle → hip + bbox_h/2 → bbox bottom center.

    Pose sanity check: an ankle keypoint is only accepted when its y is >= hip y
    (ankle must be LOWER on screen than hip). If pose is inverted (common when
    the climber is in an extreme stretch or YOLO mislabels keypoints), we
    reject the "ankle" and fall through to the hip-based fallback.
    """
    K = len(kpts)
    out = np.full((K, 2), np.nan, np.float64)
    for i in range(K):
        la_x, la_y, la_c = kpts[i, L_ANKLE, 0], kpts[i, L_ANKLE, 1], kpts[i, L_ANKLE, 2]
        ra_x, ra_y, ra_c = kpts[i, R_ANKLE, 0], kpts[i, R_ANKLE, 1], kpts[i, R_ANKLE, 2]
        lh_x, lh_y, lh_c = kpts[i, L_HIP, 0], kpts[i, L_HIP, 1], kpts[i, L_HIP, 2]
        rh_x, rh_y, rh_c = kpts[i, R_HIP, 0], kpts[i, R_HIP, 1], kpts[i, R_HIP, 2]

        # Reference hip y to validate ankle sanity
        hip_ys = []
        if np.isfinite(lh_y) and lh_c >= conf_thresh:
            hip_ys.append(lh_y)
        if np.isfinite(rh_y) and rh_c >= conf_thresh:
            hip_ys.append(rh_y)
        hip_ref_y = max(hip_ys) if hip_ys else None  # lowest hip = largest y

        def ankle_ok(ay, ac):
            if not (np.isfinite(ay) and ac >= conf_thresh):
                return False
            if hip_ref_y is not None and ay < hip_ref_y - 10:
                # Ankle must be at or below hip. Allow 10 px slack.
                return False
            return True

        la_ok = ankle_ok(la_y, la_c)
        ra_ok = ankle_ok(ra_y, ra_c)
        if la_ok and ra_ok:
            if la_y >= ra_y:
                out[i] = [la_x, la_y]
            else:
                out[i] = [ra_x, ra_y]
            continue
        if la_ok:
            out[i] = [la_x, la_y]
            continue
        if ra_ok:
            out[i] = [ra_x, ra_y]
            continue
        if hip_ys:
            bh = bbox[i, 3] - bbox[i, 1] if np.isfinite(bbox[i, 1]) else 0
            cx_src = []
            if np.isfinite(lh_x) and lh_c >= conf_thresh:
                cx_src.append(lh_x)
            if np.isfinite(rh_x) and rh_c >= conf_thresh:
                cx_src.append(rh_x)
            mx = float(np.mean(cx_src)) if cx_src else (bbox[i, 0] + bbox[i, 2]) * 0.5
            my = float(np.mean(hip_ys)) + 0.5 * bh
            out[i] = [mx, my]
            continue
        if np.isfinite(bbox[i, 3]):
            cx = 0.5 * (bbox[i, 0] + bbox[i, 2])
            out[i] = [cx, bbox[i, 3]]
    return out


def warp_points_b_to_a(xy_b: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply H to (K,2) points; return (K,2). NaN rows preserved."""
    import cv2
    out = xy_b.copy()
    mask = ~np.isnan(xy_b).any(axis=1)
    if mask.sum() == 0:
        return out
    pts = xy_b[mask].reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(-1, 2)
    out[mask] = warped
    return out


def progress_curves_shared(
    anchor_y_a: np.ndarray,
    anchor_y_b_in_a: np.ndarray,
    frame_height: int,
    smooth_window: int = 15,
    median_window: int = 7,
    hampel_window: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both using the SAME lo/hi derived from A's valid range only.

    This is the core trick: A's lo (feet-at-start) and A's hi (feet-at-top) define
    the physical anchors on the route (via A's camera). B's warped y values are
    normalized against those same anchors, so p_A = p_B iff both are on the same
    physical row of holds.
    """
    ya = _clean(anchor_y_a, hampel_window, median_window, smooth_window)
    yb = _clean(anchor_y_b_in_a, hampel_window, median_window, smooth_window)
    up_a = frame_height - ya
    up_b = frame_height - yb
    lo = np.percentile(up_a, 1.0)
    hi = np.percentile(up_a, 99.0)
    if hi - lo < 1e-3:
        return np.zeros_like(up_a), np.zeros_like(up_b)
    pa = np.clip((up_a - lo) / (hi - lo), 0.0, 1.0)
    # NOTE: don't clip B — we want "B above A's top" or "B below A's bottom" to be visible
    pb = (up_b - lo) / (hi - lo)
    pb = np.clip(pb, 0.0, 1.0)
    return pa, pb


def _clean(y_raw: np.ndarray, hampel_window: int, median_window: int, smooth_window: int) -> np.ndarray:
    y = _ffill_bfill(y_raw.astype(np.float64))
    if hampel_window >= 5 and len(y) >= hampel_window:
        y = _hampel(y, window=hampel_window, n_sigma=3.0)
    if median_window >= 3 and median_window % 2 == 1 and len(y) >= median_window:
        from scipy.signal import medfilt
        y = medfilt(y, kernel_size=median_window)
    w = min(smooth_window, len(y))
    if w >= 5 and w % 2 == 0:
        w -= 1
    if w >= 5:
        y = savgol_filter(y, w, 2)
    return y


def _hampel(x: np.ndarray, window: int = 11, n_sigma: float = 3.0) -> np.ndarray:
    """Replace samples far from local median (MAD-based) with the local median."""
    y = x.copy()
    half = window // 2
    for i in range(len(y)):
        a = max(0, i - half)
        b = min(len(y), i + half + 1)
        med = np.nanmedian(y[a:b])
        mad = np.nanmedian(np.abs(y[a:b] - med))
        if mad > 0 and abs(y[i] - med) > n_sigma * 1.4826 * mad:
            y[i] = med
    return y


def progress_curve(
    anchor_y: np.ndarray,
    frame_height: int,
    smooth_window: int = 15,
    median_window: int = 7,
    hampel_window: int = 15,
) -> np.ndarray:
    y = _ffill_bfill(anchor_y.astype(np.float64))
    # Hampel: robust outlier replacement
    if hampel_window >= 5 and len(y) >= hampel_window:
        y = _hampel(y, window=hampel_window, n_sigma=3.0)
    # median filter to kill isolated 2-3 sample spurs
    if median_window >= 3 and median_window % 2 == 1 and len(y) >= median_window:
        from scipy.signal import medfilt
        y = medfilt(y, kernel_size=median_window)
    up = frame_height - y
    w = min(smooth_window, len(up))
    if w >= 5 and w % 2 == 0:
        w -= 1
    if w >= 5:
        up = savgol_filter(up, w, 2)
    lo = np.percentile(up, 1.0)
    hi = np.percentile(up, 99.0)
    if hi - lo < 1e-3:
        return np.zeros_like(up)
    p = (up - lo) / (hi - lo)
    return np.clip(p, 0.0, 1.0)


def trim_climb_window(
    p: np.ndarray,
    rise_above_baseline: float = 0.1,
    end_thresh: float = 0.97,
    baseline_samples: int = 15,
    min_hold_samples: int = 8,
) -> tuple[int, int]:
    """Detect climb window relative to each curve's own baseline.

    - baseline = median of the first `baseline_samples` samples (= standing-still progress).
    - k_start: first index where p stays >= baseline + rise_above_baseline for
      min_hold_samples consecutive samples.
    - k_end: first index where p >= end_thresh after k_start.
    """
    K = len(p)
    if K == 0:
        return 0, 0
    nb = min(baseline_samples, K)
    baseline = float(np.median(p[:nb]))
    start_level = baseline + rise_above_baseline
    k0 = 0
    for i in range(K - min_hold_samples + 1):
        if np.all(p[i:i + min_hold_samples] >= start_level):
            k0 = i
            break
    tail = np.where(p[k0:] >= end_thresh)[0]
    k1 = int(k0 + tail[0]) if len(tail) else K - 1
    k1 = max(k1, k0 + 1)
    return k0, k1
