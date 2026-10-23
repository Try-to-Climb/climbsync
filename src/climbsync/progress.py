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
L_WRIST, R_WRIST = 9, 10
L_SHOULDER, R_SHOULDER = 5, 6


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


def anchor_hand_xy(
    kpts: np.ndarray,
    bbox: np.ndarray,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """Return per-sample (x, y) of the HIGHEST wrist (= smallest y = top-reaching hand).

    This complements `anchor_xy` (lower foot). Together they capture the full
    body-extension state of the climber. Fallbacks: upper wrist → shoulder → bbox top.

    Sanity: wrist y must be <= hip y (wrist above hip on screen). If YOLO
    outputs an inverted pose, reject and fall through.
    """
    K = len(kpts)
    out = np.full((K, 2), np.nan, np.float64)
    for i in range(K):
        lw_x, lw_y, lw_c = kpts[i, L_WRIST, 0], kpts[i, L_WRIST, 1], kpts[i, L_WRIST, 2]
        rw_x, rw_y, rw_c = kpts[i, R_WRIST, 0], kpts[i, R_WRIST, 1], kpts[i, R_WRIST, 2]
        lh_y, lh_c = kpts[i, L_HIP, 1], kpts[i, L_HIP, 2]
        rh_y, rh_c = kpts[i, R_HIP, 1], kpts[i, R_HIP, 2]

        hip_ys = []
        if np.isfinite(lh_y) and lh_c >= conf_thresh:
            hip_ys.append(lh_y)
        if np.isfinite(rh_y) and rh_c >= conf_thresh:
            hip_ys.append(rh_y)
        hip_ref_y = min(hip_ys) if hip_ys else None  # highest hip = smallest y

        def wrist_ok(wy, wc):
            if not (np.isfinite(wy) and wc >= conf_thresh):
                return False
            if hip_ref_y is not None and wy > hip_ref_y + 20:
                return False  # wrist below hip = inverted pose, reject
            return True

        lw_ok = wrist_ok(lw_y, lw_c)
        rw_ok = wrist_ok(rw_y, rw_c)
        if lw_ok and rw_ok:
            if lw_y <= rw_y:
                out[i] = [lw_x, lw_y]
            else:
                out[i] = [rw_x, rw_y]
            continue
        if lw_ok:
            out[i] = [lw_x, lw_y]
            continue
        if rw_ok:
            out[i] = [rw_x, rw_y]
            continue
        # Fallback: shoulder
        ls_x, ls_y, ls_c = kpts[i, L_SHOULDER, 0], kpts[i, L_SHOULDER, 1], kpts[i, L_SHOULDER, 2]
        rs_x, rs_y, rs_c = kpts[i, R_SHOULDER, 0], kpts[i, R_SHOULDER, 1], kpts[i, R_SHOULDER, 2]
        sh_pts = []
        if np.isfinite(ls_y) and ls_c >= conf_thresh:
            sh_pts.append((ls_x, ls_y))
        if np.isfinite(rs_y) and rs_c >= conf_thresh:
            sh_pts.append((rs_x, rs_y))
        if sh_pts:
            out[i] = [float(np.mean([p[0] for p in sh_pts])),
                      float(np.mean([p[1] for p in sh_pts]))]
            continue
        if np.isfinite(bbox[i, 1]):
            cx = 0.5 * (bbox[i, 0] + bbox[i, 2])
            out[i] = [cx, bbox[i, 1]]
    return out


def combine_progress(
    p_foot: np.ndarray,
    p_hand: np.ndarray,
    hand_lead: float = 0.3,
) -> np.ndarray:
    """Scheme-2 combine: progress = foot + hand_lead * max(0, hand - foot).

    The lower foot is the physical ground truth of climber position, but when
    the hand reaches up for the next hold, the climber is "in progress" even
    though the foot has not moved yet. We allow hand-superiority to pull the
    progress forward by a fraction (`hand_lead`), smoothing the plateau when
    hand moves first and foot follows.

    hand_lead ∈ [0, 1]:  0.0 = foot-only;  0.3 = moderate (default);  1.0 = aggressive.
    """
    if len(p_foot) != len(p_hand):
        raise ValueError("p_foot and p_hand must have same length")
    forward = np.maximum(0.0, p_hand - p_foot)
    return np.clip(p_foot + hand_lead * forward, 0.0, 1.0)


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
    baseline: str = "a",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both curves using a shared reference.

    baseline:
        "a"     — use A's p1/p99 of the (cleaned) up signal (legacy behaviour).
        "b"     — use B's p1/p99 (useful if B climbed higher than A).
        "union" — use min(lo_a, lo_b) and max(hi_a, hi_b).

    Both signals are expressed as `up = frame_height - y`, so larger = higher
    on the wall. The normalization lives on the SAME physical scale (A camera
    frame, because B has already been warped through H).
    """
    ya = _clean(anchor_y_a, hampel_window, median_window, smooth_window)
    yb = _clean(anchor_y_b_in_a, hampel_window, median_window, smooth_window)
    up_a = frame_height - ya
    up_b = frame_height - yb

    lo_a = float(np.percentile(up_a, 1.0))
    hi_a = float(np.percentile(up_a, 99.0))
    lo_b = float(np.percentile(up_b, 1.0))
    hi_b = float(np.percentile(up_b, 99.0))

    if baseline == "a":
        lo, hi = lo_a, hi_a
    elif baseline == "b":
        lo, hi = lo_b, hi_b
    elif baseline == "union":
        lo, hi = min(lo_a, lo_b), max(hi_a, hi_b)
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    if hi - lo < 1e-3:
        return np.zeros_like(up_a), np.zeros_like(up_b)
    pa = np.clip((up_a - lo) / (hi - lo), 0.0, 1.0)
    pb = np.clip((up_b - lo) / (hi - lo), 0.0, 1.0)
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
    end_frac_of_peak: float = 0.95,
    baseline_samples: int = 15,
    min_hold_samples: int = 8,
) -> tuple[int, int]:
    """Detect climb window relative to each curve's own baseline.

    - baseline = median of the first `baseline_samples` samples (= standing-still).
    - k_start: first index where p stays >= baseline + rise_above_baseline for
      min_hold_samples consecutive samples.
    - k_end: first index where p >= effective_end AFTER k_start.
      effective_end = min(end_thresh, end_frac_of_peak * max(p[k0:])).
      This way a climber that only reached, say, p=0.9 (because the other
      climber went higher and set the normalization) still has a well-defined
      end: 95% of its OWN peak.
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
    peak = float(np.max(p[k0:])) if k0 < K else 0.0
    effective_end = min(end_thresh, end_frac_of_peak * peak)
    tail = np.where(p[k0:] >= effective_end)[0]
    k1 = int(k0 + tail[0]) if len(tail) else K - 1
    k1 = max(k1, k0 + 1)
    return k0, k1


def detect_completion(
    p: np.ndarray,
    k_end: int,
    sample_fps: float,
    hold_seconds: float = 3.0,
    fall_drop: float = 0.3,
) -> dict:
    """Heuristic: did the climber complete or fall?

    After `k_end` (= where the climb window ends, at the climber's peak), look
    at the next `hold_seconds` of signal. If p stays within `fall_drop` of the
    peak, the climber held the top → completed. If p drops below peak - fall_drop,
    the climber fell (or was lowered quickly, which from a sync perspective
    looks the same).

    Returns dict with:
        completed:   True/False
        peak_p:      the max p reached
        tail_min_p:  the lowest p within hold_seconds after peak
        hold_time:   actual seconds observed (may be < hold_seconds at EOF)
    """
    if k_end >= len(p) - 1:
        return {"completed": True, "peak_p": float(p[k_end]), "tail_min_p": None,
                "hold_time": 0.0, "reason": "no post-peak data"}
    hold_samples = max(1, int(round(hold_seconds * sample_fps)))
    end = min(len(p), k_end + hold_samples)
    peak_p = float(np.max(p[max(0, k_end - 2):k_end + 2]))
    tail_min = float(np.min(p[k_end:end]))
    actual_hold = (end - k_end) / sample_fps
    completed = (peak_p - tail_min) < fall_drop
    return {"completed": completed, "peak_p": peak_p, "tail_min_p": tail_min,
            "hold_time": actual_hold,
            "reason": "stayed near peak" if completed else "fell/lowered past threshold"}
