"""DTW on progress curves with Sakoe-Chiba band constraint.

Signals:
- p_a: (Na,) progress of A at sample times t_a
- p_b: (Nb,) progress of B at sample times t_b

Cost:
- |p_a[i] - p_b[j]| (both already in [0,1])

Steps (monotone, no skips, with waiting):
- (i+1, j+1): advance both
- (i+1, j):   advance A, B waits   → paused_side = 'b'
- (i,   j+1): advance B, A waits   → paused_side = 'a'

Output:
- path: list of (i, j) with i ∈ [0,Na-1], j ∈ [0,Nb-1], strictly non-decreasing in both.
"""
from __future__ import annotations

import numpy as np


def dtw_align(
    p_a: np.ndarray,
    p_b: np.ndarray,
    band: float = 0.2,
) -> list[tuple[int, int]]:
    """
    band: Sakoe-Chiba window as a fraction of max(Na, Nb). Set <=0 to disable.
    """
    Na, Nb = len(p_a), len(p_b)
    INF = np.inf
    cost = np.full((Na + 1, Nb + 1), INF, dtype=np.float64)
    cost[0, 0] = 0.0

    if band > 0:
        w = max(2, int(band * max(Na, Nb)))
    else:
        w = max(Na, Nb)

    # scale factor to keep diagonal and wait steps comparable
    for i in range(1, Na + 1):
        # band around diagonal i/Na == j/Nb
        j_center = i * Nb / Na
        j_lo = max(1, int(j_center - w))
        j_hi = min(Nb, int(j_center + w))
        for j in range(j_lo, j_hi + 1):
            d = abs(p_a[i - 1] - p_b[j - 1])
            cost[i, j] = d + min(
                cost[i - 1, j - 1],   # diag
                cost[i - 1, j],       # A advances, B waits
                cost[i, j - 1],       # B advances, A waits
            )

    # backtrace
    i, j = Na, Nb
    if not np.isfinite(cost[i, j]):
        # band too tight; fall back to full
        return dtw_align(p_a, p_b, band=0.0)
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]))
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def smooth_schedule(
    schedule: list[tuple[int, int, str]],
    out_fps: float,
    min_gap_sec: float = 3.0,
) -> list[tuple[int, int, str]]:
    """Coalesce wait events so that any two wait segments are >= min_gap_sec apart.

    Design:
      - Block length = 2 * min_gap_sec * out_fps
      - For each block, compute net A/B advancement from the DTW schedule.
      - Lay out the block as: [all waits] + [normal play].
        Waits occupy the FIRST HALF (up to min_gap_sec worth), normal
        occupies the SECOND HALF (at least min_gap_sec worth).
      - Adjacent blocks: wait at start of block i+1 is at least `min_gap_sec`
        seconds after the end of block i's wait, because block i's second
        half is purely normal play.
    """
    if min_gap_sec <= 0 or not schedule:
        return schedule

    gap_frames = max(1, int(round(min_gap_sec * out_fps)))
    block_size = 2 * gap_frames
    out: list[tuple[int, int, str]] = []
    N = len(schedule)
    i = 0
    while i < N:
        j = min(i + block_size, N)
        fa_s, fb_s, _ = schedule[i]
        fa_e, fb_e, _ = schedule[j - 1]
        block_len = j - i
        delta_a = fa_e - fa_s
        delta_b = fb_e - fb_s

        if delta_a < delta_b:
            wait_tag = "wait_a"
            n_wait = delta_b - delta_a
        elif delta_b < delta_a:
            wait_tag = "wait_b"
            n_wait = delta_a - delta_b
        else:
            wait_tag = None
            n_wait = 0

        # Cap wait length at half the block (= gap_frames) to preserve
        # minimum gap property. If DTW demanded more, we lose some alignment
        # precision in this block but keep UX smooth.
        n_wait = min(n_wait, gap_frames, block_len - 1)
        n_normal = block_len - n_wait

        if wait_tag == "wait_a":
            # Lay out: n_wait frames of (fa_s, advancing fb), then n_normal both advance.
            # fb will advance delta_b frames in total; wait absorbs n_wait of them.
            fb_mid = fb_s + n_wait  # fb position when wait ends
            fa_mid = fa_s
            for k in range(n_wait):
                fb_cur = fb_s + k + 1
                out.append((fa_mid, min(fb_cur, fb_e), wait_tag))
            # Then normal play from (fa_mid, fb_mid) to (fa_e, fb_e)
            if n_normal > 1:
                step_a = (fa_e - fa_mid) / (n_normal - 1)
                step_b = (fb_e - fb_mid) / (n_normal - 1)
            else:
                step_a = step_b = 0
            for k in range(n_normal):
                out.append((
                    int(round(fa_mid + k * step_a)),
                    int(round(fb_mid + k * step_b)),
                    "",
                ))
        elif wait_tag == "wait_b":
            fa_mid = fa_s + n_wait
            fb_mid = fb_s
            for k in range(n_wait):
                fa_cur = fa_s + k + 1
                out.append((min(fa_cur, fa_e), fb_mid, wait_tag))
            if n_normal > 1:
                step_a = (fa_e - fa_mid) / (n_normal - 1)
                step_b = (fb_e - fb_mid) / (n_normal - 1)
            else:
                step_a = step_b = 0
            for k in range(n_normal):
                out.append((
                    int(round(fa_mid + k * step_a)),
                    int(round(fb_mid + k * step_b)),
                    "",
                ))
        else:
            # No wait needed, straight-line interp
            if block_len > 1:
                step_a = delta_a / (block_len - 1)
                step_b = delta_b / (block_len - 1)
            else:
                step_a = step_b = 0
            for k in range(block_len):
                out.append((
                    int(round(fa_s + k * step_a)),
                    int(round(fb_s + k * step_b)),
                    "",
                ))
        i = j
    return out


def path_to_frame_schedule(
    path: list[tuple[int, int]],
    track_a_frame_idx: np.ndarray,
    track_b_frame_idx: np.ndarray,
    fps_a: float,
    fps_b: float,
    out_fps: float,
) -> list[tuple[int, int, str]]:
    """Expand sample-level DTW path to output-frame schedule.

    Returns list of (frame_a, frame_b, wait_tag) at length ~ out_fps * duration.
    wait_tag ∈ {'', 'wait_a', 'wait_b'}.

    We sample the path uniformly in path-step index, then resample so output video
    has `out_fps` frames per second of PATH TIME, where one path step = max of
    the two sample strides, keeping real-time feel on the leading side.
    """
    # path step corresponds to advancing one sample on at least one side.
    # Define path real-time duration as path_len * sample_dt where
    # sample_dt = 1 / sample_fps. We don't know sample_fps here directly, but
    # we can take median dt from the two tracks.
    if len(path) < 2:
        return []
    # estimate sample dt per side
    dta = (track_a_frame_idx[-1] - track_a_frame_idx[0]) / max(1, (len(track_a_frame_idx) - 1)) / fps_a
    dtb = (track_b_frame_idx[-1] - track_b_frame_idx[0]) / max(1, (len(track_b_frame_idx) - 1)) / fps_b
    sample_dt = 0.5 * (dta + dtb)
    total_time = len(path) * sample_dt
    n_out = max(1, int(round(total_time * out_fps)))

    schedule: list[tuple[int, int, str]] = []
    prev_i, prev_j = path[0]
    for k in range(n_out):
        # map output index to path index
        pk = int(round(k * (len(path) - 1) / max(1, n_out - 1)))
        i, j = path[pk]
        fa = int(track_a_frame_idx[i])
        fb = int(track_b_frame_idx[j])
        tag = ""
        if pk > 0:
            pi, pj = path[pk - 1]
            if i == pi and j != pj:
                tag = "wait_a"
            elif j == pj and i != pi:
                tag = "wait_b"
        schedule.append((fa, fb, tag))
        prev_i, prev_j = i, j
    return schedule
