"""Rendering: side-by-side and overlay.

Design notes:
- We hold two VideoCapture objects and seek forward only (set CAP_PROP_POS_FRAMES
  is slow on some codecs, so we cache and read sequentially where possible).
- Skeletons are interpolated from the nearest POSE SAMPLE around the output frame,
  which keeps rendering decoupled from pose sampling fps.
- Output is written via imageio-ffmpeg (H.264).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import PoseTrack

# COCO skeleton edges
_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),             # arms
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
    (5, 6), (11, 12), (5, 11), (6, 12),          # torso
    (0, 5), (0, 6),                               # head→shoulders
]


class FrameReader:
    """Sequential-forward frame cache. Falls back to seek if request goes backwards."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        self.cur = -1
        self._last: np.ndarray | None = None

    def read(self, idx: int) -> np.ndarray:
        if idx == self.cur and self._last is not None:
            return self._last
        if idx < self.cur:
            # seek backward (rare: only when "wait" freezes but internal cur advanced)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.cur = idx - 1
        while self.cur < idx:
            ret = self.cap.grab()
            if not ret:
                break
            self.cur += 1
        ret, frame = self.cap.retrieve()
        if not ret or frame is None:
            # as a last resort, seek
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            self.cur = idx
        self._last = frame
        return frame

    def release(self):
        self.cap.release()


def _nearest_valid_sample(track: PoseTrack, frame_idx: int, max_dist: int = 30) -> int | None:
    """Return the index of the nearest sample with has_person=True within max_dist frames."""
    if len(track.frame_idx) == 0:
        return None
    diffs = np.abs(track.frame_idx - frame_idx)
    valid_mask = track.has_person
    if not valid_mask.any():
        return None
    diffs_masked = np.where(valid_mask, diffs, np.iinfo(np.int64).max)
    k = int(np.argmin(diffs_masked))
    if diffs_masked[k] > max_dist:
        return None
    return k


def _kpts_at_frame(track: PoseTrack, frame_idx: int) -> np.ndarray | None:
    """Nearest valid sample's keypoints (K,17,3). Returns None if no valid sample nearby."""
    k = _nearest_valid_sample(track, frame_idx)
    if k is None:
        return None
    return track.kpts[k]


def _draw_skeleton(
    img: np.ndarray,
    kpts: np.ndarray,
    color: tuple[int, int, int],
    H: np.ndarray | None = None,
    conf_thresh: float = 0.3,
) -> None:
    pts = kpts[:, :2].copy()
    confs = kpts[:, 2]
    if H is not None:
        pts_h = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        pts = pts_h
    for i in range(len(pts)):
        if confs[i] < conf_thresh:
            continue
        x, y = pts[i]
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), 4, color, -1, lineType=cv2.LINE_AA)
    for a, b in _EDGES:
        if confs[a] < conf_thresh or confs[b] < conf_thresh:
            continue
        xa, ya = pts[a]
        xb, yb = pts[b]
        if np.any(np.isnan([xa, ya, xb, yb])):
            continue
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)


def _info_bar(w: int, h_bar: int, text: str) -> np.ndarray:
    bar = np.zeros((h_bar, w, 3), np.uint8)
    cv2.putText(bar, text, (12, h_bar - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bar


COLOR_A = (0, 220, 0)       # BGR green
COLOR_B = (255, 60, 220)    # BGR magenta


@dataclass
class RenderConfig:
    out_fps: float = 30.0
    show_skeleton: bool = True
    draw_wait_badge: bool = True
    draw_diag_lines: bool = False       # horizontal foot-height reference lines
    draw_progress_overlay: bool = False  # show p_a / p_b in info bar


def _horiz_line(img, y, color, dashed=False, thickness=3):
    """Draw a horizontal line across img at pixel y with a white halo for visibility."""
    import cv2
    h, w = img.shape[:2]
    if y is None or np.isnan(y):
        return
    y = int(round(y))
    if y < 0 or y >= h:
        return
    # White halo (slightly thicker + white) for visibility against colorful walls
    if not dashed:
        cv2.line(img, (0, y), (w, y), (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.line(img, (0, y), (w, y), color, thickness, cv2.LINE_AA)
    else:
        step = 24
        for x in range(0, w, step):
            x2 = min(x + step // 2, w)
            cv2.line(img, (x, y), (x2, y), (255, 255, 255), thickness + 2, cv2.LINE_AA)
            cv2.line(img, (x, y), (x2, y), color, thickness, cv2.LINE_AA)


def render_side_by_side(
    track_a: PoseTrack,
    track_b: PoseTrack,
    H_b2a: np.ndarray | None,  # used for diagnostic overlay only
    schedule: list[tuple[int, int, str]],
    out_path: Path,
    cfg: RenderConfig,
    anchor_xy_a: np.ndarray | None = None,  # (K_a, 2) A frame coords
    anchor_xy_b: np.ndarray | None = None,  # (K_b, 2) B frame coords (raw)
    anchor_xy_b_in_a: np.ndarray | None = None,  # (K_b, 2) B warped to A
    p_a_curve: np.ndarray | None = None,     # (K_a,)
    p_b_curve: np.ndarray | None = None,     # (K_b,)
) -> None:
    """Both videos shown as-is, placed side by side. Only timing is synced.

    B is letterboxed to A's height (keeping its own aspect ratio) if sizes differ.
    """
    import imageio_ffmpeg

    W, Hh = track_a.width, track_a.height
    # B rendered in its own aspect, scaled to A's height
    scale_b = Hh / track_b.height
    Wb = int(round(track_b.width * scale_b))
    bar_h = 36
    out_w = W + Wb
    out_h = Hh + bar_h

    writer = imageio_ffmpeg.write_frames(
        str(out_path),
        (out_w, out_h),
        fps=cfg.out_fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)

    ra = FrameReader(track_a.video_path)
    rb = FrameReader(track_b.video_path)
    try:
        n = len(schedule)
        for idx, (fa, fb, tag) in enumerate(schedule):
            frame_a = ra.read(fa)
            frame_b = rb.read(fb)
            if frame_a is None:
                frame_a = np.zeros((Hh, W, 3), np.uint8)
            if frame_b is None:
                frame_b = np.zeros((track_b.height, track_b.width, 3), np.uint8)
            if (frame_b.shape[1], frame_b.shape[0]) != (Wb, Hh):
                frame_b_r = cv2.resize(frame_b, (Wb, Hh))
            else:
                frame_b_r = frame_b.copy()

            if cfg.show_skeleton:
                kpa = _kpts_at_frame(track_a, fa)
                if kpa is not None:
                    _draw_skeleton(frame_a, kpa, COLOR_A)
                kpb = _kpts_at_frame(track_b, fb)
                if kpb is not None:
                    # scale keypoints into resized B canvas
                    S = np.array([[scale_b, 0, 0], [0, scale_b, 0], [0, 0, 1]], np.float64)
                    _draw_skeleton(frame_b_r, kpb, COLOR_B, H=S)

            # Diagnostic foot-height reference lines
            pA_val = pB_val = None
            if cfg.draw_diag_lines or cfg.draw_progress_overlay:
                ka = _nearest_valid_sample(track_a, fa)
                kb = _nearest_valid_sample(track_b, fb)
                if cfg.draw_diag_lines:
                    if anchor_xy_a is not None and ka is not None:
                        y_a = anchor_xy_a[ka, 1]
                        _horiz_line(frame_a, y_a, COLOR_A, dashed=False)
                    if anchor_xy_b is not None and kb is not None:
                        y_b = anchor_xy_b[kb, 1]
                        _horiz_line(frame_b_r, y_b * scale_b, COLOR_B, dashed=False)
                    # B's foot projected into A's coord frame → dashed on A side
                    if anchor_xy_b_in_a is not None and kb is not None:
                        y_b_in_a = anchor_xy_b_in_a[kb, 1]
                        _horiz_line(frame_a, y_b_in_a, COLOR_B, dashed=True)
                if cfg.draw_progress_overlay:
                    if p_a_curve is not None and ka is not None and ka < len(p_a_curve):
                        pA_val = float(p_a_curve[ka])
                    if p_b_curve is not None and kb is not None and kb < len(p_b_curve):
                        pB_val = float(p_b_curve[kb])

            if cfg.draw_wait_badge and tag == "wait_a":
                cv2.putText(frame_a, "WAIT", (W // 2 - 60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            if cfg.draw_wait_badge and tag == "wait_b":
                cv2.putText(frame_b_r, "WAIT", (Wb // 2 - 60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

            top = np.concatenate([frame_a, frame_b_r], axis=1)
            t_a = fa / track_a.fps
            t_b = fb / track_b.fps
            info_parts = [f"out {idx+1}/{n}", f"A t={t_a:5.1f}s f={fa}",
                          f"B t={t_b:5.1f}s f={fb}"]
            if pA_val is not None and pB_val is not None:
                diff = abs(pA_val - pB_val)
                info_parts.append(f"pA={pA_val:.2f} pB={pB_val:.2f} dp={diff:.2f}")
            if tag:
                info_parts.append(tag)
            info = "   ".join(info_parts)
            bar = _info_bar(out_w, bar_h, info)
            full = np.concatenate([top, bar], axis=0)
            writer.send(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
        ra.release()
        rb.release()


def render_overlay(
    track_a: PoseTrack,
    track_b: PoseTrack,
    H_b2a: np.ndarray,
    schedule: list[tuple[int, int, str]],
    out_path: Path,
    cfg: RenderConfig,
    alpha: float = 0.5,
) -> None:
    """Only one canvas: A's. B is warped onto A with transparency, skeletons overlaid."""
    import imageio_ffmpeg

    W, Hh = track_a.width, track_a.height
    bar_h = 36
    out_h = Hh + bar_h
    writer = imageio_ffmpeg.write_frames(
        str(out_path),
        (W, out_h),
        fps=cfg.out_fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)

    ra = FrameReader(track_a.video_path)
    rb = FrameReader(track_b.video_path)
    try:
        n = len(schedule)
        for idx, (fa, fb, tag) in enumerate(schedule):
            frame_a = ra.read(fa)
            frame_b = rb.read(fb)
            if frame_a is None:
                frame_a = np.zeros((Hh, W, 3), np.uint8)
            if frame_b is None:
                frame_b = np.zeros((Hh, W, 3), np.uint8)
            frame_b_w = cv2.warpPerspective(frame_b, H_b2a, (W, Hh))
            blended = cv2.addWeighted(frame_a, 1 - alpha, frame_b_w, alpha, 0)

            if cfg.show_skeleton:
                kpa = _kpts_at_frame(track_a, fa)
                if kpa is not None:
                    _draw_skeleton(blended, kpa, COLOR_A)
                kpb = _kpts_at_frame(track_b, fb)
                if kpb is not None:
                    _draw_skeleton(blended, kpb, COLOR_B, H=H_b2a)
            if cfg.draw_wait_badge and tag:
                side = "A" if tag == "wait_a" else "B"
                cv2.putText(blended, f"WAIT {side}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
            t_a = fa / track_a.fps
            t_b = fb / track_b.fps
            info = (f"out {idx+1}/{n}   A t={t_a:5.1f}s   B t={t_b:5.1f}s   {tag}")
            bar = _info_bar(W, bar_h, info)
            full = np.concatenate([blended, bar], axis=0)
            writer.send(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
        ra.release()
        rb.release()
