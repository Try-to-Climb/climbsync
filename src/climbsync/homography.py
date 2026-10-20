"""Estimate a single homography H mapping video B pixels to video A pixels.

Strategy:
- Pick N "clean" frames from each video — frames where the climber is small
  or absent, so most of the frame is wall.
- Match SIFT features between every A-frame and every B-frame, with the climber
  bbox (dilated) masked out.
- RANSAC-estimate H per pair, keep ones with enough inliers.
- Aggregate: take the H with the most inliers (not a weighted-median across H's,
  since averaging homographies in matrix space is ill-defined).

Fallback: if no pair yields ≥ min_inliers inliers, return None.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import PoseTrack


@dataclass
class HomographyResult:
    H: np.ndarray | None  # (3,3) B→A, or None if failed
    inliers: int
    num_pairs_tried: int
    best_pair: tuple[int, int] | None  # (frame_a, frame_b)


def _grab_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _person_mask(shape_hw: tuple[int, int], bbox: np.ndarray, pad: float = 0.2) -> np.ndarray:
    h, w = shape_hw
    mask = np.full((h, w), 255, np.uint8)
    if np.any(np.isnan(bbox)):
        return mask
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    mask[y1:y2, x1:x2] = 0
    return mask


def _select_clean_samples(track: PoseTrack, n: int = 6) -> list[int]:
    """Return sample indices k with the smallest person bbox area (incl. no-detection)."""
    area = track.bbox_area.copy()
    area[~track.has_person] = 0.0  # no person = cleanest
    order = np.argsort(area)  # ascending
    return [int(k) for k in order[:n]]


def estimate_homography(
    track_a: PoseTrack,
    track_b: PoseTrack,
    n_samples: int = 6,
    min_inliers: int = 25,
    ransac_thresh: float = 4.0,
    verbose: bool = False,
) -> HomographyResult:
    sift = cv2.SIFT_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    sel_a = _select_clean_samples(track_a, n=n_samples)
    sel_b = _select_clean_samples(track_b, n=n_samples)

    # pre-compute SIFT per selected frame
    def feats(track: PoseTrack, k: int):
        frame = _grab_frame(track.video_path, int(track.frame_idx[k]))
        if frame is None:
            return None, None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = _person_mask(gray.shape, track.bbox[k])
        kp, des = sift.detectAndCompute(gray, mask)
        return frame, kp, des

    feats_a = [(k, *feats(track_a, k)) for k in sel_a]
    feats_b = [(k, *feats(track_b, k)) for k in sel_b]

    best = HomographyResult(H=None, inliers=0, num_pairs_tried=0, best_pair=None)
    for (ka, _fa, kpa, desa) in feats_a:
        if desa is None or len(kpa) < 10:
            continue
        for (kb, _fb, kpb, desb) in feats_b:
            if desb is None or len(kpb) < 10:
                continue
            best.num_pairs_tried += 1
            # knnMatch with Lowe ratio
            raw = bf.knnMatch(desb, desa, k=2)  # B→A direction
            good = []
            for m in raw:
                if len(m) < 2:
                    continue
                m1, m2 = m
                if m1.distance < 0.75 * m2.distance:
                    good.append(m1)
            if len(good) < 20:
                continue
            src = np.float32([kpb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kpa[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
            if H is None:
                continue
            inl = int(mask.sum())
            if verbose:
                print(f"  pair A[{ka}]↔B[{kb}]: {inl}/{len(good)} inliers")
            if inl > best.inliers:
                best = HomographyResult(
                    H=H.astype(np.float64),
                    inliers=inl,
                    num_pairs_tried=best.num_pairs_tried,
                    best_pair=(int(track_a.frame_idx[ka]), int(track_b.frame_idx[kb])),
                )

    if best.inliers < min_inliers:
        best.H = None
    return best


def warp_b_to_a(frame_b: np.ndarray, H: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    w, h = out_wh
    return cv2.warpPerspective(frame_b, H, (w, h), flags=cv2.INTER_LINEAR)


def diagnostic_plot(
    track_a: PoseTrack,
    track_b: PoseTrack,
    result: HomographyResult,
    out_path: Path,
) -> None:
    """Save a side-by-side showing B's four corners projected onto A."""
    import matplotlib.pyplot as plt

    if result.H is None or result.best_pair is None:
        return
    fa, fb = result.best_pair
    frame_a = _grab_frame(track_a.video_path, fa)
    frame_b = _grab_frame(track_b.video_path, fb)
    if frame_a is None or frame_b is None:
        return
    h, w = frame_a.shape[:2]
    corners_b = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_b, result.H).reshape(-1, 2)
    warped_b = warp_b_to_a(frame_b, result.H, (w, h))

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"A frame {fa}")
    poly = np.vstack([warped_corners, warped_corners[:1]])
    axes[0].plot(poly[:, 0], poly[:, 1], "y-", lw=2)
    axes[1].imshow(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"B frame {fb}")
    blend = cv2.addWeighted(frame_a, 0.5, warped_b, 0.5, 0)
    axes[2].imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"A ⊕ warp(B)  inliers={result.inliers}")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
