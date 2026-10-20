"""Per-frame person detection + pose on sampled frames, with ByteTrack tracking.

Design:
- Use ultralytics YOLO.track() with persist=True to get stable track IDs.
- Collect ALL tracks per sample, then post-hoc pick the "climber" track:
    score = y_range * active_duration * upward_motion
  Rationale:
    - climber has the largest vertical travel
    - climber is present for most of the climb duration
    - climber's net y-velocity is negative (going up in image coords)
  This filters out passers-by (brief appearance, small y range) and
  belayers / observers (large y range only on error, small net motion).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PoseTrack:
    video_path: Path
    fps: float
    width: int
    height: int
    total_frames: int
    frame_idx: np.ndarray  # (K,) original frame indices sampled
    t: np.ndarray  # (K,) timestamps in seconds
    bbox: np.ndarray  # (K, 4) xyxy; NaN if no detection
    kpts: np.ndarray  # (K, 17, 3) x, y, conf; NaN if no detection
    has_person: np.ndarray  # (K,) bool
    chosen_track_id: int | None = None
    chosen_track_ids: list[int] | None = None
    diag_candidates: list | None = None  # list of (track_id, y_range, n_samples, net_up)

    @property
    def centroid(self) -> np.ndarray:
        cx = (self.bbox[:, 0] + self.bbox[:, 2]) * 0.5
        cy = (self.bbox[:, 1] + self.bbox[:, 3]) * 0.5
        return np.stack([cx, cy], axis=1)

    @property
    def bbox_area(self) -> np.ndarray:
        w = np.clip(self.bbox[:, 2] - self.bbox[:, 0], 0, None)
        h = np.clip(self.bbox[:, 3] - self.bbox[:, 1], 0, None)
        return w * h


def _pick_climber_tracks(
    per_track_obs: dict[int, list[tuple[int, np.ndarray]]],
    frame_height: int,
    frame_width: int,
    y_range_frac_thresh: float = 0.08,
    y_range_frac_thresh_stitch: float = 0.015,
    min_samples: int = 8,
    max_bbox_frac: float = 0.5,
    stitch_gap_samples: int = 60,
    stitch_dist_px: float = 300.0,
) -> tuple[list[int], list]:
    """Return list of climber track_ids, selected by stitching ID fragments spatially.

    Strategy:
      1. Compute per-track summary (first/last sample, first/last center, y_range,
         median bbox area).
      2. Reject tracks whose median bbox area exceeds `max_bbox_frac * frame area`
         (= extreme close-up passer-by).
      3. Pick seed = largest y_range candidate.
      4. Chain backward and forward: find other candidate tracks whose temporal gap
         to the current chain's endpoint is <= stitch_gap_samples AND spatial
         distance between the endpoints' centers is <= stitch_dist_px.
    """
    summaries = {}
    frame_area = float(frame_width) * frame_height
    for tid, obs in per_track_obs.items():
        if len(obs) < min_samples:
            continue
        samples = np.array([s for s, _ in obs])
        bboxes = np.stack([b for _, b in obs])
        order = np.argsort(samples)
        samples = samples[order]; bboxes = bboxes[order]
        cx = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
        cy = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        y_range = float(cy.max() - cy.min())
        net_up = float(cy[0] - cy[-1])
        median_area_frac = float(np.median(areas) / frame_area)
        summaries[tid] = {
            "first_sample": int(samples[0]),
            "last_sample": int(samples[-1]),
            "first_center": np.array([cx[0], cy[0]]),
            "last_center": np.array([cx[-1], cy[-1]]),
            "y_range": y_range,
            "n": len(samples),
            "net_up": net_up,
            "median_area_frac": median_area_frac,
        }

    debug = [(tid, s["y_range"], s["n"], s["net_up"], s["median_area_frac"])
             for tid, s in summaries.items()]
    debug.sort(key=lambda x: -x[1])

    # Filter: too close to camera = passer-by special-case
    seed_pool = {
        tid: s for tid, s in summaries.items()
        if s["median_area_frac"] <= max_bbox_frac
        and s["y_range"] >= y_range_frac_thresh * frame_height
    }
    # Pool used for stitching: more permissive — accepts short fragments
    # (e.g. climber temporarily lost to a passer-by close to camera).
    stitch_pool = {
        tid: s for tid, s in summaries.items()
        if s["median_area_frac"] <= max_bbox_frac
        and s["y_range"] >= y_range_frac_thresh_stitch * frame_height
    }
    if not seed_pool:
        return [], debug

    # Seed selection logic:
    # - If any track has clear ascent (net_up > 15% frame), use the biggest among those.
    # - Else: exclude pure-descent fragments (net_up < -30% of own y_range) and pick
    #   the track with the longest coverage (most samples) tied on y_range — this
    #   handles the case where ByteTrack kept the whole climb+lower in one id
    #   (net_up ≈ 0 but y_range large and n_samples large).
    strong_ascent = {t: s for t, s in seed_pool.items()
                     if s["net_up"] > 0.15 * frame_height
                     and s["n"] >= 30}  # at least ~10s of ascent evidence
    if strong_ascent:
        pool = strong_ascent
    else:
        pool = {t: s for t, s in seed_pool.items()
                if s["net_up"] > -0.3 * s["y_range"]}
        if not pool:
            pool = seed_pool
    seed_tid = max(pool.keys(),
                   key=lambda t: (min(pool[t]["n"], 100) * pool[t]["y_range"],
                                  pool[t]["n"]))
    chain = [seed_tid]
    used = {seed_tid}
    candidates = stitch_pool

    # Chain forward
    while True:
        cur = chain[-1]
        cur_last_sample = candidates[cur]["last_sample"]
        cur_last_center = candidates[cur]["last_center"]
        best = None
        best_cost = float("inf")
        for tid, s in candidates.items():
            if tid in used:
                continue
            gap = s["first_sample"] - cur_last_sample
            if gap < -5 or gap > stitch_gap_samples:
                continue
            d = float(np.linalg.norm(s["first_center"] - cur_last_center))
            if d > stitch_dist_px:
                continue
            cost = max(0, gap) + 0.2 * d
            if cost < best_cost:
                best_cost = cost; best = tid
        if best is None:
            break
        chain.append(best); used.add(best)

    # Chain backward
    while True:
        cur = chain[0]
        cur_first_sample = candidates[cur]["first_sample"]
        cur_first_center = candidates[cur]["first_center"]
        best = None; best_cost = float("inf")
        for tid, s in candidates.items():
            if tid in used:
                continue
            gap = cur_first_sample - s["last_sample"]
            if gap < -5 or gap > stitch_gap_samples:
                continue
            d = float(np.linalg.norm(s["last_center"] - cur_first_center))
            if d > stitch_dist_px:
                continue
            cost = max(0, gap) + 0.2 * d
            if cost < best_cost:
                best_cost = cost; best = tid
        if best is None:
            break
        chain.insert(0, best); used.add(best)

    return chain, debug


def _merge_tracks_by_proximity(
    sample_dets: list[dict[int, tuple[np.ndarray, np.ndarray]]],
    chosen_tids: list[int],
    max_jump_px: float = 300.0,
    stale_reset_samples: int = 8,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """For each sample, pick ONE (bbox, kpts) from among the chosen track_ids.

    Strategy:
    - If exactly one chosen tid present → take it.
    - If multiple → pick the one spatially closest to the previous selected
      bbox center. First pick uses the largest bbox.
    - If a candidate's center jumps more than max_jump_px from prev, reject this
      sample unless `prev_center` is older than `stale_reset_samples` samples
      ago — in which case we treat the pick as a fresh seed (ignore distance).
    - If none → None.
    """
    K = len(sample_dets)
    out: list[tuple[np.ndarray, np.ndarray] | None] = [None] * K
    prev_center: np.ndarray | None = None
    prev_sample_idx: int = -(stale_reset_samples + 1)
    chosen_set = set(chosen_tids)
    for i, dets in enumerate(sample_dets):
        cand = {tid: d for tid, d in dets.items() if tid in chosen_set}
        if not cand:
            out[i] = None
            continue
        is_stale = (prev_center is None) or (i - prev_sample_idx > stale_reset_samples)
        if is_stale:
            # first pick (or long gap): take the largest bbox, no distance check
            tid = max(cand.keys(), key=lambda t: (cand[t][0][2] - cand[t][0][0])
                                                 * (cand[t][0][3] - cand[t][0][1]))
        else:
            def dist(t):
                b = cand[t][0]
                c = np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])])
                return float(np.linalg.norm(c - prev_center))
            tid = min(cand.keys(), key=dist)
            if dist(tid) > max_jump_px:
                out[i] = None
                continue
        bb, kp = cand[tid]
        out[i] = (bb, kp)
        prev_center = np.array([0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])])
        prev_sample_idx = i
    return out


def run_pose(
    video_path: str | Path,
    model_path: str | Path,
    sample_fps: float = 5.0,
    conf: float = 0.05,
    device: str = "cpu",
    verbose: bool = False,
    use_tracking: bool = True,
    dump_dir: str | Path | None = None,
) -> PoseTrack:
    from ultralytics import YOLO
    import json

    video_path = Path(video_path)
    yolo_fp = None
    bt_fp = None
    if dump_dir is not None:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem
        yolo_fp = open(dump_dir / f"yolo_{stem}.jsonl", "w")
        bt_fp = open(dump_dir / f"bytetrack_{stem}.jsonl", "w")

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps / sample_fps))

    model = YOLO(str(model_path))

    # Per-track-id → list of (sample_idx, bbox)
    per_track_bbox: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    # Per-sample → dict[track_id] = (bbox, kpts)
    sample_dets: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    frame_idx_list: list[int] = []

    f = 0
    s_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if f % stride == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            if use_tracking:
                res = model.track(
                    frame, conf=conf, device=device, verbose=False, classes=[0],
                    persist=True, tracker="bytetrack.yaml",
                )[0]
            else:
                res = model.predict(
                    frame, conf=conf, device=device, verbose=False, classes=[0],
                )[0]
            frame_idx_list.append(f)
            dets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                if use_tracking and res.boxes.id is not None:
                    ids = res.boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.arange(len(xyxy))
                if res.keypoints is not None and res.keypoints.data is not None:
                    kps = res.keypoints.data.cpu().numpy()
                    if kps.shape[-1] == 2:
                        kps = np.concatenate(
                            [kps, np.ones((kps.shape[0], 17, 1), np.float32)], axis=-1
                        )
                else:
                    kps = np.full((len(xyxy), 17, 3), np.nan, np.float32)
                for i in range(len(xyxy)):
                    tid = int(ids[i])
                    bbox = xyxy[i].astype(np.float32)
                    kp = kps[i].astype(np.float32)
                    dets[tid] = (bbox, kp)
                    per_track_bbox[tid].append((s_idx, bbox))

                    # Dump raw YOLO (no id) and ByteTrack (with id) per detection
                    if yolo_fp is not None:
                        yolo_fp.write(json.dumps({
                            "sample_idx": s_idx,
                            "frame_idx": f,
                            "det_idx": i,
                            "bbox_xyxy": bbox.tolist(),
                            "conf": float(res.boxes.conf[i].item()) if res.boxes.conf is not None else None,
                            "keypoints": kp.tolist(),
                        }) + "\n")
                    if bt_fp is not None:
                        bt_fp.write(json.dumps({
                            "sample_idx": s_idx,
                            "frame_idx": f,
                            "track_id": tid,
                            "bbox_xyxy": bbox.tolist(),
                            "conf": float(res.boxes.conf[i].item()) if res.boxes.conf is not None else None,
                            "keypoints": kp.tolist(),
                        }) + "\n")
            sample_dets.append(dets)
            if verbose and s_idx % 50 == 0 and s_idx > 0:
                print(f"  [{video_path.name}] sampled {s_idx} @ f={f}/{total}, "
                      f"tracks seen={len(per_track_bbox)}")
            s_idx += 1
        f += 1
    cap.release()

    # Pick climber tracks (possibly multiple ID fragments)
    chosen_tids, candidates = _pick_climber_tracks(per_track_bbox, height, width)
    if verbose:
        print(f"  [{video_path.name}] all candidates (sorted by y_range):")
        for row in candidates[:10]:
            print(f"    tid={row[0]} y_range={row[1]:.0f}px n_samples={row[2]} "
                  f"net_up={row[3]:+.0f}px bbox_area_frac={row[4]:.3f}")
        print(f"  [{video_path.name}] stitched climber track_ids={chosen_tids}")

    # Merge chosen tracks per sample by spatial proximity
    merged = _merge_tracks_by_proximity(sample_dets, chosen_tids)

    # Assemble output arrays from merged tracks
    K = len(sample_dets)
    bbox_out = np.full((K, 4), np.nan, np.float32)
    kpts_out = np.full((K, 17, 3), np.nan, np.float32)
    has = np.zeros(K, dtype=bool)
    for i, m in enumerate(merged):
        if m is not None:
            bb, kp = m
            bbox_out[i] = bb
            kpts_out[i] = kp
            has[i] = True

    frame_idx = np.array(frame_idx_list, dtype=np.int64)
    if yolo_fp is not None:
        yolo_fp.close()
    if bt_fp is not None:
        bt_fp.close()
    return PoseTrack(
        video_path=video_path,
        fps=float(fps),
        width=width,
        height=height,
        total_frames=total,
        frame_idx=frame_idx,
        t=frame_idx.astype(np.float64) / fps,
        bbox=bbox_out,
        kpts=kpts_out,
        has_person=has,
        chosen_track_id=chosen_tids[0] if chosen_tids else None,
        chosen_track_ids=chosen_tids,
        diag_candidates=candidates,
    )
