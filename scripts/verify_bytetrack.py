"""Independently run ByteTrack on YOLO jsonl output and emit a new bytetrack jsonl.

Reads yolo_*.jsonl, feeds detections frame-by-frame into BYTETracker, writes a
bytetrack_verify_*.jsonl with the recovered track_ids.

Compare the result against the pipeline's bytetrack_*.jsonl to verify that
ByteTrack really is "produced from YOLO output".

Usage:
    python scripts/verify_bytetrack.py \
        out_v15/raw/yolo_VID_20261003_200654.jsonl \
        out_v15/raw/bytetrack_verify_VID_20261003_200654.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from ultralytics.trackers.byte_tracker import BYTETracker


def load_yolo_jsonl(path: Path):
    """Return list of per-sample detection dicts.

    Each entry: {sample_idx, frame_idx, dets: [{bbox_xyxy, conf, kpts, det_idx}, ...]}
    """
    per_sample = defaultdict(lambda: {"frame_idx": None, "dets": []})
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            s = r["sample_idx"]
            per_sample[s]["frame_idx"] = r["frame_idx"]
            per_sample[s]["dets"].append(r)
    # ordered
    out = []
    for s in sorted(per_sample):
        out.append((s, per_sample[s]["frame_idx"], per_sample[s]["dets"]))
    return out


class _FakeBoxes:
    """Minimal numpy-backed shim exposing the attributes BYTETracker.update expects."""

    def __init__(self, dets):
        if dets:
            xyxy = np.array([d["bbox_xyxy"] for d in dets], dtype=np.float32)
            conf = np.array([d["conf"] if d["conf"] is not None else 1.0 for d in dets],
                            dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros(len(xyxy), dtype=np.float32)
        # xywh
        if len(xyxy):
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            cx = xyxy[:, 0] + w / 2
            cy = xyxy[:, 1] + h / 2
            self.xywh = np.stack([cx, cy, w, h], axis=1)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, mask):
        new = _FakeBoxes.__new__(_FakeBoxes)
        new.xyxy = self.xyxy[mask]
        new.conf = self.conf[mask]
        new.cls = self.cls[mask]
        new.xywh = self.xywh[mask]
        return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yolo_jsonl", type=Path)
    ap.add_argument("out_jsonl", type=Path)
    args = ap.parse_args()

    # ByteTrack default config (match ultralytics' bytetrack.yaml)
    cfg = SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    tracker = BYTETracker(cfg)

    samples = load_yolo_jsonl(args.yolo_jsonl)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_tracked = 0
    with open(args.out_jsonl, "w") as fp:
        for s_idx, f_idx, dets in samples:
            boxes = _FakeBoxes(dets)
            # BYTETracker expects boxes-like object; img is only needed for GMC
            tracks = tracker.update(boxes, img=None)
            # tracks: (N, 8) [x1, y1, x2, y2, track_id, score, cls, idx]
            for row in tracks:
                x1, y1, x2, y2, tid, score, cls, idx = row.tolist()
                tid = int(tid); idx = int(idx)
                if idx < 0 or idx >= len(dets):
                    continue
                d = dets[idx]
                fp.write(json.dumps({
                    "sample_idx": s_idx,
                    "frame_idx": f_idx,
                    "track_id": tid,
                    "bbox_xyxy": d["bbox_xyxy"],
                    "conf": d["conf"],
                    "keypoints": d["keypoints"],
                }) + "\n")
                n_tracked += 1
    print(f"wrote {n_tracked} tracked detections → {args.out_jsonl}")


if __name__ == "__main__":
    main()
