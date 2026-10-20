"""Build track summaries from bytetrack jsonl.

Each summary describes one track_id with features suitable for Jev's role classifier.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FRAME_W = 720
FRAME_H = 1280
FRAME_AREA = FRAME_W * FRAME_H


def summarize_tracks(bt_jsonl_path: Path, sample_fps: float = 3.0) -> list[dict]:
    per_track = defaultdict(list)
    with open(bt_jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            per_track[r["track_id"]].append(r)

    out = []
    for tid, rows in per_track.items():
        rows.sort(key=lambda r: r["sample_idx"])
        bboxes = np.array([r["bbox_xyxy"] for r in rows])
        confs = np.array([r["conf"] for r in rows])
        samples = np.array([r["sample_idx"] for r in rows])
        cx = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
        cy = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        area = w * h
        y_range = float(cy.max() - cy.min())
        x_range = float(cx.max() - cx.min())
        net_up = float(cy[0] - cy[-1])  # positive = climbed up (y decreased)
        median_area_frac = float(np.median(area) / FRAME_AREA)
        median_cy_frac = float(np.median(cy) / FRAME_H)  # 0 = top, 1 = bottom
        median_cx_frac = float(np.median(cx) / FRAME_W)
        duration_sec = (samples[-1] - samples[0] + 1) / sample_fps

        out.append({
            "track_id": int(tid),
            "n_samples": int(len(rows)),
            "duration_seconds": round(duration_sec, 1),
            "first_sample": int(samples[0]),
            "last_sample": int(samples[-1]),
            "first_time_s": round(samples[0] / sample_fps, 1),
            "last_time_s": round(samples[-1] / sample_fps, 1),
            "y_range_frac_of_frame": round(y_range / FRAME_H, 3),
            "x_range_frac_of_frame": round(x_range / FRAME_W, 3),
            "net_vertical_motion_frac_of_frame": round(net_up / FRAME_H, 3),  # +=up
            "median_bbox_area_frac": round(median_area_frac, 3),
            "median_center_y_frac": round(median_cy_frac, 3),  # 0=top-of-frame
            "median_center_x_frac": round(median_cx_frac, 3),
            "mean_confidence": round(float(confs.mean()), 2),
            "first_center": [round(float(cx[0]), 1), round(float(cy[0]), 1)],
            "last_center": [round(float(cx[-1]), 1), round(float(cy[-1]), 1)],
        })
    out.sort(key=lambda r: -r["y_range_frac_of_frame"])
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        tracks = summarize_tracks(Path(p))
        print(f"\n=== {Path(p).name} ===")
        for t in tracks:
            print(json.dumps(t, ensure_ascii=False))
