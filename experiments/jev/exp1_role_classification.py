"""Experiment 1: Role classification.

Ask Jev to classify each track as climber / belayer_or_bystander / passerby_close /
noise. Compare against ground truth and our heuristic.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_summary import summarize_tracks

API_URL = "https://api.typesafe.ai/v1/systemone"
API_KEY = os.environ["TYPESAFE_API_KEY"]


# Hand ground truth based on our understanding of the videos.
# Format: {video_name: {track_id: label}}
GROUND_TRUTH = {
    "VID_20261003_200654": {
        5: "climber",     # ascent main
        23: "climber",    # top portion (same person, ID switched)
        43: "climber",    # descent (same person)
        0: "passerby_close",   # 98.5% bbox area, mid-video
        1: "passerby_close",   # 99% bbox area, static
        2: "passerby_or_noise",     # 10 samples, 3s, small bbox, moving horizontally
        30: "noise",      # 8 samples near top, tiny motion
        41: "passerby_close",  # 98% bbox, static
        46: "passerby_close",  # 99% bbox, 4 samples
        28: "passerby_close",  # 99% bbox
    },
    "VID_20261003_203017": {
        10: "climber",    # ascent main
        53: "climber",    # top transition (same person)
        63: "climber",    # descent (same person)
        2: "climber_start_or_belayer",  # moves horizontally at bottom early - could be climber starting or belayer
        70: "belayer_or_bystander",     # at bottom, horizontal motion
        0: "noise",       # 6 samples spanning 106s = detection flickers
        36: "noise",      # 6 samples, 2s
        73: "noise",      # 2 samples
        3: "noise",       # 2 samples
        50: "noise",      # 2 samples
        5: "noise",       # 1 sample
        7: "noise",       # 1 sample
        14: "noise",      # 1 sample
        20: "noise",      # 1 sample
    },
}


# Our heuristic's classification: just a function of thresholds
def heuristic_classify(track: dict) -> str:
    if track["median_bbox_area_frac"] > 0.5:
        return "passerby_close"
    if track["n_samples"] < 8:
        return "noise"
    if track["y_range_frac_of_frame"] < 0.08:
        return "belayer_or_bystander"
    return "climber"


OPTIONS = {
    "climber": "A person actively climbing the wall. Vertical motion is the dominant movement, bbox stays small-to-medium (fits a human body on the wall), persists for many seconds, bbox center spans a large vertical range over time.",
    "belayer_or_bystander": "A person on the ground belaying or watching. Mostly stationary vertically, bbox center stays low (near frame bottom), may drift horizontally. Could also be a passer-by walking at normal distance.",
    "passerby_close": "A person very close to the camera (bbox occupies most of the frame, > 50% area). Appears briefly as they walk past the lens.",
    "noise": "Detection noise: too few observations (< 8 samples) or barely-visible fragment. Not a meaningful human track.",
}


def classify_with_jev(track: dict, scene_context: str) -> dict:
    """Call Jev to classify one track. Returns the full response."""
    state = {
        "scene_context": scene_context,
        "video_resolution": "720x1280 (portrait)",
        "track_summary": track,
    }
    payload = {
        "state": json.dumps(state, ensure_ascii=False),
        "model": "jev-latest",
        "questions": {
            "role": {
                "type": "choice",
                "instructions": (
                    "Classify this person-tracking fragment by its ROLE in the climbing scene. "
                    "The 'track_summary' field gives aggregate statistics about one bounding-box "
                    "track across the video (y_range = vertical movement range, "
                    "net_vertical_motion = net upward displacement where positive=upward, "
                    "median_bbox_area_frac = how much of the frame the person typically occupies, "
                    "median_center_y_frac = 0 means top of frame, 1 means bottom)."
                ),
                "criteria": OPTIONS,
            }
        },
    }
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main():
    repo = Path("/home/exekixl/climbing")
    raw_dir = repo / "out_v15" / "raw"
    results = []

    for video_name in ["VID_20261003_200654", "VID_20261003_203017"]:
        bt_path = raw_dir / f"bytetrack_{video_name}.jsonl"
        tracks = summarize_tracks(bt_path)
        scene = (
            "Indoor climbing gym. A single climber ascends a vertical route tracked by a "
            "fixed-position camera at ground level pointing up at the wall. A belayer is on the ground. "
            "Occasionally other people walk very close past the lens."
        )
        for tr in tracks:
            tid = tr["track_id"]
            if tid not in GROUND_TRUTH.get(video_name, {}):
                continue
            truth = GROUND_TRUTH[video_name][tid]
            heur = heuristic_classify(tr)
            print(f"\n--- {video_name} tid={tid} truth={truth} heur={heur}")
            try:
                resp = classify_with_jev(tr, scene)
                answer = resp["answers"]["role"]
                print(f"  Jev: choice={answer['choice']}  conf={answer['confidence']:.2f}  "
                      f"probs={answer['probabilities']}")
                results.append({
                    "video": video_name, "track_id": tid,
                    "truth": truth, "heuristic": heur,
                    "jev_choice": answer["choice"],
                    "jev_confidence": answer["confidence"],
                    "jev_probabilities": answer["probabilities"],
                })
                time.sleep(0.2)  # be nice to API
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"video": video_name, "track_id": tid, "truth": truth,
                                "heuristic": heur, "error": str(e)})

    out = Path(__file__).parent / "exp1_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary table
    print("\n\n=== Summary ===")
    print(f"{'video':<22} {'tid':>4} {'truth':<28} {'heur':<24} {'jev':<24} {'conf':>5}")
    print("-" * 110)
    heur_hits = 0; jev_hits = 0; total = 0
    for r in results:
        if "error" in r: continue
        total += 1
        heur_ok = r["heuristic"] == r["truth"] or (
            r["truth"] in ("climber_start_or_belayer", "passerby_or_noise") and
            r["heuristic"] in ("climber", "belayer_or_bystander", "passerby_close", "noise"))
        jev_ok = r["jev_choice"] == r["truth"] or (
            r["truth"] in ("climber_start_or_belayer", "passerby_or_noise") and
            r["jev_choice"] in ("climber", "belayer_or_bystander", "passerby_close", "noise"))
        heur_hits += heur_ok; jev_hits += jev_ok
        print(f"{r['video']:<22} {r['track_id']:>4} {r['truth']:<28} {r['heuristic']:<24} "
              f"{r['jev_choice']:<24} {r['jev_confidence']:.2f}")
    print(f"\nHeuristic strict-match: {heur_hits}/{total}")
    print(f"Jev strict-match:       {jev_hits}/{total}")
    print(f"(Ambiguous ground truths like 'climber_start_or_belayer' get a free pass.)")


if __name__ == "__main__":
    main()
