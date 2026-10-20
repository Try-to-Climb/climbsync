"""Experiment 2: ID recall via Jev.

ByteTrack sometimes drops a track and assigns a new ID to the same person.
We want to know: given two track fragments, can Jev tell whether they are
the same person (= ID should be recalled) or different people?

Test cases are constructed from the known fragments in both videos:
- same_person: tid=5 end ↔ tid=23 start (both climber, 13 sample gap)
- same_person: tid=23 end ↔ tid=43 start (climber top ↔ descent)
- different: tid=5 vs tid=0 (climber vs passerby)
- different: tid=5 vs tid=1 (climber vs big-bbox passerby)
- different: tid=10 vs tid=70 (climber vs belayer at bottom)
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


def get_track(tracks, tid):
    for t in tracks:
        if t["track_id"] == tid:
            return t
    raise KeyError(tid)


def ask_recall(frag_a: dict, frag_b: dict, scene_context: str) -> dict:
    """Ask Jev: are these two track fragments the same person?"""
    state = {
        "scene_context": scene_context,
        "video_resolution": "720x1280 (portrait)",
        "fragment_A_ended_earlier": frag_a,
        "fragment_B_started_later": frag_b,
    }
    payload = {
        "state": json.dumps(state, ensure_ascii=False),
        "model": "jev-latest",
        "questions": {
            "same_person": {
                "type": "noul",
                "instructions": (
                    "Are these two tracking fragments the SAME physical person whose ID was lost "
                    "and restarted by the tracker? Use temporal continuity (how close in time the "
                    "fragments are), spatial continuity (how close the last position of A is to the "
                    "first position of B), bbox size consistency, and role consistency "
                    "(both appear to be the same role: climber, belayer, passerby, etc). "
                    "Return the probability that they are the same person (0 = definitely "
                    "different, 1 = definitely same)."
                ),
            },
            "same_role": {
                "type": "choice",
                "instructions": "Do both fragments show the same role (climber/belayer/passerby/noise)?",
                "criteria": {
                    "yes_same_role": "Both fragments show behavior consistent with the same role.",
                    "no_different_roles": "The two fragments show behavior of different roles.",
                },
            },
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
    scene = (
        "Indoor climbing gym. One climber ascends and later descends a route tracked by "
        "a fixed camera. A belayer stands at the ground. Occasionally other people walk "
        "very close past the lens. ByteTrack frequently assigns new track_ids when a "
        "person's detection confidence drops momentarily (e.g., climber small at top of wall)."
    )

    tracks_a = summarize_tracks(raw_dir / "bytetrack_VID_20261003_200654.jsonl")
    tracks_b = summarize_tracks(raw_dir / "bytetrack_VID_20261003_203017.jsonl")

    # Compact per-track representation (drop fields that would mislead Jev into
    # treating absolute times as more important than they are)
    def strip(t):
        keep = ["track_id", "n_samples", "duration_seconds",
                "first_time_s", "last_time_s",
                "y_range_frac_of_frame", "x_range_frac_of_frame",
                "net_vertical_motion_frac_of_frame",
                "median_bbox_area_frac", "median_center_y_frac", "median_center_x_frac",
                "mean_confidence", "first_center", "last_center"]
        return {k: t[k] for k in keep}

    cases = [
        # (video_label, truth, frag_a, frag_b)
        ("A: tid=5 → tid=23", "same", strip(get_track(tracks_a, 5)), strip(get_track(tracks_a, 23))),
        ("A: tid=23 → tid=43", "same", strip(get_track(tracks_a, 23)), strip(get_track(tracks_a, 43))),
        ("A: tid=5 → tid=43 (big gap)", "same", strip(get_track(tracks_a, 5)), strip(get_track(tracks_a, 43))),
        ("A: tid=5 → tid=0 (climber vs close passerby)", "different",
             strip(get_track(tracks_a, 5)), strip(get_track(tracks_a, 0))),
        ("A: tid=5 → tid=1 (climber vs big-bbox)", "different",
             strip(get_track(tracks_a, 5)), strip(get_track(tracks_a, 1))),
        ("B: tid=10 → tid=53", "same", strip(get_track(tracks_b, 10)), strip(get_track(tracks_b, 53))),
        ("B: tid=53 → tid=63", "same", strip(get_track(tracks_b, 53)), strip(get_track(tracks_b, 63))),
        ("B: tid=10 → tid=70 (climber vs belayer)", "different",
             strip(get_track(tracks_b, 10)), strip(get_track(tracks_b, 70))),
        ("B: tid=10 → tid=2 (climber vs climber-start-or-belayer)", "ambiguous",
             strip(get_track(tracks_b, 10)), strip(get_track(tracks_b, 2))),
    ]

    results = []
    print(f"{'case':<60} {'truth':<10} {'noul':>6}  {'role':<20} {'conf':>5}")
    print("-" * 110)
    for label, truth, a, b in cases:
        try:
            resp = ask_recall(a, b, scene)
            noul = resp["answers"]["same_person"]["noul"]
            role = resp["answers"]["same_role"]
            print(f"{label:<60} {truth:<10} {noul:>6.2f}  {role['choice']:<20} {role['confidence']:.2f}")
            results.append({"case": label, "truth": truth, "noul_same_person": noul,
                            "role_choice": role["choice"], "role_conf": role["confidence"]})
            time.sleep(0.2)
        except Exception as e:
            print(f"{label}  ERROR {e}")
            results.append({"case": label, "truth": truth, "error": str(e)})

    out = Path(__file__).parent / "exp2_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Verdict
    print("\nVerdict:")
    for r in results:
        if "error" in r: continue
        want_same = r["truth"] == "same"
        is_predicted_same = r["noul_same_person"] >= 0.5
        if r["truth"] == "ambiguous":
            tag = "???"
        else:
            tag = "✓" if want_same == is_predicted_same else "✗"
        print(f"  {tag}  {r['case']:<55} truth={r['truth']}  noul={r['noul_same_person']:.2f}")


if __name__ == "__main__":
    main()
