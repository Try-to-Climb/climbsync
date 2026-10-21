# climbsync

**Synchronize two climbing videos of the same route** into one side-by-side comparison. Given two takes with different start times, different climbing rhythms, and different camera angles, climbsync produces one output where **the two climbers appear at the same physical height on the wall at every moment**. When one climber is behind, the other side pauses to wait.

```
Video A ─┐
          ├── pose track → progress curve (H-normalized) ──┐
Video B ─┘                                                 ├── DTW alignment → side-by-side mp4
          ├─────────── Homography (B → A) ─────────────────┘
```

## How it works

1. **Detect the climber** with YOLOv8-pose + ByteTrack, then stitch ID fragments by spatial continuity. Filters out passers-by (bbox too close to camera) and belayers (negligible vertical motion).
2. **Extract an anchor point** per frame — the lower foot's ankle (falls back to hip-based estimate if ankle is unreliable, including a sanity check against inverted pose keypoints).
3. **Estimate a homography** `H: B → A` from SIFT matches on the wall (climber bbox masked out) — this lets us compare the two climbers in a single coordinate system.
4. **Build a shared progress curve** `p ∈ [0, 1]` by warping B's anchor into A's frame and normalising both against A's climb span. `p_A = p_B` now means both climbers are at the same physical row on the wall.
5. **DTW-align** the two curves. The resulting path tells us, for every output frame, which frames of A and B to show, and when one side should wait.
6. **Render** a side-by-side mp4 with optional skeletons, wait indicators, and diagnostic overlays.

## Install

Requires Python 3.10+. One-time setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first run will auto-download the YOLOv8 pose model into the working directory.

## Usage

Minimum:

```bash
climbsync A.mp4 B.mp4
```

That's it. Output lands in `./out/`. Good defaults are baked in:
- pose model: `yolov8m-pose.pt` (auto-downloaded on first run)
- sample fps: 3
- climb-window padding: 3s before, 5s after
- min-wait-gap: 3s (coalesces micro-waits into one wait per 3s block)

Common overrides:

```bash
climbsync A.mp4 B.mp4 -o my_out/ --skeleton      # draw pose skeletons for debug
climbsync A.mp4 B.mp4 --device cuda              # GPU inference
climbsync A.mp4 B.mp4 --mode both                # also produce overlay.mp4
climbsync A.mp4 B.mp4 --diag-lines --diag-progress  # overlay anchor lines + pA/pB
```

Produces:

```
out/
├── side_by_side.mp4         # main output
├── progress_curves.png      # three-panel diagnostic: raw anchor y / progress curves + climb windows / DTW path
├── sync_map.csv             # (out_frame, frame_a, frame_b, wait_tag) per output frame
├── summary.json             # metadata
└── diagnostic/
    └── homography.png       # A, B, and A⊕warp(B) side-by-side (verify the wall registration)
```

### Key flags

| Flag | Purpose | Default |
|---|---|---|
| `-o / --out` | Output directory | `./out` |
| `--pose-model` | YOLOv8 pose weights (n / s / m / l / x) | `yolov8s-pose.pt` |
| `--sample-fps` | Pose inference sample rate (lower = faster) | `5.0` |
| `--out-fps` | Output video fps | `30.0` |
| `--mode` | `side_by_side` / `overlay` / `both` | `side_by_side` |
| `--pre-pad` / `--post-pad` | Extend climb window by N seconds before start / after end | `0` |
| `--min-wait-gap` | Enforce minimum seconds between wait events (smooths UX) | `0` |
| `--diag-lines` | Draw horizontal foot-height reference lines (A green, B magenta, B→A dashed) | off |
| `--diag-progress` | Overlay `pA / pB / dp` in the info bar | off |
| `--no-skeleton` | Hide skeleton overlays | skeleton on |
| `--dump-raw` | Dump raw YOLO + ByteTrack jsonl under `out/raw/` | off |
| `--device` | `cuda` / `cpu` / `auto` | `cpu` |

### Diagnostic lines

With `--diag-lines`, the side-by-side video shows:
- **Solid green line** on A side: A climber's anchor foot y.
- **Solid magenta line** on B side: B climber's anchor foot y.
- **Dashed magenta line** on A side: B's foot warped into A's coordinate frame. When this dashed line lines up with the solid green line, the two climbers are at the same physical height.

The info bar shows `pA / pB / dp` (progress and difference). With sync working well, `dp ≈ 0` outside of wait regions.

## Design notes & gotchas

- **The one‑dimensional progress signal.** A climber's state is really 2D (hand + foot heights), but climbsync projects it to a single number (lower foot y). This is enough for most alignment but creates a lag during "hand moves first, foot catches up" sequences. In practice `--min-wait-gap 3` hides most of the resulting micro-waits.
- **Wall-plane assumption.** A single homography is only exact for a flat wall. Pronounced overhangs or big volumes introduce millimetre-scale residuals at the extremes. The diagnostic image `diagnostic/homography.png` shows whether the warp fits your route.
- **ByteTrack ID drift.** Pose detection on climbers in extreme stretches is noisy; ByteTrack frequently reassigns IDs. `_pick_climber_tracks` stitches candidate tracks by temporal + spatial continuity. Edge cases where this doesn't work are visible in the `all candidates (sorted by y_range)` log line.
- **Pose keypoint flipping.** YOLOv8-pose sometimes outputs an "ankle" above the hip in extreme climbing poses. A sanity check in `anchor_xy` rejects ankle keypoints that fall above the hip and falls back to hip-based geometry.

## Repo layout

```
src/climbsync/
├── detect.py        # YOLO + ByteTrack, climber selection, track stitching
├── progress.py      # anchor_xy, Hampel/median/SG cleaning, shared normalization
├── homography.py    # SIFT + RANSAC homography B → A + diagnostic
├── align.py         # DTW with Sakoe-Chiba band + wait-gap smoothing
├── render.py        # side-by-side & overlay rendering (ffmpeg H.264)
└── cli.py           # Typer CLI

experiments/jev/     # Early exploration of using TypeSafe Jev for role classification / ID recall
scripts/verify_bytetrack.py   # Verify that ByteTrack output is a deterministic function of YOLO output
docs/                # Development notes
```

## License

MIT. See `LICENSE`.
