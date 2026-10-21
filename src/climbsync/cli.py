"""climbsync CLI."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import typer

from .align import dtw_align, path_to_frame_schedule, smooth_schedule
from .detect import run_pose
from .homography import diagnostic_plot, estimate_homography
from .progress import (
    anchor_xy,
    progress_curves_shared,
    trim_climb_window,
    warp_points_b_to_a,
)
from .render import RenderConfig, render_overlay, render_side_by_side

app = typer.Typer(add_completion=False, help="Sync two climbing videos of the same route.")


def _plot_curves(p_a_full, p_b_full, anchor_a, anchor_b, t_a, t_b,
                 ka0, ka1, kb0, kb1, path, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    # Row 0: raw anchor y vs time (lower = higher on screen → inverted axis)
    axes[0].plot(t_a, anchor_a, color="green", label="A ankle/foot y")
    axes[0].plot(t_b, anchor_b, color="magenta", label="B ankle/foot y")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("anchor y (px, lower = higher)")
    axes[0].set_xlabel("time (s)")
    axes[0].legend(); axes[0].set_title("Raw anchor signal (per video time)")
    # Row 1: progress curves + climb window shading
    axes[1].plot(t_a, p_a_full, color="green", label="A")
    axes[1].plot(t_b, p_b_full, color="magenta", label="B")
    axes[1].axvspan(t_a[ka0], t_a[ka1], color="green", alpha=0.1, label="A climb window")
    axes[1].axvspan(t_b[kb0], t_b[kb1], color="magenta", alpha=0.1, label="B climb window")
    axes[1].set_ylabel("progress")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(); axes[1].set_title("Progress curves + detected climb windows")
    # Row 2: DTW path on cropped segments
    pa_idx = [i for i, _ in path]
    pb_idx = [j for _, j in path]
    axes[2].plot(pa_idx, pb_idx, color="black", lw=0.8)
    axes[2].set_xlabel("sample A (within climb window)")
    axes[2].set_ylabel("sample B (within climb window)")
    axes[2].set_title("DTW path")
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close(fig)


def _pose_cached(video_path, model_path, sample_fps, device, verbose, cache_dir, dump_dir=None):
    key_src = f"{video_path.resolve()}|{Path(model_path).name}|{sample_fps}|{device}"
    key = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    cache_file = cache_dir / f"pose_{Path(video_path).stem}_{key}.pkl"
    # If dump requested, always re-run (so dump files get produced)
    if cache_file.exists() and dump_dir is None:
        typer.echo(f"      [cache hit] {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    tr = run_pose(video_path, model_path, sample_fps=sample_fps,
                  device=device, verbose=verbose, dump_dir=dump_dir)
    with open(cache_file, "wb") as f:
        pickle.dump(tr, f)
    return tr


@app.command()
def sync(
    video_a: Path = typer.Argument(..., exists=True, help="Baseline / primary video (A)."),
    video_b: Path = typer.Argument(..., exists=True, help="Secondary video (B) warped onto A."),
    out_dir: Path = typer.Option(Path("./out"), "--out", "-o"),
    pose_model: Path = typer.Option(Path("yolov8m-pose.pt"), "--pose-model"),
    sample_fps: float = typer.Option(3.0, "--sample-fps"),
    out_fps: float = typer.Option(30.0, "--out-fps"),
    device: str = typer.Option("cpu", "--device"),
    mode: str = typer.Option("side_by_side", "--mode", help="side_by_side | overlay | both"),
    dtw_band: float = typer.Option(0.2, "--dtw-band"),
    skip_homography: bool = typer.Option(False, "--no-register"),
    skeleton: bool = typer.Option(False, "--skeleton/--no-skeleton",
                                   help="Overlay pose skeletons on both videos (debug)."),
    max_duration: float = typer.Option(0.0, "--max-duration",
                                        help="Cap output duration in seconds (0 = unlimited)."),
    pre_pad: float = typer.Option(3.0, "--pre-pad",
                                   help="Seconds to pad BEFORE detected climb start (both videos)."),
    post_pad: float = typer.Option(5.0, "--post-pad",
                                    help="Seconds to pad AFTER detected climb end (both videos)."),
    min_wait_gap: float = typer.Option(3.0, "--min-wait-gap",
                                        help="Minimum seconds between wait events (0 = no smoothing)."),
    dump_raw: bool = typer.Option(False, "--dump-raw",
                                   help="Dump raw YOLO + ByteTrack detections to out_dir/raw/. Forces pose re-run."),
    diag_lines: bool = typer.Option(False, "--diag-lines",
                                     help="Draw horizontal foot-height reference lines (A green, B magenta, B-warp-to-A dashed)."),
    diag_progress: bool = typer.Option(False, "--diag-progress",
                                        help="Overlay pA / pB values in the info bar."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = out_dir / "diagnostic"
    diag.mkdir(exist_ok=True)
    cache = Path("/tmp/climbsync_cache"); cache.mkdir(exist_ok=True)
    dump_dir = (out_dir / "raw") if dump_raw else None

    # 1. pose
    typer.echo(f"[1/5] Pose on A: {video_a.name}")
    track_a = _pose_cached(video_a, pose_model, sample_fps, device, verbose, cache, dump_dir)
    typer.echo(f"      {len(track_a.frame_idx)} samples, {track_a.has_person.sum()} with person")
    typer.echo(f"[1/5] Pose on B: {video_b.name}")
    track_b = _pose_cached(video_b, pose_model, sample_fps, device, verbose, cache, dump_dir)
    typer.echo(f"      {len(track_b.frame_idx)} samples, {track_b.has_person.sum()} with person")

    # 2. Homography B→A first (needed for shared-frame progress normalization)
    typer.echo("[2/5] Homography B→A (needed for progress normalization)")
    H = None
    if not skip_homography:
        hres = estimate_homography(track_a, track_b, verbose=verbose)
        typer.echo(f"      pairs tried={hres.num_pairs_tried}  inliers={hres.inliers}")
        if hres.H is not None:
            H = hres.H
            diagnostic_plot(track_a, track_b, hres, diag / "homography.png")
            typer.echo(f"      saved {diag / 'homography.png'}")
        else:
            typer.echo("      WARN: no reliable H; falling back to per-video normalization")
    else:
        typer.echo("      skipped (per-video normalization will be used)")

    # 3. Progress curves (shared normalization via H)
    typer.echo("[3/5] Progress curves")
    axy_a = anchor_xy(track_a.kpts, track_a.bbox)
    axy_b = anchor_xy(track_b.kpts, track_b.bbox)
    if H is not None:
        axy_b_in_a = warp_points_b_to_a(axy_b, H)
        p_a, p_b = progress_curves_shared(axy_a[:, 1], axy_b_in_a[:, 1], track_a.height)
    else:
        # fallback: per-video (previous behavior)
        p_a, _ = progress_curves_shared(axy_a[:, 1], axy_a[:, 1], track_a.height)
        p_b, _ = progress_curves_shared(axy_b[:, 1], axy_b[:, 1], track_b.height)

    ka0, ka1 = trim_climb_window(p_a)
    kb0, kb1 = trim_climb_window(p_b)
    # Apply pre-pad (seconds) by stepping back in samples
    if pre_pad > 0:
        dta = 1.0 / sample_fps
        pad_samples = int(round(pre_pad / dta))
        ka0 = max(0, ka0 - pad_samples)
        kb0 = max(0, kb0 - pad_samples)
    if post_pad > 0:
        dta = 1.0 / sample_fps
        pad_samples = int(round(post_pad / dta))
        ka1 = min(len(p_a) - 1, ka1 + pad_samples)
        kb1 = min(len(p_b) - 1, kb1 + pad_samples)
    typer.echo(f"      A climb window samples [{ka0}:{ka1}]  "
               f"t=[{track_a.t[ka0]:.1f},{track_a.t[ka1]:.1f}]s  "
               f"p=[{p_a[ka0]:.2f},{p_a[ka1]:.2f}]")
    typer.echo(f"      B climb window samples [{kb0}:{kb1}]  "
               f"t=[{track_b.t[kb0]:.1f},{track_b.t[kb1]:.1f}]s  "
               f"p=[{p_b[kb0]:.2f},{p_b[kb1]:.2f}]")

    # Monotone within window
    pa_c = np.maximum.accumulate(p_a[ka0:ka1 + 1].copy())
    pb_c = np.maximum.accumulate(p_b[kb0:kb1 + 1].copy())
    fa_idx = track_a.frame_idx[ka0:ka1 + 1]
    fb_idx = track_b.frame_idx[kb0:kb1 + 1]

    p_a_plot = p_a.copy(); p_a_plot[ka0:ka1 + 1] = pa_c
    p_b_plot = p_b.copy(); p_b_plot[kb0:kb1 + 1] = pb_c

    # 4. DTW
    typer.echo("[4/5] DTW alignment")
    path = dtw_align(pa_c, pb_c, band=dtw_band)
    typer.echo(f"      path length = {len(path)} samples")
    _plot_curves(p_a_plot, p_b_plot,
                 axy_a[:, 1],
                 (warp_points_b_to_a(axy_b, H)[:, 1] if H is not None else axy_b[:, 1]),
                 track_a.t, track_b.t,
                 ka0, ka1, kb0, kb1, path, out_dir / "progress_curves.png")

    # 5. Schedule → render
    typer.echo("[5/5] Rendering")
    schedule = path_to_frame_schedule(
        path, fa_idx, fb_idx, track_a.fps, track_b.fps, out_fps=out_fps
    )
    if min_wait_gap > 0:
        n_before = sum(1 for _, _, t in schedule if t)
        schedule = smooth_schedule(schedule, out_fps, min_gap_sec=min_wait_gap)
        n_after = sum(1 for _, _, t in schedule if t)
        typer.echo(f"      smoothed waits: {n_before} → {n_after} wait-frames "
                   f"(gap ≥ {min_wait_gap}s)")
    if max_duration > 0:
        schedule = schedule[: int(max_duration * out_fps)]
    typer.echo(f"      {len(schedule)} output frames  ≈ {len(schedule)/out_fps:.1f}s")

    # dump sync map
    with open(out_dir / "sync_map.csv", "w") as f:
        f.write("out_frame,frame_a,frame_b,tag\n")
        for i, (fa, fb, tag) in enumerate(schedule):
            f.write(f"{i},{fa},{fb},{tag}\n")

    cfg = RenderConfig(out_fps=out_fps, show_skeleton=skeleton,
                       draw_diag_lines=diag_lines, draw_progress_overlay=diag_progress)

    # Precompute B anchor warped to A for the side-by-side diagnostic
    axy_b_in_a = warp_points_b_to_a(axy_b, H) if H is not None else axy_b

    if mode in ("side_by_side", "both"):
        p = out_dir / "side_by_side.mp4"
        typer.echo(f"      → {p}")
        render_side_by_side(track_a, track_b, H, schedule, p, cfg,
                            anchor_xy_a=axy_a, anchor_xy_b=axy_b,
                            anchor_xy_b_in_a=axy_b_in_a,
                            p_a_curve=p_a, p_b_curve=p_b)
    if mode in ("overlay", "both"):
        if H is None:
            typer.echo("      skip overlay: no homography")
        else:
            p = out_dir / "overlay.mp4"
            typer.echo(f"      → {p}")
            render_overlay(track_a, track_b, H, schedule, p, cfg)

    # dump summary
    summary = {
        "video_a": str(video_a),
        "video_b": str(video_b),
        "fps_a": track_a.fps,
        "fps_b": track_b.fps,
        "sample_fps": sample_fps,
        "out_fps": out_fps,
        "path_len": len(path),
        "n_out_frames": len(schedule),
        "homography_used": H is not None,
        "trim_a_frames": [int(fa_idx[0]), int(fa_idx[-1])],
        "trim_b_frames": [int(fb_idx[0]), int(fb_idx[-1])],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo("Done.")


if __name__ == "__main__":
    app()
