from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pid_control import PIDConfig, PIDInput, reset_controller, run_feedback_step
from backend.vision.config import default_config
from backend.vision.pipeline import VisionPipeline


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _fmt(value, digits: int = 3) -> str:
    if value is None or not _finite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline droplet recognition and PID simulation on a video.")
    parser.add_argument("video", help="Path to AVI/MP4 video")
    parser.add_argument("--target-um", type=float, default=60.0)
    parser.add_argument("--pixel-to-micron", type=float, default=0.24)
    parser.add_argument("--initial-q1", type=float, default=50.0)
    parser.add_argument("--initial-q2", type=float, default=20.0)
    parser.add_argument("--control-interval-ms", type=float, default=500.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--out-dir", default="simulation_outputs")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"), default=None)
    parser.add_argument("--min-radius", type=float, default=None)
    parser.add_argument("--max-radius", type=float, default=None)
    parser.add_argument("--min-center-distance", type=float, default=None)
    parser.add_argument("--edge-gradient-threshold", type=int, default=None)
    parser.add_argument("--edge-anchor-threshold", type=int, default=None)
    parser.add_argument("--edge-min-circle-ratio", type=float, default=None)
    parser.add_argument("--edge-min-support-ratio", type=float, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = (frame_count / fps) if fps > 0 else 0.0

    cfg = default_config()
    cfg.debug.draw_overlay = True
    if args.min_radius is not None:
        cfg.detector.min_radius = float(args.min_radius)
    if args.max_radius is not None:
        cfg.detector.max_radius = float(args.max_radius)
    if args.min_center_distance is not None:
        cfg.detector.min_center_distance = float(args.min_center_distance)
    if args.edge_gradient_threshold is not None:
        cfg.detector.edge_gradient_threshold = int(args.edge_gradient_threshold)
    if args.edge_anchor_threshold is not None:
        cfg.detector.edge_anchor_threshold = int(args.edge_anchor_threshold)
    if args.edge_min_circle_ratio is not None:
        cfg.detector.edge_min_circle_ratio = float(args.edge_min_circle_ratio)
    if args.edge_min_support_ratio is not None:
        cfg.detector.edge_min_support_ratio = float(args.edge_min_support_ratio)
    if args.roi is not None:
        x0, x1, y0, y1 = args.roi
        cfg.roi.enabled = True
        cfg.roi.x_start_ratio = float(x0)
        cfg.roi.x_end_ratio = float(x1)
        cfg.roi.y_start_ratio = float(y0)
        cfg.roi.y_end_ratio = float(y1)
    pipeline = VisionPipeline(cfg)
    pipeline.configure_expected_diameter(
        float(args.target_um),
        float(args.pixel_to_micron),
    )
    reset_controller()
    PIDConfig()

    rows: list[dict[str, object]] = []
    valid_diameters: list[float] = []
    commands = []
    freeze_count = 0
    stop_count = 0
    detected_frames = 0
    valid_frames = 0
    preview_saved = 0
    q1 = float(args.initial_q1)
    q2 = float(args.initial_q2)
    dt = max(1e-3, float(args.control_interval_ms) / 1000.0)

    frame_no = 0
    processed = 0
    started_at = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if args.sample_every > 1 and (frame_no - 1) % int(args.sample_every) != 0:
            continue
        if args.max_frames > 0 and processed >= int(args.max_frames):
            break
        processed += 1

        source_timestamp = (
            float(frame_no - 1) / fps
            if fps > 0.0
            else float(processed - 1) * max(1e-3, float(args.control_interval_ms) / 1000.0)
        )
        result = pipeline.process_frame(frame, timestamp=source_timestamp)
        control = result.metrics.control
        frame_avg_px = control.frame_avg_diameter
        frame_avg_um = None if frame_avg_px is None else float(frame_avg_px) * float(args.pixel_to_micron)
        droplet_count = int(control.frame_droplet_count)
        valid = bool(control.valid_for_control and droplet_count > 0 and frame_avg_um is not None)
        if droplet_count > 0:
            detected_frames += 1
        if valid:
            valid_frames += 1
            valid_diameters.append(float(frame_avg_um))

        pid_input = PIDInput(
            target_diameter_um=float(args.target_um),
            current_diameter_um=frame_avg_um,
            current_q1=q1,
            current_q2=q2,
            dt=dt,
            frame_id=int(result.frame_index),
            vision_valid=valid,
            pump_communication_ok=True,
            droplet_count=droplet_count,
            measurement_noise_est=float(control.frame_diameter_cv or 0.0),
        )
        cmd = run_feedback_step(pid_input)
        if cmd.freeze_feedback:
            freeze_count += 1
        if cmd.suggested_stop:
            stop_count += 1
        if not cmd.freeze_feedback and not cmd.suggested_stop:
            q1 = float(cmd.q1)
            q2 = float(cmd.q2)
        commands.append(cmd)

        if preview_saved < int(args.preview_count) and (droplet_count > 0 or valid):
            preview_path = out_dir / f"preview_frame_{frame_no:05d}.jpg"
            cv2.imwrite(str(preview_path), result.annotated_frame)
            preview_saved += 1

        rows.append(
            {
                "source_frame": frame_no,
                "pipeline_frame": result.frame_index,
                "raw_detections": len(result.detections.centers),
                "droplets": droplet_count,
                "valid_for_control": valid,
                "avg_diameter_px": frame_avg_px,
                "avg_diameter_um": frame_avg_um,
                "single_cell_rate": control.frame_single_cell_rate,
                "diameter_cv": control.frame_diameter_cv,
                "pid_freeze": cmd.freeze_feedback,
                "pid_stop": cmd.suggested_stop,
                "pid_reason": cmd.reason,
                "pid_error_um": cmd.diameter_error,
                "pid_adjustment": cmd.adjustment,
                "q1_command": cmd.q1,
                "q2_command": cmd.q2,
            }
        )

    cap.release()
    elapsed_s = max(1e-9, time.perf_counter() - started_at)

    csv_path = out_dir / "video_pid_simulation.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    avg_um = sum(valid_diameters) / len(valid_diameters) if valid_diameters else None
    std_um = None
    cv_um = None
    if len(valid_diameters) >= 2 and avg_um:
        var = sum((x - avg_um) ** 2 for x in valid_diameters) / (len(valid_diameters) - 1)
        std_um = math.sqrt(var)
        cv_um = std_um / avg_um * 100.0 if avg_um > 0 else None

    non_frozen = [cmd for cmd in commands if not cmd.freeze_feedback and not cmd.suggested_stop]
    print("VIDEO")
    print(f"path={video_path}")
    print(f"frames={frame_count} fps={_fmt(fps, 3)} duration_s={_fmt(duration_s, 3)} size={width}x{height}")
    print("RECOGNITION")
    print(f"processed_frames={processed}")
    print(f"processing_elapsed_s={_fmt(elapsed_s, 3)} processing_fps={_fmt(processed / elapsed_s if processed else 0, 2)}")
    print(f"detected_frames={detected_frames} ({_fmt(detected_frames / processed * 100 if processed else 0, 2)}%)")
    print(f"valid_control_frames={valid_frames} ({_fmt(valid_frames / processed * 100 if processed else 0, 2)}%)")
    print(f"avg_diameter_um={_fmt(avg_um)} std_um={_fmt(std_um)} cv_percent={_fmt(cv_um)}")
    print("PID")
    print(f"target_um={_fmt(args.target_um)} initial_q1={_fmt(args.initial_q1)} initial_q2={_fmt(args.initial_q2)} dt_s={_fmt(dt)}")
    print(f"updates={len(non_frozen)} frozen={freeze_count} suggested_stop={stop_count}")
    print(f"final_q1={_fmt(q1, 6)} final_q2={_fmt(q2, 6)}")
    if commands:
        last = commands[-1]
        print(
            "last_command="
            f"freeze={last.freeze_feedback} stop={last.suggested_stop} "
            f"err={_fmt(last.diameter_error)} adj={_fmt(last.adjustment, 6)} "
            f"q1={_fmt(last.q1, 6)} q2={_fmt(last.q2, 6)} reason={last.reason}"
        )
    print("OUTPUT")
    print(f"csv={csv_path}")
    print(f"preview_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
