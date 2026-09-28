"""Single-still wall proposal and rectification, explicitly not an acceptance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.vision.parallel_walls import prepare_frame, pair_candidates
from backend.vision.plug_geometry import rectified_axes
from backend.vision.rectified_measurement import (
    detect_rectified_generation_plugs,
    trace_summary,
)
from backend.vision.rectified_roi import rectify_channel_frame
from backend.vision.config import DetectorConfig, DebugConfig
from backend.vision.detector import DropletDetector
from backend.runtime_paths import user_data_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--walls-proposal", type=Path,
                        help="Recheck a visually selected same-session proposal; never marks it validated")
    parser.add_argument("--end-x-px", type=int,
                        help="Visually reviewed end of the straight channel; proposal only")
    args = parser.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise RuntimeError("Cannot read diagnostic image")
    prepared = prepare_frame(frame, contrast_enhance=True)
    pairs = sorted((pair for pair in pair_candidates(prepared, frame.shape[:2])
                    if not pair.rejection), key=lambda pair: pair.score, reverse=True)
    if len(pairs) != 1 and args.walls_proposal is None:
        raise RuntimeError(f"Need visual selection: {len(pairs)} candidate pairs")
    h, w = frame.shape[:2]
    walls = (json.loads(args.walls_proposal.read_text(encoding="utf-8"))["walls"]
             if args.walls_proposal else pairs[0].wall_lines(w, h))
    if args.end_x_px is not None:
        if not 0 < args.end_x_px < w:
            raise ValueError("Channel endpoint must be inside the image")
        extended = []
        for line in walls:
            x1, x2 = float(line["x1"]), float(line["x2"])
            if x2 <= x1 or args.end_x_px / w <= x1:
                raise ValueError("Invalid wall segment")
            end_x = args.end_x_px / w
            end_y = float(line["y1"]) + (float(line["y2"]) - float(line["y1"])) * (end_x - x1) / (x2 - x1)
            if not 0 < end_y < 1:
                raise ValueError("Extended wall leaves the image")
            extended.append(dict(line, x2=end_x, y2=end_y))
        walls = extended
    roi = rectify_channel_frame(frame, walls)
    if roi is None:
        raise RuntimeError("Rectification failed")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tuning = json.loads((user_data_dir() / "config/vision_tuning_parameters.json").read_text(encoding="utf-8"))
    config = DetectorConfig(**tuning["detector"])
    config.measurement_mode = "generation_plug"
    config.generation_min_length_ratio = 0.5
    detector = DropletDetector(config, DebugConfig())
    # 像素域诊断：显式声明本帧观测截面，使门槛与画面自洽；不产生 µm 结论。
    # 用**像素个数**（= wall_separation_px），不是索引跨度。
    axes = rectified_axes(gray.shape[:2])
    detector.declare_pixel_cross_section(float(axes.transverse_pixel_count),
                                        reason="diagnostic_pixel_domain")
    # 走生产路径**同一个检测内核**：候选与门槛因此与线上一致。
    trace: dict = {}
    detect_rectified_generation_plugs(gray, detector=detector, trace=trace)
    summary = trace_summary(trace)
    preview = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    for line in walls:
        cv2.line(preview, (round(line["x1"] * w), round(line["y1"] * h)),
                 (round(line["x2"] * w), round(line["y2"] * h)), (0, 255, 0), 1)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, image in (("wall_proposal.png", preview), ("rectified_raw.png", roi),
                        ("rectified_display.png", cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX))):
        if not cv2.imwrite(str(args.output / name), image):
            raise RuntimeError(f"Could not save {name}")
    report = {"status": "single_frame_proposal_not_validated", "source": str(args.image),
              "walls": walls, "rectified_shape": list(gray.shape),
              "axes": axes.to_dict(),
              "reference_width_px": summary.get("reference_width_px"),
              "reference_width_source": summary.get("reference_width_source"),
              "selected_intervals": summary.get("selected_intervals"),
              "raw_checks": summary.get("raw_outline_contrast_checks"),
              "physical_scale_validated": False,
              # 单张静帧没有帧身份与标尺，因此**没有**测量链结论：这些只是检测探针结果。
              "measurement_chain": {"attempted": False,
                                    "why": "孤立静帧没有帧身份与标尺，不给测量结论"}}
    report["visually_reviewed_end_x_px"] = args.end_x_px
    (args.output / "proposal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
