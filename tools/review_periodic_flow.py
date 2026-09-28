"""Reprocess a saved frame stack without hardware access or physical-flow assumptions."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.vision.flow_locking import duct_walls
from backend.vision.offline_analysis import periodic_phase_average, spatial_pitch, velocity_windows


def extract_profiles(stack: np.ndarray, walls, offset: float, half: int = 2) -> np.ndarray:
    """Sample the fitted wall coordinate system without extending past the image."""
    if not walls.ok:
        raise ValueError(f"wall fit failed: {walls.reason}")
    xs = np.arange(walls.x_lo, walls.x_hi)
    mid = walls.mid_row_at_x_lo + walls.mid_slope * (xs - walls.x_lo)
    centers = np.rint(mid + offset).astype(int)
    if np.any(centers - half < 0) or np.any(centers + half >= stack.shape[1]):
        raise ValueError("profile sampling would leave the image")
    return np.stack([stack[:, center - half:center + half + 1, x].mean(axis=1)
                     for x, center in zip(xs, centers)], axis=1)


def write_review_images(stack: np.ndarray, walls, directory: Path) -> list[dict]:
    """Provide raw and annotated crops for later independent boundary labels."""
    indices = sorted(set([0, len(stack) // 2, len(stack) - 1]))
    records = []
    xs = np.arange(walls.x_lo, walls.x_hi)
    mid = walls.mid_row_at_x_lo + walls.mid_slope * (xs - walls.x_lo)
    low = max(0, int(np.floor(mid.min() - walls.gap_px / 2 - 12)))
    high = min(stack.shape[1], int(np.ceil(mid.max() + walls.gap_px / 2 + 12)))
    for index in indices:
        raw = np.asarray(stack[index, low:high, walls.x_lo:walls.x_hi], np.uint8)
        preview = cv2.cvtColor(cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
        for distance, color in ((-walls.gap_px / 2, (0, 255, 0)), (walls.gap_px / 2, (0, 255, 0))):
            points = np.column_stack((xs - walls.x_lo, np.rint(mid + distance - low))).astype(np.int32)
            cv2.polylines(preview, [points], False, color, 1)
        preview = cv2.resize(preview, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        preview = cv2.copyMakeBorder(preview, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        for x in range(((walls.x_lo + 24) // 25) * 25, walls.x_hi, 25):
            pos = 2 * (x - walls.x_lo)
            cv2.line(preview, (pos, 22), (pos, 28), (255, 255, 255), 1)
            cv2.putText(preview, str(x), (pos - 12, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        raw_name, preview_name = f"frame_{index:04d}_raw.png", f"frame_{index:04d}_walls.png"
        for name, image in ((raw_name, raw), (preview_name, preview)):
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError(f"cannot encode {name}")
            (directory / name).write_bytes(encoded.tobytes())
        records.append({"frame": index, "crop_origin_xy": [walls.x_lo, low],
                        "raw": raw_name, "wall_overlay": preview_name,
                        "manual_meniscus_pairs_full_image_xy": [], "annotation_status": "pending"})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="saved grayscale .npy stack")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--time-source", required=True, help="provenance of rate, not a precision claim")
    parser.add_argument("--output", type=Path, required=True, help="a new directory")
    parser.add_argument("--direction", choices=["unknown", "left", "right"], default="unknown")
    parser.add_argument("--max-displacement", type=float)
    parser.add_argument("--bound-source", default="")
    parser.add_argument("--offset", type=float, default=13, help="sampling offset from wall midpoint in pixels")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--min-pitch", type=int, default=120)
    parser.add_argument("--max-pitch", type=int, default=230)
    args = parser.parse_args()
    if not np.isfinite(args.rate) or args.rate <= 0:
        parser.error("--rate must be positive and finite")
    if args.max_displacement is not None and (not np.isfinite(args.max_displacement)
                                             or args.max_displacement <= 0 or not args.bound_source.strip()):
        parser.error("--max-displacement requires a positive finite value and --bound-source")
    if not np.isfinite(args.offset) or not args.time_source.strip():
        parser.error("finite --offset and nonempty --time-source required")
    if args.output.exists():
        parser.error("output directory already exists; preserve previous results and choose a new one")
    stack = np.load(args.video, mmap_mode="r", allow_pickle=False)
    if stack.ndim != 3 or stack.shape[0] < 12 or stack.dtype != np.uint8:
        parser.error("expected at least 12 grayscale uint8 frames")
    walls = duct_walls(stack[:min(300, len(stack))])
    profiles = extract_profiles(stack, walls, args.offset)
    pitch, strength = spatial_pitch(profiles, args.min_pitch, args.max_pitch)
    phase = periodic_phase_average(profiles, pitch).to_dict() if pitch > 0 and strength >= 0.25 else None
    direction = {"unknown": None, "left": -1, "right": 1}[args.direction]
    windows = velocity_windows(profiles, 1 / args.rate, window=args.window, step=args.step,
                               minimum_pitch=args.min_pitch, maximum_pitch=args.max_pitch,
                               direction=direction, max_displacement=args.max_displacement,
                               bound_source=args.bound_source)
    digest = hashlib.sha256()
    with args.video.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    args.output.mkdir(parents=True, exist_ok=False)
    annotations = write_review_images(stack, walls, args.output)
    report = {"source": str(args.video), "sha256": digest.hexdigest(), "frames": len(stack),
              "rate_hz": args.rate, "time_source": args.time_source, "direction_prior": args.direction,
              "profile_offset_px": args.offset, "walls": asdict(walls), "pitch_px": pitch,
              "pitch_strength": strength, "phase_average": phase, "windows": windows,
              "status_counts": dict(Counter(row["status"] for row in windows)),
              "control_authorized": False, "annotations": annotations,
              "limitations": ["Intensity duty is not droplet volume fraction or phase identity.",
                              "Phase alignment recovers phase modulo pitch, not travel distance.",
                              "No physical scale, syringe calibration or true flow is inferred.",
                              "Window times assume the supplied rate; per-frame timing was not validated.",
                              "Green walls are algorithm overlays, not independent manual labels."]}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = ["# 离线重算结果", "", f"输入：{args.video.name}；{len(stack)} 帧。",
               f"管壁间距：{walls.gap_px:.3f} px；空间周期：{pitch:.3f} px。", "",
               f"速度窗口状态：{report['status_counts']}。", "",
               "未提供独立位移范围时，候选速度不代表真实速度；本报告不能授权控制。", "",
               "| 起止帧（右端不含） | 周期 px | 候选位移 px/帧 | 状态 |", "| --- | --- | --- | --- |"]
    for row in windows:
        candidate = row.get("candidate_px_per_frame")
        value = "—" if candidate is None else f"{candidate:.3f}"
        summary.append(f"| {row['start_frame']}–{row['stop_frame_exclusive']} | {row['pitch_px']:.3f} | {value} | {row['status']} |")
    if phase:
        summary += ["", f"相位平均保留 {len(phase['accepted_frames'])}/{len(stack)} 帧，"
                    f"强度峰峰值 {phase['contrast_gray']:.3f}；亮度占空比 {phase['intensity_duty']}。",
                    "亮度占空比不能当作液滴长度、体积占比或相别。"]
    summary += ["", "抽样帧的原始裁剪和管壁叠加图已输出，人工弯月面标注尚待填写。"]
    (args.output / "report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pitch_px": pitch, "status_counts": report["status_counts"],
                      "phase_contrast": None if phase is None else phase["contrast_gray"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
