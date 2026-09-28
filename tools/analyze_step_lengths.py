"""Make an exploratory pixel-length curve from a completed step experiment.

This is offline only. It never connects to the camera or pump, and it accepts
    only body-between-carrier detector branches whose endpoints can be checked on overlays.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.flow_locking import duct_walls
from backend.vision.rectified_measurement import detect_rectified_generation_plugs
from backend.vision.rectified_roi import rectify_channel_frame


def frame_times(path: Path) -> tuple[np.ndarray, list[int]]:
    stamps, identifiers = [], []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            fact = json.loads(line)
            if fact["software_frame_index"] != index:
                raise RuntimeError(f"frame fact {index} is out of sequence")
            stamps.append(float(fact["host_monotonic_timestamp"]["value"]))
            identifiers.append(int(fact["hardware_frame_id"]["value"]))
    values = np.asarray(stamps)
    if not len(values) or np.any(np.diff(values) <= 0):
        raise RuntimeError("frame timestamps are missing or nonmonotonic")
    return values, identifiers


def stage_walls(video: Path, stamps: np.ndarray, segments: list[dict]) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("cannot open raw video")
    results = []
    try:
        for stage, segment in enumerate(segments, 1):
            candidates = []
            for start_s in (45, 75, 120, 150, 180, 240):
                indices = [int(np.searchsorted(stamps,
                           segment["observation_started_monotonic"] + start_s + 3 * i))
                           for i in range(10)]
                images = []
                for index in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f"cannot read wall reference frame {index}")
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                fit = duct_walls(np.stack(images), top=225, bot=280, reach=12,
                                 min_gap=12, x_lo=120, x_hi=720)
                if fit.ok and fit.gap_sd_px <= 1.5:
                    candidates.append((fit.gap_sd_px, indices, fit))
            if not candidates:
                raise RuntimeError(f"stage {stage} has no stable wall reference window")
            _, indices, fit = min(candidates, key=lambda item: item[0])
            lines = []
            for sign in (-1, 1):
                y0 = fit.mid_row_at_x_lo + sign * fit.gap_px / 2
                y1 = y0 + fit.mid_slope * (fit.x_hi - fit.x_lo)
                lines.append({"x1": fit.x_lo / 720, "y1": y0 / 540,
                              "x2": fit.x_hi / 720, "y2": y1 / 540})
            results.append({"stage": stage, "source_frame_indices": indices,
                            "fit": asdict(fit), "walls": lines})
    finally:
        cap.release()
    return results


def draw_review(image: np.ndarray, intervals: list[tuple[int, int, float]]) -> np.ndarray:
    marked = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for left, right, _ in intervals:
        cv2.line(marked, (int(left), 0), (int(left), image.shape[0] - 1),
                 (0, 0, 255), 1)
        cv2.line(marked, (int(right), 0), (int(right), image.shape[0] - 1),
                 (0, 255, 0), 1)
    return cv2.resize(marked, None, fx=2, fy=5, interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    out = args.session_dir.resolve()
    session = json.loads((out / "session_summary.json").read_text(encoding="utf-8"))
    segments = session["segments"]
    if not segments or not session["result_layers"]["run_completed"]:
        raise RuntimeError("physical campaign did not complete")
    if not session["stop"]["verified"]:
        raise RuntimeError("pump stop was not verified")
    stamps, hardware_ids = frame_times(out / "frames.ndjson")
    if len(stamps) != session["frame_facts"]["raw_frames"]:
        raise RuntimeError("frame facts and video frame count differ")
    video = out / "raw_frames.mkv"
    geometries = stage_walls(video, stamps, segments)
    (out / "analysis_geometry.json").write_text(
        json.dumps(geometries, ensure_ascii=False, indent=2), encoding="utf-8")
    detectors = [DropletDetector(DetectorConfig(measurement_mode="generation_plug"),
                                 DebugConfig()) for _ in segments]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("cannot open video for analysis")
    samples = []
    reviews: dict[tuple[int, int], np.ndarray] = {}
    next_at = [float(s["observation_started_monotonic"]) for s in segments]
    targets = (30, 120, 240)
    try:
        for index, stamp in enumerate(stamps):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"video stopped before frame fact {index}")
            stage = next((i for i, seg in enumerate(segments)
                          if seg["observation_started_monotonic"] <= stamp
                          < seg["observation_finished_monotonic"]), None)
            if stage is None or stamp < next_at[stage]:
                continue
            next_at[stage] = float(stamp + 0.5)
            rectified = rectify_channel_frame(image, geometries[stage]["walls"])
            if rectified is None:
                raise RuntimeError("rectification failed")
            gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
            result, trace = detect_rectified_generation_plugs(
                gray, detector=detectors[stage])
            intervals = ([(int(left), int(right), float(length))
                          for (left, right, length), valid in zip(
                              trace["selected_intervals"], result.diameter_valid) if valid]
                         if trace["interval_source"] in {
                             "dark_body_between_carrier_gaps",
                             "transverse_body_between_carrier_gaps",
                         }
                         else [])
            relative = float(stamp - segments[stage]["observation_started_monotonic"])
            samples.append({"stage": stage + 1, "elapsed_s": relative,
                            "capture_monotonic": float(stamp),
                            "source_frame": index, "hardware_frame_id": hardware_ids[index],
                            "interval_source": trace["interval_source"],
                            "plug_count": len(intervals),
                            "median_length_px": float(np.median([v[2] for v in intervals]))
                            if intervals else None})
            for target in targets:
                key = (stage, target)
                if key not in reviews and relative >= target:
                    reviews[key] = draw_review(gray, intervals)
            if index and index % 10000 == 0:
                print(f"decoded {index}/{len(stamps)}, measured {len(samples)}", flush=True)
    finally:
        cap.release()
    if not samples:
        raise RuntimeError("no samples in observation periods")
    for (stage, target), image in reviews.items():
        cv2.imwrite(str(out / f"review_stage{stage+1}_{target}s.png"), image)
    with (out / "length_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    fig, axes = plt.subplots(len(segments), 1, figsize=(12, 3.3 * len(segments)),
                             squeeze=False, sharex=True)
    summaries = []
    for stage, segment in enumerate(segments, 1):
        rows = [r for r in samples if r["stage"] == stage]
        valid = [r for r in rows if r["median_length_px"] is not None]
        x = np.asarray([r["elapsed_s"] for r in valid])
        y = np.asarray([r["median_length_px"] for r in valid])
        ax = axes[stage - 1, 0]
        ax.scatter(x, y, s=9, alpha=0.4, color="#1670a9")
        if len(x):
            trend = [float(np.median(y[np.abs(x - t) <= 7.5])) for t in x]
            ax.plot(x, trend, color="#d85b24", linewidth=1.5)
        ax.set_xlim(0, 300)
        ax.set_ylabel("Plug length (px)")
        ax.set_title(f'{int(segment["q1_ul_min"])}/{int(segment["q2_ul_min"])} '
                     f'uL/min: {len(valid)}/{len(rows)} valid samples')
        ax.grid(alpha=0.2)
        tail = [r["median_length_px"] for r in valid if r["elapsed_s"] >= 240]
        summaries.append({"label": segment["label"],
                          "duration_s": round(segment["observation_finished_monotonic"]
                                              - segment["observation_started_monotonic"], 3),
                          "samples": len(rows), "valid_samples": len(valid),
                          "coverage": round(len(valid) / len(rows), 3) if rows else 0,
                          "median_px": round(float(np.median(y)), 2) if len(y) else None,
                          "last_60s_median_px": round(float(np.median(tail)), 2)
                          if tail else None})
    axes[-1, 0].set_xlabel("Time since stage start (s)")
    fig.suptitle("Dark plug length over time (exploratory pixel measurement)")
    fig.tight_layout()
    fig.savefig(out / "length_vs_time_corrected.png", dpi=160)
    plt.close(fig)
    report = {"source": str(out), "length_unit": "rectified image pixels",
              "physical_scale_validated": False,
              "sampling_premise_ok": session["result_layers"]["sampling_premise_ok"],
              "stages": summaries}
    (out / "length_analysis_summary_corrected.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
