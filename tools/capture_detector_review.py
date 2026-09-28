from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.runtime_paths import user_data_dir
from backend.vision.cameras.adapters.hikrobot_camera import HikrobotCameraAdapter
from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.parallel_walls import ParallelWallLocalizer
from backend.vision.plug_geometry import rectified_axes
from backend.vision.rectified_measurement import (
    FrameEvidence,
    ScaleEvidence,
    measure_generation_plugs,
)
from backend.vision.rectified_roi import rectify_channel_frame

CONTEXT_FRAMES = 24


def review_frame(raw, detector, localizer, *, frame_id: int, timestamp: float):
    """No saved ROI fallback: localization failure is not a zero-droplet result.

    测量走**生产入口** ``measure_generation_plugs``（不再直接调 detector 私有方法、
    也不再用 ``min(shape)`` 取参考宽度）。
    """
    localizer.observe(raw, frame_id=frame_id, capture_monotonic=timestamp)
    localization = localizer.localize(now_monotonic=timestamp)
    record = {"frame_id": frame_id, "timestamp": timestamp,
              "localization": localization.to_dict(), "status": "not_measured",
              "physical_scale_validated": False, "selected": None}
    if not localization.usable:
        record["reason"] = localization.reason
        return record, None, None
    probe = rectify_channel_frame(raw, localization.wall_lines)
    if probe is None:
        record["reason"] = "rectification_failed"
        return record, None, None
    # 像素域诊断：把名义通道**显式声明**为本帧观测到的像素截面，使门槛与画面自洽。
    # 这不是标尺（隐含 1 px = 1 µm），物理单位仍被 ScaleEvidence 与深度闸门挡住。
    # 用**像素个数**：声明值必须与 wall_separation_px（= 横向像素个数）自洽。
    axes_probe = rectified_axes(probe.shape[:2])
    detector.declare_pixel_cross_section(float(axes_probe.transverse_pixel_count),
                                        reason="diagnostic_pixel_domain")
    trace: dict = {}
    measurement = measure_generation_plugs(
        raw,
        detector=detector,
        localization=localization,
        scale=ScaleEvidence(um_per_px=None, source="configured_optical", validated=False,
                            detail="像素域诊断：未声明独立标尺"),
        frame_evidence=FrameEvidence(
            frame_id=frame_id, hardware_frame_id=frame_id, capture_monotonic=timestamp,
            localization_frame_id=frame_id, time_source="host_clock_proxy"),
        duct_depth_um=None, duct_depth_source="unknown", duct_depth_validated=False,
        trace=trace,
    )
    summary = measurement.detection_trace_summary()
    selected = summary.get("selected_intervals", [])
    source_image = measurement.rectified_preview
    if source_image is None:
        record["reason"] = measurement.reason or "rectification_failed"
        return record, None, None
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY) if source_image.ndim == 3 else source_image
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for left, right, _length in selected:
        cv2.rectangle(preview, (int(left), 1), (int(right), int(gray.shape[0]) - 2), (0, 255, 0), 1)
    record.update(status="measured_pixels", reason=measurement.reason,
                  valid=bool(measurement.valid), shape=list(gray.shape),
                  axes=measurement.axes.to_dict() if measurement.axes else None,
                  reference_width_px=summary.get("reference_width_px"),
                  reference_width_source=summary.get("reference_width_source"),
                  bodies=summary.get("body_intervals"), selected=selected,
                  raw_checks=summary.get("raw_outline_contrast_checks"))
    return record, gray, preview


def video_context(video: Path, facts: Path, samples: list[dict]):
    """Use recorded acquisition times, not artificial spacing of isolated images."""
    def target_index(sample: dict) -> int:
        return int(sample.get("software_frame_index", sample.get("frame_index")))

    wanted = {i for sample in samples for i in range(
        max(0, target_index(sample) - CONTEXT_FRAMES + 1), target_index(sample) + 1)}
    times = {}
    with facts.open(encoding="utf-8") as stream:
        for line in stream:
            fact = json.loads(line)
            index = fact["software_frame_index"]
            if index in wanted:
                times[index] = float(fact["host_monotonic_timestamp"]["value"])
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        for sample in samples:
            target = target_index(sample)
            start = max(0, target - CONTEXT_FRAMES + 1)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise RuntimeError("Video seek failed")
            context = []
            for index in range(start, target + 1):
                if round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                    raise RuntimeError("Video frame index mismatch")
                ok, frame = capture.read()
                if not ok or index not in times:
                    raise RuntimeError(f"Missing raw frame or timestamp: {index}")
                context.append((index, times[index], frame))
            yield context
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--facts", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((user_data_dir() / "config/frontend_settings.json").read_text(encoding="utf-8"))
    settings = settings.get("values", settings.get("settings", settings))
    tuning = json.loads((user_data_dir() / "config/vision_tuning_parameters.json").read_text(encoding="utf-8"))
    if args.capture:
        devices = HikrobotCameraAdapter.discover_devices()
        device = next(item for item in devices if item.unique_id == settings["video_source"])
        adapter = HikrobotCameraAdapter()
        try:
            adapter.open(device)
            adapter.start_stream()
            deadline = time.monotonic() + 8
            saved = 0
            previous = None
            captured = []
            while time.monotonic() < deadline and saved < 40:
                frame = adapter.read_frame(500)
                if frame.valid and frame.frame_id != previous:
                    previous = frame.frame_id
                    cv2.imwrite(str(args.output / f"raw_{saved:03d}.png"), frame.image)
                    captured.append({"frame_id": int(frame.frame_id), "timestamp": time.monotonic()})
                    saved += 1
            print(f"Captured {saved} frames")
        finally:
            try:
                adapter.stop_stream()
            finally:
                adapter.close()
        if saved == 0:
            raise RuntimeError("Camera returned no valid frames within 8 seconds")
        (args.output / "capture_times.json").write_text(json.dumps(captured), encoding="utf-8")
    config = DetectorConfig(**tuning["detector"])
    config.measurement_mode = "generation_plug"
    config.generation_min_length_ratio = 0.5
    detector = DropletDetector(config, DebugConfig())
    report = []
    paths = sorted(args.output.glob("raw_*.png"))
    localizer = ParallelWallLocalizer(contrast_enhance=True)
    contexts = None
    if args.video:
        samples = json.loads((args.output / "sample_index.json").read_text(encoding="utf-8"))
        if len(samples) != len(paths):
            raise RuntimeError("Sample index does not match raw images")
        contexts = iter(video_context(args.video, args.facts or args.video.parent / "frames.ndjson", samples))
    capture_times = (json.loads((args.output / "capture_times.json").read_text(encoding="utf-8"))
                     if args.capture else None)
    for ordinal, path in enumerate(paths):
        raw = cv2.imread(str(path))
        if raw is None:
            raise RuntimeError(f"Cannot read {path}")
        if contexts is not None:
            context = next(contexts)
            if raw.shape != context[-1][2].shape or not (raw == context[-1][2]).all():
                raise RuntimeError(f"Video/sample identity mismatch: {path}")
            localizer = ParallelWallLocalizer(contrast_enhance=True)
            for index, timestamp, preceding in context[:-1]:
                localizer.observe(preceding, frame_id=index, capture_monotonic=timestamp)
            frame_id, timestamp, _ = context[-1]
        elif capture_times is not None:
            frame_id = capture_times[ordinal]["frame_id"]
            timestamp = capture_times[ordinal]["timestamp"]
        else:
            raise RuntimeError("Isolated images have no temporal evidence; supply --video and frame facts")
        record, gray, preview = review_frame(raw, detector, localizer, frame_id=frame_id, timestamp=timestamp)
        record["frame"] = path.name
        if gray is not None:
            cv2.imwrite(str(args.output / f"localized_roi_{path.stem}.png"), gray)
            cv2.imwrite(str(args.output / f"localized_overlay_{path.stem}.png"), preview)
        report.append(record)
    (args.output / "localized_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps([{"frame": r["frame"], "status": r["status"], "reason": r["reason"]} for r in report]))


if __name__ == "__main__":
    main()
