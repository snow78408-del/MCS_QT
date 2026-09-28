"""Bounded camera-only preflight; keeps frames in memory and writes no video."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.device_lock import DeviceLock, camera_lock_key
from backend.runtime_paths import user_data_dir
from backend.vision.cameras.adapters.hikrobot_camera import HikrobotCameraAdapter
from backend.vision.parallel_walls import ParallelWallLocalizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--exposure-us", type=float, default=80.0)
    parser.add_argument("--snapshot", type=Path,
                        help="Save exactly one diagnostic still image; no video")
    parser.add_argument("--temporary-gain", type=float,
                        help="Try this gain during preflight, then restore the original value")
    parser.add_argument("--walls-proposal", type=Path,
                        help="Diagnostic measurement using visually reviewed walls, not automatic acceptance")
    args = parser.parse_args()
    if not 3 <= args.frames <= 100 or not 1 <= args.exposure_us <= 100000:
        parser.error("frames must be 3..100 and exposure 1..100000 us")
    payload = json.loads((user_data_dir() / "config/frontend_settings.json").read_text(encoding="utf-8"))
    settings = payload.get("values", payload.get("settings", payload))
    source = str(settings["video_source"])
    matches = [item for item in HikrobotCameraAdapter.discover_devices()
               if item.unique_id == source and item.available]
    if len(matches) != 1:
        raise RuntimeError("Configured camera unavailable or ambiguous")
    localizer = ParallelWallLocalizer(contrast_enhance=True)
    status_counts: dict[str, int] = {}
    last = None
    adapter = HikrobotCameraAdapter()
    original_gain = None
    detector = None
    walls = None
    measured_frames = 0
    complete_plugs = 0
    last_reference = None
    last_reference_source = None
    last_axes = None
    if args.walls_proposal:
        from backend.vision.config import DetectorConfig, DebugConfig
        from backend.vision.detector import DropletDetector
        from backend.vision.plug_geometry import rectified_axes
        from backend.vision.rectified_measurement import (
            detect_rectified_generation_plugs,
            trace_summary,
        )
        from backend.vision.rectified_roi import rectify_channel_frame
        walls = json.loads(args.walls_proposal.read_text(encoding="utf-8"))["walls"]
        tuning = json.loads((user_data_dir() / "config/vision_tuning_parameters.json").read_text(encoding="utf-8"))
        config = DetectorConfig(**tuning["detector"])
        config.measurement_mode = "generation_plug"
        config.generation_min_length_ratio = 0.5
        detector = DropletDetector(config, DebugConfig())
    with DeviceLock(camera_lock_key(source)):
        try:
            adapter.open(matches[0])
            if args.temporary_gain is not None:
                original_gain = float(adapter.get_feature("gain"))
                adapter.set_feature("gain", args.temporary_gain)
            adapter.set_feature("exposure", args.exposure_us)
            readback = float(adapter.get_feature("exposure"))
            if abs(readback - args.exposure_us) > max(1.0, args.exposure_us * 0.05):
                raise RuntimeError(f"Exposure readback {readback} us differs from request")
            adapter.start_stream()
            deadline = time.monotonic() + 8.0
            count = 0
            previous = None
            snapshot_saved = False
            intensity_ranges = []
            while count < args.frames and time.monotonic() < deadline:
                packet = adapter.read_frame(500)
                if not packet.valid or packet.image is None or packet.frame_id == previous:
                    continue
                previous = packet.frame_id
                intensity_ranges.append((int(packet.image.min()), int(packet.image.max())))
                if detector is not None:
                    import cv2
                    roi = rectify_channel_frame(packet.image, walls)
                    if roi is None:
                        raise RuntimeError("Reviewed walls cannot be rectified")
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                    # 像素域诊断：显式声明本帧观测截面；不产生 µm 结论、不构造测量链。
                    # 用**像素个数**而不是索引跨度：声明值要与 wall_separation_px
                    # （= 透视输出高度 = 横向像素个数）自洽，否则会被几何一致性闸门拒掉。
                    detector.declare_pixel_cross_section(
                        float(rectified_axes(gray.shape[:2]).transverse_pixel_count),
                        reason="diagnostic_pixel_domain")
                    trace: dict = {}
                    detect_rectified_generation_plugs(gray, detector=detector, trace=trace)
                    summary = trace_summary(trace)
                    selected = summary.get("selected_intervals", []) or []
                    measured_frames += bool(selected)
                    complete_plugs += len(selected)
                    last_reference = summary.get("reference_width_px")
                    last_reference_source = summary.get("reference_width_source")
                    last_axes = rectified_axes(gray.shape[:2]).to_dict()
                if args.snapshot is not None and not snapshot_saved:
                    import cv2
                    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(args.snapshot), packet.image):
                        raise RuntimeError("Could not save diagnostic still")
                    snapshot_saved = True
                stamp = time.monotonic()
                localizer.observe(packet.image, frame_id=int(packet.frame_id),
                                  capture_monotonic=stamp)
                last = localizer.localize(now_monotonic=stamp)
                status_counts[last.status] = status_counts.get(last.status, 0) + 1
                count += 1
            print(json.dumps({"frames": count, "exposure_readback_us": readback,
                              "last_image_shape": last.geometry.get("image_shape") if last else None,
                              "localization_status_counts": status_counts,
                              "last_reason": last.reason if last else "no_frames",
                              "last_walls": last.wall_lines if last and last.usable else [],
                              "manual_proposal_measured_frames": measured_frames,
                              "manual_proposal_plug_observations": complete_plugs,
                              "manual_proposal_geometry_validated": False,
                              "manual_proposal_reference_width_px": last_reference,
                              "manual_proposal_reference_width_source": last_reference_source,
                              "manual_proposal_axes": last_axes,
                              "measurement_chain": {"attempted": False,
                                                    "why": "预检只做像素域检测探针，不给测量结论"},
                              "intensity_range": [min(x[0] for x in intensity_ranges),
                                                  max(x[1] for x in intensity_ranges)] if intensity_ranges else None,
                              "video_saved": False}, ensure_ascii=False))
            return 0 if last and last.usable else 2
        finally:
            try:
                if adapter.is_streaming():
                    adapter.stop_stream()
            finally:
                try:
                    if original_gain is not None and adapter.is_open():
                        adapter.set_feature("gain", original_gain)
                finally:
                    if adapter.is_open():
                        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
