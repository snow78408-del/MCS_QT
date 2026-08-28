from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.benchmark import evaluate_labeled_frames, load_labeled_frames
from backend.vision.config import DetectorConfig, default_config
from backend.vision.detector import DropletDetector
from backend.vision.tuning import TuningFrame


def _read_labeled_video_frames(video_path: Path, indices: set[int]) -> list[TuningFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    frames: list[TuningFrame] = []
    last_index = max(indices)
    try:
        for frame_index in range(last_index + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in indices:
                frames.append(TuningFrame(frame_index, frame))
    finally:
        capture.release()
    return frames


def _load_detector_config(path: Path | None) -> DetectorConfig:
    if path is None:
        return default_config().detector
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("detector"), dict):
        payload = payload["detector"]
    if not isinstance(payload, dict):
        raise ValueError("检测配置必须是 JSON 对象")
    allowed = {field.name for field in fields(DetectorConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"检测配置包含未知字段：{', '.join(sorted(unknown))}")
    return DetectorConfig(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="评测液滴检测的准确率、半径误差和耗时")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    labels = load_labeled_frames(args.annotations)
    frames = _read_labeled_video_frames(args.video, {item.index for item in labels})
    detector_config = _load_detector_config(args.config)
    pipeline_config = default_config()
    benchmark = evaluate_labeled_frames(
        DropletDetector(detector_config, pipeline_config.debug),
        frames,
        labels,
    )
    output = asdict(benchmark)
    output["meets_targets"] = benchmark.meets_targets()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
