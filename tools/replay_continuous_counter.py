"""Offline gate benchmark. Never opens cameras or sends pump commands."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.vision.line_counter import ContinuousLineCounter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--walls", nargs=8, type=float, required=True,
                        metavar="RATIO", help="two walls: x1 y1 x2 y2, normalized 0..1")
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.video.is_file() or args.period <= 0 or not all(0 <= v <= 1 for v in args.walls):
        parser.error("existing local video, positive period and normalized walls required")
    walls = [dict(zip(("x1", "y1", "x2", "y2"), args.walls[i:i+4])) for i in (0, 4)]
    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        parser.error("video cannot be decoded or has no valid FPS")
    counter = ContinuousLineCounter()
    elapsed = []
    windows = []
    end = args.period
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id = len(elapsed)
            timestamp = frame_id / fps
            started = time.perf_counter()
            counter.observe_frame(frame, frame_id, timestamp, walls)
            elapsed.append((time.perf_counter()-started)*1000)
            while timestamp >= end:
                windows.append(dict(start=end-args.period, end=end,
                                    **asdict(counter.window(end-args.period, end))))
                end += args.period
    finally:
        cap.release()
    report = dict(frames=len(elapsed), file_fps=fps, period_s=args.period,
                  timing_scope="gate only; excludes decode, size analysis, UI and scheduling",
                  mean_ms=float(np.mean(elapsed)) if elapsed else None,
                  p95_ms=float(np.percentile(elapsed, 95)) if elapsed else None,
                  max_ms=max(elapsed, default=None), windows=windows,
                  passage_frames=[round(t*fps) for t in counter.passage_times()],
                  note="Passage counts are not independent ground truth; first 32 frames calibrate the gate.")
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
