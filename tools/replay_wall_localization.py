"""按**真实采集时间**回放已有录像，通过生产定位/测量路径（离线，不碰硬件）。

用途：任务书 §3 —— 用已有的连续录像检查修复后的生产路径在真实画面上做到哪一步。

要点：
* 片段由清单**预先登记**（``--manifest``），本工具不改动取样规则。
* 片段边界按 ``frames.ndjson`` 里的**真实采集时间**（``host_monotonic_timestamp``）
  解析，不用帧序号 ÷ 标称帧率。
* 逐帧**连续**送入，**同一** ``ParallelWallLocalizer`` 实例（其历史是
  ``deque(maxlen=max_buffer_frames)`` 有界缓冲），不对每帧重建定位器。
* 暖机前缀是片段**之前**的帧（同一录像、更早的时刻），在结果里逐段列出；
  绝不使用片段之后的帧。
* 测量经生产入口 ``measure_generation_plugs``；标尺未声明，因此物理单位一律不产出。
* 叠加区域用**正确的逆透视**映回原图，并与出具几何的帧号绑定。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.provenance import (                                        # noqa: E402
    EffectiveConfig,
    IncrementalIndexWriter,
    ProvenanceError,
    build_bundle,
    declared,
    frame_reference,
    read_committed_rows,
    sha256_bytes,
    serialize_component,
    sha256_file,
    unknown,
    verify_effective_config,
    verify_frame_references,
    verify_source_snapshot,
    wall_binding,
)
from backend.vision.config import DebugConfig, DetectorConfig          # noqa: E402
from backend.vision.detector import DropletDetector                    # noqa: E402
from backend.vision.parallel_walls import ParallelWallLocalizer        # noqa: E402
from backend.vision.rectified_measurement import (                     # noqa: E402
    FrameEvidence,
    ScaleEvidence,
    measure_generation_plugs,
    trace_summary,
)
from backend.vision.rectified_roi import wall_line_quad                # noqa: E402

MANIFEST_VERSION = 1
REPLAY_FRAME_REF_NOTE = (
    "容器帧序与 frames.ndjson 记录序一一对应（2026-09-23 审计核对过容器帧数==记录数），"
    "因此 decoded_frame_index 取软件帧序号"
)


def load_facts(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_start(facts: list[dict], start_monotonic: float) -> int:
    """返回第一个采集时间 >= start_monotonic 的软件帧序号。"""
    for row in facts:
        value = row["host_monotonic_timestamp"]["value"]
        if value is not None and float(value) >= float(start_monotonic):
            return int(row["software_frame_index"])
    raise RuntimeError(f"清单起点 {start_monotonic} 超出录像时间范围")


def resolve_end(facts: list[dict], start_index: int, duration_s: float) -> tuple[int, float]:
    """按**真实采集时间**求片段末帧：第一个时间 >= 起点+duration 的帧（不含）。"""
    start_time = None
    for row in facts:
        if int(row["software_frame_index"]) == start_index:
            start_time = float(row["host_monotonic_timestamp"]["value"])
            break
    if start_time is None:
        raise RuntimeError(f"帧 {start_index} 没有采集时间")
    deadline = start_time + float(duration_s)
    end_index = start_index
    for row in facts:
        index = int(row["software_frame_index"])
        if index < start_index:
            continue
        value = row["host_monotonic_timestamp"]["value"]
        if value is None or float(value) < deadline:
            end_index = index + 1
            continue
        end_index = index
        break
    else:
        end_index = int(facts[-1]["software_frame_index"]) + 1
    return end_index, start_time


def rectified_region_to_original(image_shape, wall_lines, x0: float, x1: float,
                                 y0: float, y1: float) -> np.ndarray | None:
    """把扶正坐标的矩形区域经逆透视映回原图四边形。"""
    height, width = image_shape[:2]
    geometry = wall_line_quad(width, height, wall_lines)
    if geometry is None:
        return None
    source, out_w, out_h = geometry
    destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                           dtype=np.float32)
    inverse = np.linalg.inv(cv2.getPerspectiveTransform(source, destination))
    corners = [(x0 - 0.5, y0 - 0.5), (x1 + 0.5, y0 - 0.5), (x1 + 0.5, y1 + 0.5), (x0 - 0.5, y1 + 0.5)]
    points = []
    for cx, cy in corners:
        vector = inverse @ np.array([cx, cy, 1.0], dtype=np.float64)
        points.append((float(vector[0] / vector[2]), float(vector[1] / vector[2])))
    return np.asarray(points, dtype=np.float32)


LOCALIZATION_DETAIL_KEYS = (
    "candidates", "pair_rejections", "competition", "motion", "coverage",
    "measurement_segment_px", "separation_px", "mid_y_px", "tilt_ratio",
    "pair_score", "interior_parallel_competitor_ids", "static_interior_competitor_ids",
    "candidate_ids", "rejection_histogram", "support",
)
"""定位失败时必须保留的诊断字段：竞争线对、共见区间、运动与覆盖证据、拒绝统计。

没有这些，只能看到一句截断的理由，无法判断「另一组同样可信的线对」到底是哪一组、
在哪、凭什么被判为竞争——也就无法区分「物理上有右侧管道」与「检测器偏好」。
"""


def _jsonable(value, depth: int = 0):
    """把定位证据变成可落盘的 JSON；ndarray 只记形状，不把像素灌进记录。"""
    if depth > 5:
        return "<max-depth>"
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={list(value.shape)} dtype={value.dtype}>"
    if isinstance(value, np.generic):
        return value.item()
    return f"<{type(value).__name__}>"


def localization_detail(localization) -> dict:
    """从一次定位结果里取出可用于诊断与可视化的部分（含失败帧）。"""
    geometry = getattr(localization, "geometry", None) or {}
    return {key: _jsonable(geometry.get(key)) for key in LOCALIZATION_DETAIL_KEYS
            if key in geometry}


def _candidate_lines(detail: dict) -> list[dict]:
    lines = []
    for item in (detail.get("candidates") or [])[:12]:
        if isinstance(item, dict) and "endpoints_px" in item:
            lines.append(item)
    return lines


def annotate(image: np.ndarray, localization, measurement, detail: dict | None = None) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:  # 单通道录像：查看材料统一成 BGR，避免 VideoWriter 崩
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    detail = detail or {}
    height, width = canvas.shape[:2]
    usable = bool(getattr(localization, "usable", False))
    # 先把候选线对画出来：**包括定位失败的帧**。失败帧里正是这些候选造成了歧义。
    for item in _candidate_lines(detail):
        x1, y1, x2, y2 = (float(v) for v in item["endpoints_px"])
        colour = (255, 128, 0) if item.get("length_ratio", 0) < 0.5 else (0, 200, 255)
        cv2.line(canvas, (round(x1), round(y1)), (round(x2), round(y2)), colour, 1, cv2.LINE_AA)
    if usable:
        for line in localization.wall_lines:
            cv2.line(canvas,
                     (round(float(line["x1"]) * width), round(float(line["y1"]) * height)),
                     (round(float(line["x2"]) * width), round(float(line["y2"]) * height)),
                     (0, 255, 255), 1, cv2.LINE_AA)
        summary = measurement.detection_trace_summary() if measurement is not None else {}
        axes = measurement.axes if measurement is not None else None
        if axes is not None:
            for left, right, _length in summary.get("selected_intervals", []) or []:
                quad = rectified_region_to_original(image.shape, localization.wall_lines,
                                                    float(left), float(right),
                                                    0.0, float(axes.output_height - 1))
                if quad is not None:
                    cv2.polylines(canvas, [np.round(quad).astype(np.int32)], True,
                                  (0, 255, 0), 1, cv2.LINE_AA)
    label = f"{localization.status} candidates={len(_candidate_lines(detail))}"
    if measurement is not None:
        label += f" | valid={measurement.valid} {measurement.reason}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 14), (0, 0, 0), -1)
    cv2.putText(canvas, label[:110], (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_detector(tuning_path: Path | None) -> DropletDetector:
    if tuning_path is not None and tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        config = DetectorConfig(**tuning["detector"])
    else:
        config = DetectorConfig()
    config.measurement_mode = "generation_plug"
    config.generation_min_length_ratio = 0.5
    return DropletDetector(config, DebugConfig())


def run_segment(segment: dict, *, media_root: Path, output: Path, detector: DropletDetector,
                warmup: int, still_offsets: list[int], write_video: bool,
                capture_id: str) -> dict:
    session = media_root / segment["session"]
    video = session / segment.get("video", "raw_frames.mkv")
    facts_path = session / segment.get("facts", "frames.ndjson")
    if not video.exists() or not facts_path.exists():
        return {"id": segment["id"], "status": "missing_media", "video": str(video),
                "facts": str(facts_path)}

    facts = load_facts(facts_path)
    by_index = {int(row["software_frame_index"]): row for row in facts}
    start_index = resolve_start(facts, float(segment["start_monotonic"]))
    end_index, start_time = resolve_end(facts, start_index, float(segment["duration_s"]))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"id": segment["id"], "status": "cannot_open_video", "video": str(video)}

    first = max(0, start_index - int(warmup))
    localizer = ParallelWallLocalizer(contrast_enhance=True)
    annotations: list[dict] = []
    rows_path = output / f"{segment['id']}_frames.ndjson"
    writer = IncrementalIndexWriter(rows_path)
    writer_video = None
    video_name = None
    segment_start_monotonic = None
    segment_end_monotonic = None
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    frames_with_candidate = 0
    valid_frames = 0
    frame_count = 0
    gray_min: int | None = None
    gray_max: int | None = None
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            return {"id": segment["id"], "status": "seek_failed",
                    "first_frame_index": first}
        for index in range(first, end_index):
            ok, frame = capture.read()
            if not ok:
                break
            fact = by_index.get(index)
            if fact is None:
                writer.append({"software_frame_index": index, "status": "no_frame_fact"})
                continue
            monotonic = fact["host_monotonic_timestamp"]["value"]
            if monotonic is None:
                writer.append({"software_frame_index": index, "status": "no_capture_time"})
                continue
            monotonic = float(monotonic)
            in_segment = index >= start_index
            if in_segment and segment_start_monotonic is None:
                segment_start_monotonic = monotonic
            localizer.observe(frame, frame_id=index, capture_monotonic=monotonic)
            if not in_segment:
                continue  # 暖机前缀：只观察，不出结论
            frame_count += 1
            segment_end_monotonic = monotonic
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            low = int(frame_gray.min())
            high = int(frame_gray.max())
            gray_min = low if gray_min is None else min(gray_min, low)
            gray_max = high if gray_max is None else max(gray_max, high)
            hardware_id = fact["hardware_frame_id"]["value"]
            reference = frame_reference(
                session_id=segment["session"], capture_id=capture_id,
                session_root=str(session),
                software_frame_index=index, hardware_frame_id=hardware_id,
                capture_monotonic=monotonic,
                container_path=segment.get("video", "raw_frames.mkv"),
                decoded_frame_index=index,
                # 内容哈希：就是这次解码得到的那一帧，之后可被独立重解码核对。
                content_sha256=sha256_bytes(frame.tobytes()),
                coordinate_space="original_full_frame")
            localization = localizer.localize(now_monotonic=monotonic)
            status = str(localization.status)
            detail = localization_detail(localization)
            status_counts[status] = status_counts.get(status, 0) + 1
            reason_counts[str(localization.reason)] = reason_counts.get(
                str(localization.reason), 0) + 1
            row = {
                "software_frame_index": index,
                "hardware_frame_id": hardware_id,
                "capture_monotonic": monotonic,
                "localization_status": status,
                "localization_reason": str(localization.reason)[:400],
                # 失败帧同样保留竞争线对、共见区间、运动与覆盖证据——否则无法判断歧义从哪来。
                "localization_detail": detail,
                "measurement_valid": False,
                "measurement_reason": f"localization_{status}",
                "source_frame_ref": reference,
                "source_frame_ref_note": REPLAY_FRAME_REF_NOTE,
            }
            measurement = None
            if localization.usable:
                trace: dict = {}
                measurement = measure_generation_plugs(
                    frame, detector=detector, localization=localization,
                    scale=ScaleEvidence(um_per_px=None, source="configured_optical",
                                        validated=False,
                                        detail="像素域回放：未声明独立标尺"),
                    # 硬件帧号用**事实记录里的真值**，不拿软件序号顶替；取不到就传 0
                    # 让测量如实拒测，而不是伪造一个能过一致性检查的值。
                    frame_evidence=FrameEvidence(
                        frame_id=index,
                        hardware_frame_id=(int(hardware_id)
                                           if isinstance(hardware_id, int)
                                           and not isinstance(hardware_id, bool) else 0),
                        capture_monotonic=monotonic,
                        localization_frame_id=int(localization.geometry.get("frame_id", 0) or 0),
                        time_source="host_clock_proxy"),
                    duct_depth_um=None, duct_depth_source="unknown",
                    duct_depth_validated=False, trace=trace)
                summary = trace_summary(trace)
                if measurement.valid:
                    valid_frames += 1
                if summary.get("selected_intervals"):
                    frames_with_candidate += 1
                row.update({
                    "measurement_valid": bool(measurement.valid),
                    "measurement_reason": measurement.reason,
                    "reference_width_px": summary.get("reference_width_px"),
                    "reference_width_source": summary.get("reference_width_source"),
                    "selected_intervals": summary.get("selected_intervals", []),
                })
            # 逐行增量写盘：写盘中断后已提交的行仍可核对。
            writer.append(row)

            offset = index - start_index
            want_still = offset in still_offsets
            if want_still or write_video:
                canvas = annotate(frame, localization, measurement, detail)
                if want_still:
                    name = f"{segment['id']}_f{index:07d}_annotated.jpg"
                    path = output / name
                    cv2.imwrite(str(path), canvas,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    annotations.append({
                        "still_path": name, "session_root": str(session),
                        "session_id": segment["session"], "capture_id": capture_id,
                        "software_frame_index": index, "hardware_frame_id": hardware_id,
                        "capture_monotonic": monotonic,
                        "localization_status": status,
                        "still_sha256": sha256_file(path),
                    })
                if write_video:
                    if writer_video is None:
                        video_name = f"{segment['id']}_view_only.mp4"
                        writer_video = cv2.VideoWriter(str(output / video_name),
                                                       cv2.VideoWriter_fourcc(*"mp4v"), 25.0,
                                                       (canvas.shape[1], canvas.shape[0]))
                    writer_video.write(canvas)
    finally:
        capture.release()
        if writer_video is not None:
            writer_video.release()
        writer.close()

    # 从**已提交的**逐行记录读回：这既做统计，也证明增量写盘的文件是完整的。
    committed = read_committed_rows(rows_path)
    truncated_tail = bool(committed) and committed[-1].get("_truncated_tail") is True
    rows = [row for row in committed if not row.get("_truncated_tail")]
    gaps = 0
    ordered = [row for row in rows if "software_frame_index" in row and "capture_monotonic" in row]
    for before, after in zip(ordered, ordered[1:]):
        if int(after.get("hardware_frame_id") or 0) != int(before.get("hardware_frame_id") or 0) + 1:
            gaps += 1
    # 静帧索引独立落盘：每张带文件哈希与采集身份，可离线核验。
    stills_index_path = output / f"{segment['id']}_stills_index.json"
    stills_index_path.write_text(
        json.dumps({"capture_id": capture_id, "session": segment["session"],
                    "session_root": str(session), "stills": annotations},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": segment["id"],
        "status": "ran",
        "session": segment["session"],
        "session_root": str(session),
        "note": segment.get("note", ""),
        "expectation": segment.get("expectation", ""),
        "manifest_start_monotonic": float(segment["start_monotonic"]),
        "manifest_duration_s": float(segment["duration_s"]),
        "resolved_first_frame_index": first,
        "resolved_segment_first_frame_index": start_index,
        "resolved_segment_end_frame_index_exclusive": end_index,
        "resolved_segment_first_capture_monotonic": start_time,
        "warmup_frames": int(warmup),
        "warmup_frame_indices": [first, start_index - 1] if start_index > first else [],
        "segment_first_frame_index": start_index,
        "input_frames": frame_count,
        "committed_rows": writer.committed,
        "truncated_tail_detected": truncated_tail,
        "capture_time_span_s": (None if segment_start_monotonic is None
                                or segment_end_monotonic is None
                                else round(segment_end_monotonic - segment_start_monotonic, 4)),
        "frame_id_gaps_in_segment": gaps,
        "localization_status_counts": status_counts,
        "localization_reason_top": sorted(reason_counts.items(),
                                         key=lambda item: -item[1])[:3],
        "valid_measurement_frames": valid_frames,
        "frames_with_at_least_one_candidate": frames_with_candidate,
        "independent_object_count": None,
        "independent_object_count_note": "无跟踪器，不报告独立对象数",
        "per_frame_rows": rows_path.name,
        "observed_gray_range": [gray_min, gray_max],
        "frame_reference_check": verify_frame_references(ordered, session_root=session),
        "annotated_stills": [item["still_path"] for item in annotations],
        "annotated_stills_index": stills_index_path.name,
        "view_only_video": video_name,
        "view_only_video_note": ("仅供查看，不是证据；原始证据是 session 目录下的 FFV1 录像"
                                 if video_name else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, default=Path("output"))
    parser.add_argument("--tuning", type=Path, default=None)
    parser.add_argument("--stills", default="0,100,200,300",
                        help="片段内相对帧号，逗号分隔")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    # 清单常由操作员在 Windows 上手写，编辑器可能写入 BOM：用 utf-8-sig 容忍。
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if int(manifest.get("manifest_version", 0)) != MANIFEST_VERSION:
        raise SystemExit(f"清单版本不支持：{manifest.get('manifest_version')!r}")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance_dir = args.output / "provenance"
    detector = build_detector(args.tuning)
    # 用**实际会被定位器使用的**阈值对象，不新建默认实例冒充。
    probe_localizer = ParallelWallLocalizer(contrast_enhance=True)
    thresholds = probe_localizer.thresholds
    detector_config = detector.config
    effective = EffectiveConfig(
        # 检测器按**全部**字段序列化，不用手写白名单——白名单会让「改了实际使用的
        # 参数、配置指纹却不变」。定位器同理，取真正被使用的阈值对象。
        detector=serialize_component(detector_config,
                                     reason="detector 生效对象的全部字段"),
        strict_detection_localization=declared(
            False, reason="本次是离线回放：逐帧送定位器，不走实时严格门控开关"),
        wall_source=declared("current_frame_localized",
                             reason="管壁由 ParallelWallLocalizer 逐帧定位；未通过即拒测"),
        image_shape=unknown("回放前未知；各片段的实测灰度域见 observed_gray_range"),
        scale_declaration=declared(None, reason="未声明独立标尺：像素域回放，不产出 µm"),
        depth_declaration=unknown("芯片深度未声明：图像不可推断"),
        imaging={
            "exposure_us": unknown("录像未记录有效逐帧曝光（逐帧字段恒为 0）"),
            "gain": unknown("录像未记录增益；2026-09-23 审计显示长录像未施加增益"),
            "readback_evidence": declared([], reason="离线回放无相机回读：本配置不含任何实测成像量"),
        },
        sampling=declared({"row_stride": 1, "mode": "continuous_within_segment"},
                          reason="片段内逐帧连续送入；暖机前缀只观察、不出结论"),
        localization=serialize_component(
            thresholds, reason="定位器生效对象的全部字段（取自实际 localizer 实例）"),
        wall_binding=wall_binding(source_label="current_frame_localized",
                                  wall_lines=None),
        config_change_policy=declared(
            "forbidden_after_provenance_write",
            reason="回放期间不改任何生效配置；测量行引用写入当时的指纹"),
    )
    try:
        bundle = build_bundle(provenance_dir, effective=effective.to_dict())
    except ProvenanceError as exc:
        # 离线模式没有硬件可保护，但追溯失败必须**明确输出**，不能继续产出一份
        # 无法核验的"结果"。
        print(json.dumps({"provenance_failure": repr(exc)}, ensure_ascii=False), flush=True)
        return 4

    still_offsets = [int(item) for item in str(args.stills).split(",") if item.strip()]
    warmup = int(manifest.get("warmup_frames", 24))
    results = []
    for order, segment in enumerate(manifest["segments"], start=1):
        outcome = run_segment(segment, media_root=args.media_root, output=args.output,
                              detector=detector, warmup=warmup, still_offsets=still_offsets,
                              write_video=not args.no_video,
                              capture_id=f"{segment['id']}#{order}")
        results.append(outcome)
        print(json.dumps({k: v for k, v in outcome.items() if k != "warmup_frame_indices"},
                         ensure_ascii=False), flush=True)

    verification = {
        "source_snapshot": verify_source_snapshot(provenance_dir),
        "effective_config": verify_effective_config(provenance_dir),
        "frame_references": [item.get("frame_reference_check") for item in results],
    }
    (provenance_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_path": str(args.manifest),
        "selection_rule": manifest.get("selection_rule"),
        "warmup_frames": warmup,
        "tuning_profile": (None if args.tuning is None else str(args.tuning)),
        "detector_config": {
            "measurement_mode": detector_config.measurement_mode,
            "generation_min_length_ratio": detector_config.generation_min_length_ratio,
        },
        "reference_width_source_expected": "frame_rectified_geometry",
        "physical_scale_validated": False,
        "interpretation_note": (
            "定位器报告的是**候选竞争**（存在另一组同样可信的线对）。"
            "物理来源——那是否是一条真实的右侧管道——**尚未确认**："
            "本批录像没有同步真值可以判定。"),
        "provenance": {
            "directory": str(provenance_dir),
            "source_fingerprint": bundle["source_fingerprint"],
            "config_fingerprint": bundle["config_fingerprint"],
            "consistent_snapshot": bundle["source"]["consistent_snapshot"],
            "verification": verification,
            "verify_command": (f".venv\\Scripts\\python.exe tools\\verify_provenance.py "
                               f"--directory {provenance_dir}"),
        },
        "segments": results,
    }
    (args.output / "replay_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
