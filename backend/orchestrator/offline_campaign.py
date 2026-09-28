"""离线采集入口：可安全导入、默认离线、复用 orchestrator 接口。

为什么单独建一个入口：2026-09-21 的长驻留脚本（``output/run_mpc_calibration_long_dwell.py``）
把设备连接、泵指令、测量与打分写在一个顶层脚本里，导入即产生副作用，而且测量走的是
``frame_jpeg`` 预览。本模块把「读原始帧 → 定位 → 测量 → 落盘 → 离线稳态分析」拆成可
测试、可离线的纯路径：

* 导入本模块**不**枚举相机、**不**连接串口、**不**操作泵，也不读任何配置。
* 测量输入只来自**原始帧包**（:class:`FramePacket`）；这里没有 ``frame_jpeg`` 通道，
  也不接受任何已缩放的预览。
* 只有 ``accepted`` 的行才进入稳态分析；被拒绝的帧写出理由并从尺寸列中排除。
* 设备接入由调用方注入（相机或保存帧序列），因此离线复现与将来接真机走同一条代码。

时间来源必须由帧包显式声明；本模块不把主机时钟冒称作采集时刻。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np

TIME_SOURCES = frozenset({"camera_frame_timestamp", "host_clock_proxy"})

CSV_FIELDS: tuple[str, ...] = (
    "elapsed_s",
    "phase",
    "q1_command_ul_min",
    "q2_command_ul_min",
    "frame_id",
    "hardware_frame_id",
    "capture_monotonic",
    "time_source",
    "time_is_proxy",
    "command_started_monotonic",
    "readback_completed_monotonic",
    "localization_status",
    "localization_reason",
    "separation_px",
    "measurement_valid",
    "measurement_reason",
    "scale_um_per_px",
    "scale_source",
    "scale_validated",
    "duct_width_px",
    "duct_width_um",
    "duct_depth_um",
    "duct_depth_source",
    "duct_depth_validated",
    "plug_count",
    "complete_plug_count",
    "plug_lengths_px",
    "plug_lengths_um",
    "equivalent_diameters_px",
    "equivalent_diameters_um",
    "accepted",
)


@dataclass(frozen=True)
class FramePacket:
    """一帧原始画面及其身份。

    ``image`` 必须是原始分辨率、未缩放的数组。**调用方保证**它来自原始帧通道而不是
    显示/传输用的预览：程序只能校验形状与 dtype，无法从数组本身判断它是不是被缩放过
    的预览图。若调用方知道期望尺寸，请填 ``expected_shape``，本模块会核对并在不符时拒绝。
    """

    image: np.ndarray
    frame_id: int
    hardware_frame_id: int
    capture_monotonic: float
    time_source: str = "host_clock_proxy"
    expected_shape: tuple[int, int] | None = None

    def validate(self) -> None:
        if self.image is None or getattr(self.image, "size", 0) == 0:
            raise ValueError("帧包图像为空")
        if not (isinstance(self.frame_id, int) and self.frame_id > 0):
            raise ValueError("帧包 frame_id 必须是正整数")
        if not (isinstance(self.hardware_frame_id, int) and self.hardware_frame_id > 0):
            raise ValueError("帧包 hardware_frame_id 必须是正整数")
        if self.hardware_frame_id != self.frame_id:
            raise ValueError("帧包 frame_id 与 hardware_frame_id 不一致")
        if not (math.isfinite(float(self.capture_monotonic)) and self.capture_monotonic > 0.0):
            raise ValueError("帧包 capture_monotonic 必须是正的有限值")
        if self.time_source not in TIME_SOURCES:
            raise ValueError(f"帧包时间来源必须显式声明，取值 {sorted(TIME_SOURCES)}")
        if self.expected_shape is not None:
            height, width = self.image.shape[:2]
            expected_height, expected_width = int(self.expected_shape[1]), int(self.expected_shape[0])
            if (int(width), int(height)) != (expected_width, expected_height):
                raise ValueError(
                    f"帧包尺寸与声明不符：期望 {expected_width}x{expected_height}，"
                    f"实际 {width}x{height}；声明不符说明这帧可能不是原始分辨率")


class FrameSource(Protocol):
    """帧包来源。``packets`` 只做顺序读取，不等待、不排队。"""

    def packets(self) -> Iterator[FramePacket]: ...


@dataclass(frozen=True)
class SavedFrameSequence:
    """从保存的原始帧文件读帧包（离线复现用）。

    每帧的身份与时间由调用方按文件名规则或显式列表给出；不猜时间。
    """

    paths: Sequence[Path]
    frame_ids: Sequence[int]
    capture_monotonic: Sequence[float]
    time_source: str = "host_clock_proxy"
    first_frame_id: int = 1
    interval_s: float = 1.0
    loader: Callable[[Path], np.ndarray] | None = None

    def packets(self) -> Iterator[FramePacket]:
        import cv2

        load = self.loader or (lambda path: cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        for index, path in enumerate(self.paths):
            image = load(path)
            frame_id = (int(self.frame_ids[index]) if self.frame_ids
                        else self.first_frame_id + index)
            moment = (float(self.capture_monotonic[index]) if self.capture_monotonic
                      else 1000.0 + index * float(self.interval_s))
            yield FramePacket(image=image, frame_id=frame_id, hardware_frame_id=frame_id,
                              capture_monotonic=moment, time_source=self.time_source)


@dataclass(frozen=True)
class PhaseSpec:
    """一档工况：命令值与该档的帧数上限。"""

    label: str
    q1_command_ul_min: float
    q2_command_ul_min: float
    frames: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignConfig:
    output_dir: Path
    phases: tuple[PhaseSpec, ...]
    duct_depth_um: float | None = None
    max_rows: int | None = None
    notes: str = ""


def _median_or_none(values: Sequence[float]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(finite)) if finite else None


def row_from_payload(packet: FramePacket, phase: PhaseSpec, payload: dict[str, Any],
                      *, elapsed_s: float, time_is_proxy: bool) -> dict[str, Any]:
    duct = payload.get("duct") or {}
    scale = payload.get("scale") or {}
    localization = payload.get("localization") or {}
    accepted = bool(payload.get("valid")) and bool(payload.get("has_complete_plug"))
    return {
        "elapsed_s": round(elapsed_s, 6),
        "phase": phase.label,
        "q1_command_ul_min": phase.q1_command_ul_min,
        "q2_command_ul_min": phase.q2_command_ul_min,
        "frame_id": int(packet.frame_id),
        "hardware_frame_id": int(packet.hardware_frame_id),
        "capture_monotonic": float(packet.capture_monotonic),
        "time_source": packet.time_source,
        "time_is_proxy": bool(time_is_proxy),
        # 离线入口不下发泵指令，因此这两列在离线运行为空；接真机时必须由采集侧填入
        # 指令开始与回读完成时刻，否则时间链不完整。
        "command_started_monotonic": phase.metadata.get("command_started_monotonic", ""),
        "readback_completed_monotonic": phase.metadata.get("readback_completed_monotonic", ""),
        "localization_status": localization.get("status", ""),
        "localization_reason": localization.get("reason", ""),
        "separation_px": localization.get("separation_px"),
        "measurement_valid": bool(payload.get("valid")),
        "measurement_reason": payload.get("reason", ""),
        "scale_um_per_px": scale.get("um_per_px"),
        "scale_source": scale.get("source", ""),
        "scale_validated": scale.get("validated"),
        "duct_width_px": duct.get("width_px"),
        "duct_width_um": duct.get("width_um"),
        "duct_depth_um": duct.get("depth_um"),
        # 未走到截面证据阶段（例如定位被拒）时，深度证据确实是未知的。
        "duct_depth_source": duct.get("depth_source") or "unknown",
        "duct_depth_validated": duct.get("depth_validated"),
        "plug_count": int(payload.get("plug_count", 0) or 0),
        "complete_plug_count": int(payload.get("complete_plug_count", 0) or 0),
        "plug_lengths_px": "|".join(f"{v:.3f}" for v in payload.get("plug_lengths_px", [])),
        "plug_lengths_um": "|".join(f"{v:.3f}" for v in payload.get("plug_lengths_um", [])),
        "equivalent_diameters_px": "|".join(
            f"{v:.3f}" for v in payload.get("equivalent_diameters_px", [])),
        "equivalent_diameters_um": "|".join(
            f"{v:.3f}" for v in payload.get("equivalent_diameters_um", [])),
        "accepted": accepted,
    }


def run_campaign(
    *,
    vision,
    frame_source: FrameSource,
    config: CampaignConfig,
    q1_by_phase: dict[str, float] | None = None,
    q2_by_phase: dict[str, float] | None = None,
) -> dict[str, Any]:
    """按档序消费帧包、测量并落盘。设备与泵都不在本函数内接触。

    ``vision`` 需提供生成区测量接口（:class:`backend.orchestrator.vision_adapter.PipelineVisionService`
    的 ``localize_parallel_walls`` / ``measure_generation_zone``）。``q1_by_phase`` 只用于
    **记录命令值**：本函数不下发任何泵指令。

    芯片深度以 ``CampaignConfig.duct_depth_um`` 与服务公开声明的一致性为准：两边都声明且
    不一致时直接抛 ``ValueError``，避免输出元数据与实际计算不符。写盘是增量的，异常时
    已写出的行保留在 CSV 里。
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series_path = output_dir / "diameter_series.csv"
    events_path = output_dir / "events.ndjson"

    # 芯片深度只有一个来源：优先用配置声明的，服务里也声明了就必须一致，
    # 否则输出元数据会和实际计算不符——这种情况直接拒绝运行。
    service_depth = None
    if hasattr(vision, "declared_chip_depth"):
        service_depth = vision.declared_chip_depth()
    declared_depth = config.duct_depth_um
    if (declared_depth is not None and service_depth is not None
            and service_depth.get("depth_um") is not None
            and abs(float(service_depth["depth_um"]) - float(declared_depth)) > 1e-6):
        raise ValueError(
            "芯片深度声明冲突：CampaignConfig.duct_depth_um="
            f"{declared_depth} 与 vision 声明 {service_depth['depth_um']} 不一致，"
            "拒绝运行以免元数据与实际计算不符")
    depth_for_measurement = declared_depth
    depth_source = "declared_chip_geometry" if declared_depth is not None else "unknown"
    if depth_for_measurement is None and service_depth is not None:
        depth_for_measurement = service_depth.get("depth_um")
        depth_source = str(service_depth.get("source") or "unknown")

    packets = frame_source.packets()
    events: list[dict[str, Any]] = []
    epoch: float | None = None
    rows_written = 0
    accepted_rows = 0
    rejected_rows = 0
    rejected_reasons: dict[str, int] = {}
    aborted: dict[str, Any] | None = None

    # 增量写盘：每行立即落盘，异常时已采数据不会丢。行不留在内存里，
    # 因此内存开销与帧数无关。
    with series_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        try:
            for phase in config.phases:
                q1 = (q1_by_phase or {}).get(phase.label, phase.q1_command_ul_min)
                q2 = (q2_by_phase or {}).get(phase.label, phase.q2_command_ul_min)
                effective_phase = PhaseSpec(phase.label, q1, q2, phase.frames, phase.metadata)
                emitted = 0
                while emitted < phase.frames:
                    if config.max_rows is not None and rows_written >= int(config.max_rows):
                        aborted = {"status": "row_limit_reached", "limit": int(config.max_rows)}
                        break
                    try:
                        packet = next(packets)
                    except StopIteration:
                        break
                    packet.validate()
                    if epoch is None:
                        epoch = float(packet.capture_monotonic)
                    vision.localize_parallel_walls(
                        packet.image, frame_id=packet.frame_id,
                        capture_monotonic=packet.capture_monotonic)
                    payload = vision.measure_generation_zone(
                        packet.image,
                        frame_id=packet.frame_id,
                        hardware_frame_id=packet.hardware_frame_id,
                        capture_monotonic=packet.capture_monotonic,
                        time_source=packet.time_source,
                        duct_depth_um=depth_for_measurement,
                        duct_depth_source=depth_source,
                    ) or {"valid": False, "reason": "not_evaluated"}
                    row = row_from_payload(
                        packet, effective_phase, payload,
                        elapsed_s=float(packet.capture_monotonic) - epoch,
                        time_is_proxy=packet.time_source != "camera_frame_timestamp")
                    writer.writerow(row)
                    handle.flush()
                    rows_written += 1
                    if row["accepted"]:
                        accepted_rows += 1
                    else:
                        rejected_rows += 1
                        # 定位被拒时把定位自己的原因作为直方图键：只写 localization_rejected
                        # 会把“当前帧没有线对”“证据过期”等具体原因压平成一个词。
                        key = str(row["measurement_reason"]) or "unknown"
                        if key.startswith("localization_") and row["localization_reason"]:
                            key = f"localization:{row['localization_reason']}"
                        rejected_reasons[key] = rejected_reasons.get(key, 0) + 1
                    emitted += 1
                events.append({"event": "phase_finished", "phase": phase.label, "frames": emitted,
                               "q1_command_ul_min": q1, "q2_command_ul_min": q2,
                               "pump_commands_sent": 0})
                if aborted is not None:
                    events.append({"event": "campaign_aborted", **aborted})
                    break
        except BaseException as exc:  # noqa: BLE001 — 任何异常都要保留已采数据
            handle.flush()
            aborted = {"status": "aborted", "error": f"{type(exc).__name__}: {exc}",
                       "rows_preserved": rows_written}
            events.append({"event": "campaign_aborted", **aborted})
            raise
        finally:
            handle.flush()
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    summary = {
        "output_dir": str(output_dir),
        "phases": [phase.label for phase in config.phases],
        "rows": rows_written,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "rejection_reasons": rejected_reasons,
        "duct_depth_um_used_for_measurement": depth_for_measurement,
        "duct_depth_source_used": depth_source,
        "duct_depth_declared_by_campaign": declared_depth,
        "duct_depth_declared_by_vision": (
            None if service_depth is None else service_depth.get("depth_um")),
        "notes": config.notes,
        "measurement_input": "raw frame packets only; no preview or JPEG path",
        "pump_commands_sent": 0,
        "streaming": ("增量写盘、行不驻留内存；这是有限离线回放，不是长期现场流式入口"
                      "（现场接入还需要有界缓冲、独立安全停止路径与设备侧背压）"),
        "aborted": aborted,
    }
    (output_dir / "session_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class LazyFrameSource:
    """把任意可迭代的帧包序列包成来源；**不预先物化**，内存与帧数无关。"""

    factory: Callable[[], Iterator[FramePacket]]

    def packets(self) -> Iterator[FramePacket]:
        return self.factory()


def packet_stream(packets: Iterable[FramePacket]) -> FrameSource:
    """从任意可迭代对象构造帧来源。

    若传入的是一次性生成器，只能消费一轮——这对单次 ``run_campaign`` 足够；
    需要重复运行请传序列或工厂。
    """
    if callable(packets) and not isinstance(packets, (list, tuple)):
        return LazyFrameSource(packets)
    return LazyFrameSource(lambda: iter(packets))


def replace_epoch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 elapsed_s 重算成相对首帧的相对时间（供离线分析使用）。"""
    if not rows:
        return rows
    base = float(rows[0]["capture_monotonic"])
    for row in rows:
        row["elapsed_s"] = round(float(row["capture_monotonic"]) - base, 6)
    return rows


def read_series(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _phase_time_bases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """该档的三种时间基准。

    指令开始与回读完成**不可混为一谈**：只有真的存在对应时刻才给出对应时长。
    冲突（指令晚于回读、同一字段出现多个不一致取值）不允许用来推导响应时间。
    """
    def collect(key: str) -> list[float]:
        return [value for value in (_finite_positive(row.get(key)) for row in rows)
                if value is not None]

    commands = collect("command_started_monotonic")
    readbacks = collect("readback_completed_monotonic")
    conflict: str | None = None
    if len(set(commands)) > 1:
        conflict = f"同一档内 command_started_monotonic 有 {len(set(commands))} 个不同取值"
    elif len(set(readbacks)) > 1:
        conflict = f"同一档内 readback_completed_monotonic 有 {len(set(readbacks))} 个不同取值"
    elif commands and readbacks and min(commands) > min(readbacks):
        conflict = "指令开始时刻晚于回读完成时刻，时间链自相矛盾"

    if conflict is not None:
        return {"origin": float(rows[0]["capture_monotonic"]), "kind": "observation",
                "command": None, "readback": None, "conflict": conflict}
    if commands:
        return {"origin": min(commands), "kind": "command_start",
                "command": min(commands), "readback": min(readbacks) if readbacks else None,
                "conflict": None}
    if readbacks:
        return {"origin": min(readbacks), "kind": "readback", "command": None,
                "readback": min(readbacks), "conflict": None}
    return {"origin": float(rows[0]["capture_monotonic"]), "kind": "observation",
            "command": None, "readback": None, "conflict": None}


def analyse_steady(path: Path, *, window_s: float = 120.0, cadence_s: float = 10.0,
                   start_s: float = 0.0, slope_limit: float = 0.1,
                   min_second_coverage: float = 0.6, max_gap_s: float = 15.0,
                   freshness_s: float = 5.0, min_window_points: int = 3) -> dict[str, Any]:
    """离线稳态分析：**只消费 accepted 行**，并且对稀疏、陈旧与乱序数据明确拒绝。

    时间语义（对应第四轮审核 A/B/C）：

    * 每个检查时刻**只用截至该时刻的数据**：新鲜度按**实际采样时间**判，不看未来点，
      也不用取整的桶时间；覆盖率、最大空档、新鲜度分别记录。
    * 区分“曾经确认过”与“当前是否稳定”：首次确认时间只作历史，
      当前状态单独输出（未确认 / 当前稳定 / 确认后失稳 / 确认后数据陈旧 / 覆盖不足），
      重新稳定时给出新的持续确认时间。
    * 指令开始与回读完成分别输出时长；缺哪个就留空。只有观测帧时间时输出观测时长，
      并标明不能当作泵指令响应时间。

    这些阈值是可配置的**工程口径**（120 s 窗口、10 s 检查、连续三次、|斜率|≤0.1、
    覆盖率 0.6、空档 15 s、新鲜度 5 s），不是物理标定结果；窗口确认时间也不是
    物理时间常数，三次连续检查互相重叠，不能当三次独立实验。
    """
    rows = read_series(path)
    replace_epoch(rows)
    moments = [float(row["capture_monotonic"]) for row in rows]
    out_of_order = sum(1 for a, b in zip(moments, moments[1:]) if b < a)
    duplicates = len(moments) - len(set(moments))

    phases_in_order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        phase = row["phase"]
        if phase not in grouped:
            grouped[phase] = []
            phases_in_order.append(phase)
        grouped[phase].append(row)

    results: dict[str, Any] = {
        "window_s": window_s, "cadence_s": cadence_s, "start_s": start_s,
        "slope_limit_um_s": slope_limit,
        "min_second_coverage": min_second_coverage, "max_gap_s": max_gap_s,
        "freshness_s": freshness_s,
        "criterion_is_engineering_default": (
            "窗口/节奏/连续次数/斜率门槛/覆盖率/空档/新鲜度都是可配置的工程口径，"
            "不是物理标定结果；窗口确认时间不是物理时间常数，三次连续检查互相重叠"),
        "rows": len(rows), "accepted_rows": sum(1 for row in rows if _truthy(row["accepted"])),
        "rejected_rows": sum(1 for row in rows if not _truthy(row["accepted"])),
        "time_anomalies": {"out_of_order_rows": out_of_order, "duplicate_timestamps": duplicates},
        "phases": {},
    }
    for phase in phases_in_order:
        phase_rows = grouped[phase]
        accepted = [row for row in phase_rows if _truthy(row["accepted"])]
        bases = _phase_time_bases(phase_rows)
        origin = float(bases["origin"])
        first_s = float(phase_rows[0]["capture_monotonic"]) - origin
        last_s = float(phase_rows[-1]["capture_monotonic"]) - origin
        entry: dict[str, Any] = {
            "frames": len(phase_rows), "accepted_frames": len(accepted),
            "phase_first_s": round(first_s, 3), "phase_last_s": round(last_s, 3),
            "phase_span_s": round(last_s - first_s, 3),
            "time_origin": bases["kind"],
            "time_origin_monotonic": round(origin, 4),
            "command_started_monotonic": bases["command"],
            "readback_completed_monotonic": bases["readback"],
            "time_conflict": bases["conflict"],
        }
        if bases["kind"] == "observation":
            entry["time_origin_note"] = ("没有泵指令时刻，只能报告观察窗内的确认时间，"
                                         "不能给出“指令到稳态”的时长")
        if out_of_order or duplicates:
            results["phases"][phase] = dict(entry, status="time_anomaly",
                                            reason="该会话存在乱序或重复时间戳，"
                                                   "分桶与覆盖率都会失真",
                                            current_status="time_anomaly")
            continue
        if not accepted:
            reasons: dict[str, int] = {}
            for row in phase_rows:
                reasons[str(row["measurement_reason"])] = reasons.get(
                    str(row["measurement_reason"]), 0) + 1
            results["phases"][phase] = dict(
                entry, status="no_accepted_samples",
                current_status="no_accepted_samples",
                reason="该档没有任何通过校验的完整柱塞", rejection_reasons=reasons,
                last_accepted_s=None)
            continue
        if last_s - first_s < window_s:
            results["phases"][phase] = dict(entry, status="window_not_covered",
                                            current_status="window_not_covered",
                                            reason="该档实际跨越时间不足一个窗口")
            continue

        # 每个样本保留**实际采样时间**与取值；一秒桶只在窗口内部临时生成。
        # 先按时间筛窗、再聚合：预先全局分桶会把 120.1–120.9 秒的未来样本塞进标号为
        # 120 的桶里，让“截至 120 秒”的检查看到未来数据。
        sample_times: list[float] = []
        sample_values: list[float] = []
        for row in accepted:
            values = [float(v) for v in str(row["equivalent_diameters_um"]).split("|") if v]
            if not values:
                continue
            moment = float(row["capture_monotonic"]) - origin
            sample_times.append(moment)
            sample_values.append(float(np.median(values)))
        times = np.asarray(sample_times, dtype=np.float64)
        values = np.asarray(sample_values, dtype=np.float64)
        if times.size < min_window_points:
            results["phases"][phase] = dict(entry, status="insufficient_accepted_samples",
                                            current_status="insufficient_accepted_samples",
                                            samples=int(times.size))
            continue
        last_accepted_s = float(times[-1])

        def window_check(end: float, kind: str) -> dict | None:
            """以 (end-window_s, end] 为窗口，**只用实际采样时间在窗口内的样本**。"""
            lower = end - window_s
            inside = (times > lower) & (times <= end)
            if int(inside.sum()) < min_window_points:
                return None
            window_times = times[inside]
            window_values = values[inside]
            bucket: dict[int, list[float]] = {}
            for moment, value in zip(window_times, window_values):
                bucket.setdefault(int(math.floor(moment)), []).append(value)
            seconds = sorted(bucket)
            if len(seconds) < min_window_points:
                return None
            axis = np.asarray([float(second) for second in seconds])
            series = np.asarray([float(np.median(bucket[second])) for second in seconds])
            # 覆盖率分子只数标号严格大于窗口左界的整秒：左界所在的那一秒只是部分秒，
            # 把它算进去会让分子超过窗口秒数（例如 0.5 s 网格在 120 s 窗口上得到 1.0083）。
            # 斜率仍用窗口内**全部**有数据的整秒桶，部分秒的样本也是真实测量值。
            covered_seconds = [second for second in seconds if float(second) > lower]
            coverage = len(covered_seconds) / float(window_s)
            boundaries = np.concatenate([[lower], window_times, [end]])
            max_gap = float(np.max(np.diff(boundaries)))
            age = float(end - window_times[-1])
            slope = float(np.polyfit(axis, series, 1)[0])
            coverage_ok = bool(coverage >= min_second_coverage)
            gap_ok = bool(max_gap <= max_gap_s)
            fresh = bool(age <= freshness_s)
            slope_ok = bool(abs(slope) <= slope_limit)
            return {
                "check_kind": kind,
                "end_s": round(float(end), 3),
                "window_bounds_s": [round(float(lower), 3), round(float(end), 3)],
                "window_boundary_note": "左开右闭 (end-window_s, end]；起点处的部分秒按实际采样时间计入",
                "samples": int(window_times.size),
                "second_buckets": len(seconds),
                "covered_seconds": len(covered_seconds),
                "coverage_denominator_s": float(window_s),
                "coverage_note": ("分子 = 标号严格大于窗口左界的整秒桶数（左界那一秒只是部分秒，不计入），"
                                  "分母 = 窗口秒数；因此覆盖率不超过 1，斜率仍用窗口内全部整秒桶"),
                "coverage": round(coverage, 4),
                "coverage_ok": coverage_ok,
                "span_s": round(float(axis[-1] - axis[0]), 3),
                "max_gap_s": round(max_gap, 3),
                "gap_ok": gap_ok,
                "last_sample_in_window_s": round(float(window_times[-1]), 3),
                "age_of_last_sample_s": round(age, 3),
                "fresh": fresh,
                "slope_um_s": round(slope, 6),
                "slope_ok": slope_ok,
                "pass": bool(coverage_ok and gap_ok and fresh and slope_ok),
            }

        checks = []
        earliest_end = max(window_s, start_s + window_s)
        for end in np.arange(earliest_end, last_s + 1e-9, cadence_s):
            checks.append(window_check(float(end), "cadence"))
        # 末次节奏检查可能比阶段实际末帧早不到一个 cadence；补一个以实际末帧结尾的窗口，
        # 这样“当前状态”确实截止到阶段末尾，而不是一个未声明的悬空间隔。
        if last_s > earliest_end - 1e-9:
            tail_check = window_check(float(last_s), "phase_end")
            if tail_check is not None and (not checks or checks[-1] is None
                                           or abs(checks[-1]["end_s"] - last_s) > 1e-9):
                checks.append(tail_check)
        if not checks or all(check is None for check in checks):
            results["phases"][phase] = dict(entry, status="no_window_evaluated",
                                            current_status="no_window_evaluated",
                                            reason="该档没有可评估的窗口")
            continue

        checks = [check for check in checks if check is not None]
        passes = [bool(check["pass"]) for check in checks]
        coverage_like_ok = [bool(check["coverage_ok"] and check["gap_ok"] and check["fresh"])
                            for check in checks]
        first_index = None
        for index in range(len(checks) - 2):
            if all(passes[index:index + 3]):
                first_index = index + 2
                break
        first_passing_index = next((index for index, ok in enumerate(passes) if ok), None)
        # 当前持续稳定 = 从某个检查到最后一个检查全部通过；它的**起点**是最终确认时间
        # （即该段连续通过的第三次检查），不是最后一个检查时刻。
        final_index = None
        if passes and passes[-1]:
            probe = len(passes) - 1
            while probe - 1 >= 0 and passes[probe - 1]:
                probe -= 1
            candidate = probe + 2
            if candidate <= len(checks) - 1:
                final_index = candidate
        tail_stale = bool(last_s - last_accepted_s > freshness_s)
        current_window = checks[-1]
        if first_index is None:
            # 把“覆盖不足”“斜率超限”“连续次数不够”分开，漂移不能被归因成覆盖不足。
            if not any(coverage_like_ok):
                current_status = "insufficient_coverage"
                reason = ("没有任何窗口同时满足覆盖率、最大空档与末端新鲜度；"
                          "稀疏或长期被拒的数据不能当作稳态")
            elif not any(passes):
                current_status = "slope_exceeds_limit"
                reason = "覆盖率与新鲜度满足，但所有窗口的斜率都超过门槛"
            else:
                current_status = "not_confirmed_consecutively"
                reason = "只有零散窗口达标，未形成连续三次通过"
        elif tail_stale:
            current_status = "stale_after_confirmation"
            reason = "确认之后最近一段持续被拒，不能以旧样本末时刻报告当前稳定"
        elif final_index is None:
            current_status = "lost_after_confirmation"
            reason = "确认之后出现失稳，当前窗口不满足判据"
        else:
            current_status = "currently_stable"
            reason = ""
        entry.update({
            "status": "ok" if current_status == "currently_stable" else current_status,
            "current_status": current_status,
            "reason": reason,
            "samples": int(times.size),
            "second_buckets": len({int(math.floor(t)) for t in times}),
            "last_accepted_s": round(last_accepted_s, 3),
            "tail_stale": tail_stale,
            "checks": len(checks),
            # 真实“首次通过”的检查，与“首次连续三次通过”的确认时刻分开。
            "first_passing_check_s": (None if first_passing_index is None
                                      else checks[first_passing_index]["end_s"]),
            "first_confirmation_s": None if first_index is None else checks[first_index]["end_s"],
            "first_confirmation_slope_um_s": (None if first_index is None
                                              else checks[first_index]["slope_um_s"]),
            "first_confirmation_absolute_monotonic": (
                None if first_index is None
                else round(origin + checks[first_index]["end_s"], 4)),
            "final_confirmation_s": None if final_index is None else checks[final_index]["end_s"],
            "final_confirmation_absolute_monotonic": (
                None if final_index is None
                else round(origin + checks[final_index]["end_s"], 4)),
            "final_confirmation_source": (
                None if final_index is None else
                ("reconfirmed_after_loss" if (first_index is not None
                                              and final_index != first_index)
                 else "first_confirmation")),
            "current_window_end_s": current_window["end_s"],
            "current_window_kind": current_window["check_kind"],
            "current_window_covers_phase_end": bool(
                abs(current_window["end_s"] - last_s) <= 1e-9),
            "current_window_slope_um_s": current_window["slope_um_s"],
            "current_window_pass": bool(current_window["pass"]),
            "current_window_age_s": current_window["age_of_last_sample_s"],
            "failure_counts": {
                "coverage_failed": sum(1 for check in checks if not check["coverage_ok"]),
                "gap_failed": sum(1 for check in checks if not check["gap_ok"]),
                "freshness_failed": sum(1 for check in checks if not check["fresh"]),
                "slope_failed": sum(1 for check in checks if not check["slope_ok"]),
                "passed": sum(1 for ok in passes if ok),
            },
            "later_violation": (None if first_index is None
                                else any(not ok for ok in passes[first_index + 1:])),
            "contiguous_pass_to_last_check": (None if first_index is None
                                             else all(passes[first_index:])),
            "window_detail": checks,
        })
        # 首次与最终确认各自成组：时长与尺寸必须来自**同一个**确认时刻，
        # 不能把首次时长配上最终稳态尺寸。
        for name, index in (("first_confirmation", first_index),
                            ("final_confirmation", final_index)):
            if index is None:
                entry[name] = None
                continue
            confirmed = checks[index]
            mask = (times > confirmed["end_s"] - window_s) & (times <= confirmed["end_s"])
            block = {
                "time_s": confirmed["end_s"],
                "absolute_monotonic": round(origin + confirmed["end_s"], 4),
                "slope_um_s": confirmed["slope_um_s"],
                "window_median_um": round(float(np.median(values[mask])), 4),
                "window_p10_p90_span_um": round(
                    float(np.percentile(values[mask], 90) - np.percentile(values[mask], 10)), 4),
                "observation_to_confirmation_s": round(
                    float(origin + confirmed["end_s"]
                          - float(phase_rows[0]["capture_monotonic"])), 3),
                "readback_to_confirmation_s": (
                    None if bases["readback"] is None
                    else round(float(origin + confirmed["end_s"] - float(bases["readback"])), 3)),
                "command_to_confirmation_s": (
                    None if bases["command"] is None
                    else round(float(origin + confirmed["end_s"] - float(bases["command"])), 3)),
                "note": ("时长与尺寸来自同一个确认时刻；不要用首次时长配最终稳态尺寸"
                         if name == "first_confirmation" else
                         "最终持续稳定的起点；与首次确认不同时说明中间发生过失稳"),
            }
            entry[name] = block
        first_block = entry.get("first_confirmation")
        if first_block is not None:
            entry["confirmed_window_median_um"] = first_block["window_median_um"]
            entry["confirmed_window_p10_p90_span_um"] = first_block["window_p10_p90_span_um"]
            entry["observation_to_confirmation_s"] = first_block["observation_to_confirmation_s"]
            entry["observation_to_confirmation_note"] = (
                "以该档首个观测帧为起点；它不是泵指令响应时间")
            entry["readback_to_confirmation_s"] = first_block["readback_to_confirmation_s"]
            entry["command_to_confirmation_s"] = first_block["command_to_confirmation_s"]
            entry["command_to_confirmation_note"] = (
                "对应**首次**确认；缺少指令开始时刻时留空"
                if bases["command"] is not None else "缺少指令开始时刻，不给指令到确认的时长")
            entry["command_to_steady_s"] = entry["command_to_confirmation_s"]
        results["phases"][phase] = entry
    return results


def iter_saved_frames(directory: Path, pattern: str = "*.png") -> list[Path]:
    return sorted(Path(directory).glob(pattern))
