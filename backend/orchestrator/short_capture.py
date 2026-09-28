"""短时验证采集的计划与入口：默认离线、默认不执行。

用途：把「下一次现场先录 60–120 秒完整原始动态片段」这件事变成可审核的计划对象，
而不是靠现场临时决定。本模块**不会**自己连接设备：

* :func:`plan_report` 只做校验与预算，任何设备字段缺失都列成待现场填写项；
* :func:`run_short_capture` 在 ``execution_enabled=False``（默认）时直接拒绝；
  真的执行时它也只消费调用方注入的帧来源，设备接入由现场侧装配。

三个证据分别管理，互不替代：**几何定位**（管壁/扶正）、**物理标尺**（μm/px）、
**芯片深度**（结构尺寸）。只有通道宽度假设不能标成独立校准；深度未知只输出轴向长度。

**已验证与未验证的边界**，两者不得互相顶替：

* 已验证（离线，可复跑）：有限离线回放、原始帧无损落盘与逐帧还原、逐帧停止判据
  （帧时间戳时长、帧间隔与读取超时、帧数与写入字节上限、写盘异常）、物理尺寸输出闸门。
* 未验证、依赖现场：来源卡死时的收尾（只有实现 :class:`CancellableFrameSource` 的
  ``read(timeout_s)`` 才能取消；否则只能由设备侧停止）、真机 100 Hz 无丢帧与背压、
  液量边界（没有液量传感器就不监测液量）、真实动态画面下的定位与测量精度。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

MIN_DURATION_S = 60.0
MAX_DURATION_S = 120.0
DEFAULT_FPS = 100.0


class CancellableFrameSource(Protocol):
    """现场侧应提供的可取消读取契约。

    ``read(timeout_s)`` 必须在至多 ``timeout_s`` 秒内返回一帧或 ``None``（本次无帧），
    并且**可被取消**；来源正常结束时抛 ``StopIteration``。记录器靠它区分「暂时没帧」
    与「采完了」，并在来源停滞时也能收尾。

    只提供 ``packets()`` 的阻塞迭代**无法**在卡死时返回；这种情况必须在计划里显式声明
    ``uncancellable_source_acknowledged``，并由设备侧承担停止责任。
    """

    def read(self, timeout_s: float) -> Any: ...


def _positive_finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


@dataclass(frozen=True)
class ScaleEvidencePlan:
    """标尺证据。``source`` 为 ``channel_width_assumption`` 时**不算**独立校准。"""

    source: str = "pending_site_measurement"
    reference_um: float | None = None
    reference_px: float | None = None
    image_path: str | None = None
    uncertainty_um_per_px: float | None = None
    captured_with_same_optics: bool = False

    @property
    def um_per_px(self) -> float | None:
        um = _positive_finite(self.reference_um)
        px = _positive_finite(self.reference_px)
        if um is None or px is None:
            return None
        return um / px

    @property
    def is_independent(self) -> bool:
        return bool(
            self.source in {"scale_bar", "known_width_mask", "measured"}
            and self.um_per_px is not None
            and self.captured_with_same_optics
            and self.image_path
        )

    def problems(self) -> list[str]:
        issues: list[str] = []
        if self.reference_um is not None and _positive_finite(self.reference_um) is None:
            issues.append(f"reference_um={self.reference_um} 必须是正的有限值")
        if self.reference_px is not None and _positive_finite(self.reference_px) is None:
            issues.append(f"reference_px={self.reference_px} 必须是正的有限值")
        if self.uncertainty_um_per_px is not None \
                and _positive_finite(self.uncertainty_um_per_px) is None:
            issues.append(f"uncertainty_um_per_px={self.uncertainty_um_per_px} 必须是正的有限值")
        if self.reference_um is not None and self.reference_px is not None \
                and self.um_per_px is None:
            issues.append("由 reference_um / reference_px 得到的标尺不是正的有限值")
        if self.source in {"scale_bar", "known_width_mask", "measured"} and not self.image_path:
            issues.append("声明了独立标尺来源，但没有登记标尺原图路径")
        return issues


@dataclass(frozen=True)
class DepthEvidencePlan:
    """芯片深度。图像里看不到，必须由现场声明；未声明只输出轴向长度。"""

    depth_um: float | None = None
    source: str = "unknown"
    evidence_note: str = ""

    @property
    def declared(self) -> bool:
        return bool(_positive_finite(self.depth_um) is not None
                    and self.source in {"declared_chip_geometry", "measured_depth"})

    def problems(self) -> list[str]:
        if self.depth_um is not None and _positive_finite(self.depth_um) is None:
            return [f"depth_um={self.depth_um} 必须是正的有限值"]
        return []


@dataclass(frozen=True)
class DeviceReadiness:
    """设备与液量就绪表。字段未知就留待现场填写，不猜。"""

    camera_unique_id: str | None = None
    pump_port: str | None = None
    pump_connection_verified: bool = False
    oil_remaining: str = "to_be_filled_on_site"
    water_remaining: str = "to_be_filled_on_site"
    allowed_q1_range: tuple[float, float] | None = None
    allowed_q2_range: tuple[float, float] | None = None
    available_travel_note: str = "to_be_filled_on_site"
    stop_boundary_note: str = "to_be_filled_on_site"


@dataclass(frozen=True)
class StorageBudget:
    """原始帧保存预算。默认**不抽帧**：抽帧就不是“完整原始动态片段”。"""

    width_px: int = 720
    height_px: int = 540
    bytes_per_pixel: float = 1.0        # 原始灰度无损保存
    capture_fps: float = DEFAULT_FPS
    persist_every_n: int = 1
    margin_factor: float = 1.5          # 元数据与并行写的余量
    capacity_limit_bytes: int = 8 * 1024 ** 3
    free_space_note: str = "to_be_filled_on_site"

    @property
    def persisted_fps(self) -> float:
        fps = _positive_finite(self.capture_fps)
        if fps is None:
            return 0.0
        return fps / max(1, int(self.persist_every_n))

    def problems(self) -> list[str]:
        issues: list[str] = []
        for name in ("width_px", "height_px"):
            if _positive_finite(getattr(self, name)) is None:
                issues.append(f"{name}={getattr(self, name)} 必须是正的有限值")
        if _positive_finite(self.bytes_per_pixel) is None:
            issues.append(f"bytes_per_pixel={self.bytes_per_pixel} 必须是正的有限值")
        if _positive_finite(self.capture_fps) is None:
            issues.append(f"capture_fps={self.capture_fps} 必须是正的有限值")
        if _positive_finite(self.margin_factor) is None:
            issues.append(f"margin_factor={self.margin_factor} 必须是正的有限值")
        if int(self.persist_every_n) < 1:
            issues.append(f"persist_every_n={self.persist_every_n} 必须 ≥ 1")
        if _positive_finite(self.capacity_limit_bytes) is None:
            issues.append(f"capacity_limit_bytes={self.capacity_limit_bytes} 必须是正的有限值")
        return issues

    def estimate(self, duration_s: float) -> dict[str, Any]:
        frames = int(math.ceil(max(0.0, float(duration_s)) * self.persisted_fps))
        per_frame = float(self.width_px) * float(self.height_px) * float(self.bytes_per_pixel)
        raw = per_frame * frames * float(self.margin_factor)
        return {
            "capture_fps": float(self.capture_fps),
            "persisted_fps": self.persisted_fps,
            "persist_every_n": int(self.persist_every_n),
            "frames": frames,
            "bytes_per_frame": int(per_frame),
            "estimated_bytes": int(raw),
            "estimated_gib": round(raw / float(1024 ** 3), 3),
            "capacity_limit_bytes": int(self.capacity_limit_bytes),
            "within_capacity": bool(raw <= float(self.capacity_limit_bytes)),
        }


@dataclass(frozen=True)
class CaptureGuards:
    """运行时的采集边界。这些是**运行中真正生效**的停止条件，不是说明文字。

    ``max_duration_s`` 与 ``no_frame_timeout_s`` 以**帧时间戳**计；``processing_budget_s``
    是独立的**单调墙钟**期限，只作运行守卫（来源停滞时防止无限等待），**不**参与任何
    采集时间或物理响应时间的计算。``processing_budget_s=None`` 表示不设墙钟期限。
    """

    max_duration_s: float = 120.0            # 帧时间戳计的采集时长上限
    max_bytes: int = 8 * 1024 ** 3           # 原始帧写入字节上限
    max_frames: int = 20000                  # 原始帧数上限
    no_frame_timeout_s: float = 5.0          # 相邻帧间隔 / 单次读取等待上限
    processing_budget_s: float | None = None  # 单调墙钟期限，仅作守卫

    def problems(self) -> list[str]:
        issues: list[str] = []
        if _positive_finite(self.max_duration_s) is None:
            issues.append(f"max_duration_s={self.max_duration_s} 必须是正的有限值")
        if _positive_finite(self.max_bytes) is None:
            issues.append(f"max_bytes={self.max_bytes} 必须是正的有限值")
        if int(self.max_frames) < 1:
            issues.append(f"max_frames={self.max_frames} 必须 ≥ 1")
        if _positive_finite(self.no_frame_timeout_s) is None:
            issues.append(f"no_frame_timeout_s={self.no_frame_timeout_s} 必须是正的有限值")
        if self.processing_budget_s is not None \
                and _positive_finite(self.processing_budget_s) is None:
            issues.append(f"processing_budget_s={self.processing_budget_s} 必须是正的有限值")
        return issues


@dataclass(frozen=True)
class ShortCapturePlan:
    """一次短时验证采集的计划。``execution_enabled`` 默认为 False。"""

    label: str = "baseline_70_20"
    q1_command_ul_min: float = 70.0
    q2_command_ul_min: float = 20.0
    duration_s: float = 90.0
    prime_and_purge_marked_separately: bool = True
    scale: ScaleEvidencePlan = field(default_factory=ScaleEvidencePlan)
    depth: DepthEvidencePlan = field(default_factory=DepthEvidencePlan)
    readiness: DeviceReadiness = field(default_factory=DeviceReadiness)
    budget: StorageBudget = field(default_factory=StorageBudget)
    guards: CaptureGuards = field(default_factory=CaptureGuards)
    manual_comparison_points: int = 10
    pixel_diagnostic_only: bool = False
    uncancellable_source_acknowledged: bool = False
    """来源只有阻塞式 ``packets()``（没有 ``read(timeout_s)``）时必须显式声明。

    这个声明只是**承认风险**：来源卡死时记录器无法自行收尾，停止只能由设备侧负责。
    它不构成任何停止能力，也不改变守卫行为——不要把它读成「已具备停止链」。
    """
    execution_enabled: bool = False


REQUIRED_SITE_FIELDS = (
    "camera_unique_id",
    "pump_port",
    "pump_connection_verified",
    "oil_remaining",
    "water_remaining",
    "available_travel_note",
    "stop_boundary_note",
)

SCALE_BINDING_TOLERANCE = 0.05
"""计划标尺与 vision 实际测量标尺的允许相对差。"""


def _pending_site_fields(plan: ShortCapturePlan) -> list[str]:
    pending = []
    for name in REQUIRED_SITE_FIELDS:
        value = getattr(plan.readiness, name)
        if value is None or value is False or str(value).strip() in ("", "to_be_filled_on_site"):
            pending.append(name)
    if plan.budget.free_space_note == "to_be_filled_on_site":
        pending.append("free_space_note")
    return pending


def plan_report(plan: ShortCapturePlan) -> dict[str, Any]:
    """校验计划并给出预算与待现场填写项。只读，不接触设备。

    两个状态严格分开：

    * ``offline_replay_ready``：计划**内部自洽**（数值合法、时长在验证区间、预算内、
      排气段单列、对照点足量、标尺或像素诊断模式已声明）——只说明可以离线回放。
    * ``site_ready_to_execute``：在上一状态之上，**现场字段确实填全**且显式开启执行。
      未知的现场字段不会因为一个布尔开关就变成执行许可。
    """
    parameter_blockers: list[str] = []
    site_pending: list[str] = []
    warnings: list[str] = []

    parameter_blockers += plan.scale.problems()
    parameter_blockers += plan.depth.problems()
    parameter_blockers += plan.budget.problems()
    parameter_blockers += plan.guards.problems()

    duration = _positive_finite(plan.duration_s)
    if duration is None:
        parameter_blockers.append(f"duration_s={plan.duration_s} 必须是正的有限值")
    elif not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        parameter_blockers.append(
            f"计划时长 {plan.duration_s} s 不在 {MIN_DURATION_S:g}–{MAX_DURATION_S:g} s 的验证区间内")
    if not plan.prime_and_purge_marked_separately:
        parameter_blockers.append("排气/充液段必须单独标记，且不纳入正式响应拟合")

    if plan.pixel_diagnostic_only:
        warnings.append("像素诊断模式：不认证物理尺寸，不输出 µm")
    elif not plan.scale.is_independent:
        parameter_blockers.append(
            "标尺不是独立证据：需要相同成像配置下的标尺原图、像素区间、物理长度与来源；"
            "只有通道宽度假设不能标成独立校准")
    if not plan.depth.declared:
        warnings.append("深度未声明：本次只输出轴向长度，不产出体积等效尺寸")

    if plan.manual_comparison_points < 10:
        parameter_blockers.append("人工对照点至少 10 个：原图、壁线、扶正图、液柱两端与计算尺寸")

    budget = plan.budget.estimate(plan.duration_s if duration else 0.0)
    if not budget["within_capacity"]:
        parameter_blockers.append("估算存储超过容量上限，需缩短时长或降低帧率")
    if int(plan.budget.persist_every_n) > 1:
        warnings.append(
            f"persist_every_n={plan.budget.persist_every_n} 会抽帧，"
            "不再是完整的原始动态片段；算法回归仍可用，但动态证据会被削弱")
    if plan.guards.max_duration_s > plan.duration_s:
        warnings.append("guards.max_duration_s 大于计划时长；以计划时长为准收尾")
    if plan.guards.processing_budget_s is None:
        warnings.append(
            "未设墙钟期限 processing_budget_s：来源停滞时只能靠帧间隔/读取超时或设备侧停止")
    if plan.uncancellable_source_acknowledged:
        warnings.append(
            "已声明来源不可取消（uncancellable_source_acknowledged）：来源卡死时记录器无法自行收尾，"
            "停止由设备侧负责；这只是承认风险，不是已具备的停止能力")

    site_pending += _pending_site_fields(plan)
    if not plan.scale.image_path and plan.scale.is_independent:
        site_pending.append("scale_image_path")

    offline_ready = not parameter_blockers
    site_ready = bool(offline_ready and plan.execution_enabled and not site_pending)
    return {
        "execution_enabled": bool(plan.execution_enabled),
        "offline_replay_ready": offline_ready,
        "site_ready_to_execute": site_ready,
        "parameter_blockers": parameter_blockers,
        "site_fields_pending": site_pending,
        "blockers": parameter_blockers + [f"现场字段未填写：{name}" for name in site_pending],
        "warnings": warnings,
        "duration_s": duration,
        "label": plan.label,
        "q1_command_ul_min": plan.q1_command_ul_min,
        "q2_command_ul_min": plan.q2_command_ul_min,
        "scale_is_independent": plan.scale.is_independent,
        "scale_um_per_px": plan.scale.um_per_px,
        "pixel_diagnostic_only": bool(plan.pixel_diagnostic_only),
        "uncancellable_source_acknowledged": bool(plan.uncancellable_source_acknowledged),
        "depth_usable_for_volume": plan.depth.declared,
        "storage_budget": budget,
        "guards": asdict(plan.guards),
        "manual_comparison_points": int(plan.manual_comparison_points),
        # 停止边界按「谁负责」分三类列出。把声明过的意图当成运行中生效的条件，
        # 是上一轮审核点出的过度声明；未实现项必须留在 declared_but_not_enforced 里。
        "stop_chain": {
            "enforced_by_recorder": [
                "帧时间戳达到计划 duration_s 或 guards.max_duration_s（按采集时间收尾）",
                "相邻帧间隔超过 guards.no_frame_timeout_s（迟到帧）",
                "可取消来源在 guards.no_frame_timeout_s 内无帧（读取超时）",
                "原始帧写入达到 guards.max_frames 或 guards.max_bytes",
                "原始帧写盘异常（OSError），已写帧保留",
                "单调墙钟超过 guards.processing_budget_s（仅作守卫，不是采集时间）",
            ],
            "declared_but_not_enforced": [
                "定位连续拒绝且仍无当前帧证据：记录器只逐帧记录拒绝理由，不据此停止",
                "剩余液量触及边界：没有液量传感器，记录器不监测液量",
                "来源卡死（阻塞且不可取消）：没有任何运行中判据能触发，见 source_contract",
            ],
            "device_side_responsibility": [
                "运行阻断或异常退出时的安全停止与状态确认，由设备所有者负责",
                "记录器不控制泵，也不下发任何泵指令",
                "取消并确认现场来源已真正停下",
            ],
        },
        "volume_monitoring": ("没有液量传感器：剩余液量只能由现场按已确认起始量与明确预算估算，"
                              "记录器既不测量也不据此停止"),
        "source_contract": {
            "cancellable_read_required_for_self_stop": True,
            "acknowledged_uncancellable": bool(plan.uncancellable_source_acknowledged),
            "note": ("入口只接受实现了 read(timeout_s)（可取消、有限等待，正常结束抛 StopIteration）"
                     "的来源；来源只有阻塞 packets() 时卡死无法收尾，必须声明 "
                     "uncancellable_source_acknowledged 并由设备侧停止"),
        },
        "acceptance_criteria": [
            "测量区完整包含目标直管及液柱",
            "真实检测器能在动态画面持续给出可复核轮廓",
            "像素与物理单位、帧时序一致",
            "缺失或拒绝如实记录",
        ],
        "not_a_physical_result": ("本计划只用于验证检测链；时长是计划值，"
                                  "不是达到稳态的结论"),
    }


PHYSICAL_SIZE_KEYS = ("equivalent_diameters_um", "plug_lengths_um", "duct_width_um")
"""像素诊断模式下**不得**进入记录的物理尺寸列。"""


def strip_physical_sizes(payload: dict[str, Any]) -> dict[str, Any]:
    """像素诊断模式的输出闸门：剥掉由未认证标尺换算出的物理尺寸，只留像素量。

    诊断模式不认证物理标尺，因此也不能把 vision 用**它自己的**内部标尺算出的 µm 写进
    记录——否则稳态分析会把这些列当作可用物理尺寸消费。深度是现场声明的芯片结构尺寸、
    不来自标尺，因此保留（并带着它自己的 ``duct_depth_source``）。
    """
    stripped = dict(payload)
    for key in PHYSICAL_SIZE_KEYS:
        stripped.pop(key, None)
    duct = dict(stripped.get("duct") or {})
    duct["width_um"] = None
    stripped["duct"] = duct
    scale = dict(stripped.get("scale") or {})
    scale.update({"um_per_px": None, "source": "pixel_diagnostic_only", "validated": False})
    stripped["scale"] = scale
    return stripped


class RawFrameWriter:
    """把每一帧原始像素**无损**写盘，并登记帧号/采集时间清单。

    保存与检测结果无关：先落盘、后测量，检测失败或超时都不会丢失已采帧。
    容器是 ``frames.bin`` + ``frames_manifest.csv``，按 offset/nbytes 可逐帧还原原始像素。
    """

    def __init__(self, directory: Path, *, binary_name: str = "frames.bin",
                 manifest_name: str = "frames_manifest.csv") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binary_path = self.directory / binary_name
        self.manifest_path = self.directory / manifest_name
        self._handle = self.binary_path.open("wb")
        self._manifest = self.manifest_path.open("w", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._manifest, fieldnames=[
            "index", "frame_id", "hardware_frame_id", "capture_monotonic", "time_source",
            "offset_bytes", "nbytes", "width", "height", "channels", "dtype"])
        self._writer.writeheader()
        self.frames_written = 0
        self.bytes_written = 0

    def write(self, packet) -> dict[str, Any]:
        image = np.ascontiguousarray(packet.image)
        payload = image.tobytes()
        offset = self.bytes_written
        self._handle.write(payload)
        self.bytes_written += len(payload)
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else int(image.shape[2])
        row = {
            "index": self.frames_written,
            "frame_id": int(packet.frame_id),
            "hardware_frame_id": int(packet.hardware_frame_id),
            "capture_monotonic": float(packet.capture_monotonic),
            "time_source": packet.time_source,
            "offset_bytes": offset,
            "nbytes": len(payload),
            "width": int(width),
            "height": int(height),
            "channels": channels,
            "dtype": str(image.dtype),
        }
        self._writer.writerow(row)
        self._manifest.flush()
        self.frames_written += 1
        return row

    def close(self) -> None:
        try:
            self._manifest.flush()
        finally:
            self._manifest.close()
            self._handle.flush()
            self._handle.close()


def run_short_capture(plan: ShortCapturePlan, *, vision=None, frame_source=None,
                      output_dir: Path | None = None,
                      writer_factory=RawFrameWriter) -> dict[str, Any]:
    """执行短时采集：先把每帧原始像素落盘，再测量；边界在运行中真实生效。

    本函数**不**打开相机、不连接串口、不操作泵：设备接入由现场侧装配，它只消费注入的
    帧来源。默认 ``execution_enabled=False`` 时直接拒绝；现场字段没填全同样拒绝。

    帧来源应实现 :class:`CancellableFrameSource` 的 ``read(timeout_s)``；只有阻塞
    ``packets()`` 时来源卡死无法收尾，必须在计划里显式声明
    ``uncancellable_source_acknowledged``，否则拒绝执行。

    运行中的停止判据：帧时间戳时长达上限、写入字节/帧数达上限、相邻帧间隔或单次读取
    等待超过无帧超时、写盘异常、以及单调墙钟超过 ``processing_budget_s``（仅作守卫）。
    停止时**已采集的原始帧与测量行都保留**。单帧检测失败只记录，不连带丢弃该帧的
    原始像素；但真机上检测变慢会影响下一次取帧，本记录器不保证无丢帧（见 summary）。
    """
    report = plan_report(plan)
    if not plan.execution_enabled:
        return {"status": "refused", "reason": "execution_enabled=False：本模块默认不执行采集",
                "plan_report": report}
    if not report["site_ready_to_execute"]:
        return {"status": "refused", "reason": "计划未就绪：参数或现场字段未满足",
                "plan_report": report}
    if vision is None or frame_source is None or output_dir is None:
        return {"status": "refused",
                "reason": "缺少设备侧帧来源、视觉服务或输出目录；设备接入由现场侧装配",
                "plan_report": report}

    reader = getattr(frame_source, "read", None)
    cancellable = callable(reader)
    if not cancellable and not plan.uncancellable_source_acknowledged:
        return {"status": "refused",
                "reason": ("帧来源只有阻塞式 packets()：来源卡死时记录器无法自行收尾。"
                           "请提供可取消的 read(timeout_s) 契约，或在计划里显式声明 "
                           "uncancellable_source_acknowledged=True 并由设备侧承担停止"),
                "plan_report": report}

    from .offline_campaign import CampaignConfig, CSV_FIELDS, PhaseSpec

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    measured_scale = None
    if hasattr(vision, "generation_measurement_scale"):
        try:
            measured_scale = vision.generation_measurement_scale()
        except Exception:  # noqa: BLE001 — 取不到标尺证据不能变成执行许可
            measured_scale = None
    scale_binding = {"plan_um_per_px": plan.scale.um_per_px,
                     "vision_um_per_px": None if not measured_scale else
                     measured_scale.get("um_per_px"),
                     "used_for_microns": False, "mode": "pixel_diagnostic_only"}
    if not plan.pixel_diagnostic_only:
        vision_scale = None if not measured_scale else measured_scale.get("um_per_px")
        if vision_scale is None:
            return {"status": "refused",
                    "reason": "vision 未提供可用标尺证据；无法把计划标尺绑定到实际测量",
                    "plan_report": report}
        relative = abs(float(plan.scale.um_per_px) - float(vision_scale)) / float(vision_scale)
        if relative > SCALE_BINDING_TOLERANCE:
            return {"status": "refused",
                    "reason": f"计划标尺 {plan.scale.um_per_px:.6f} 与 vision 实际测量 "
                              f"{float(vision_scale):.6f} μm/px 不一致（相对差 {relative:.3f}）",
                    "plan_report": report}
        scale_binding.update({"used_for_microns": True, "mode": "bound_to_vision",
                              "relative_difference": round(relative, 6)})

    depth_um = plan.depth.depth_um if plan.depth.declared else None
    config = CampaignConfig(output_dir=output,
                            phases=(PhaseSpec(plan.label, plan.q1_command_ul_min,
                                              plan.q2_command_ul_min, 0),),
                            duct_depth_um=depth_um, notes="short capture validation")

    recorder = writer_factory(output / "raw_frames")
    rows_path = output / "diameter_series.csv"
    stop = {"status": "completed", "reason": ""}
    processing_errors: list[dict[str, Any]] = []
    frame_id_gaps: list[dict[str, int]] = []
    phase_first: float | None = None
    previous_capture: float | None = None
    previous_hardware_id: int | None = None
    frames_seen = 0
    frames_received = 0
    started_at = time.monotonic()
    deadline = (None if plan.guards.processing_budget_s is None
                else started_at + float(plan.guards.processing_budget_s))

    def guarded_packets():
        """按可用契约取帧，并在每次取帧前检查单调墙钟期限。

        可取消来源用有限等待读取：超时就返回 ``None``（本次无帧），正常结束抛
        ``StopIteration``。只有阻塞迭代的来源只能在两帧之间检查期限——卡死时本函数
        不会返回，这正是必须在计划里声明不可取消的原因。
        """
        def wall_clock_stop() -> dict[str, Any]:
            return {"status": "stopped_wall_clock_guard",
                    "reason": f"运行墙钟超过 guards.processing_budget_s="
                              f"{plan.guards.processing_budget_s:g} s（仅作守卫，不是采集时间）"}

        if not cancellable:
            for packet in frame_source.packets():
                if deadline is not None and time.monotonic() >= deadline:
                    yield None, wall_clock_stop()
                    return
                yield packet, None
            return
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                yield None, wall_clock_stop()
                return
            wait = float(plan.guards.no_frame_timeout_s)
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - now))
            try:
                packet = reader(wait)
            except StopIteration:
                return
            if packet is None:
                if deadline is not None and time.monotonic() >= deadline:
                    yield None, wall_clock_stop()
                else:
                    yield None, {"status": "stopped_no_frame_timeout",
                                 "reason": f"来源在 {wait:.3f} s 内没有给出任何帧"}
                return
            yield packet, None

    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        try:
            for packet, guard_stop in guarded_packets():
                if guard_stop is not None:
                    stop = guard_stop
                    break
                if phase_first is None:
                    phase_first = float(packet.capture_monotonic)
                frames_received += 1
                if previous_hardware_id is not None \
                        and int(packet.hardware_frame_id) != previous_hardware_id + 1:
                    frame_id_gaps.append({
                        "after_hardware_frame_id": previous_hardware_id,
                        "next_hardware_frame_id": int(packet.hardware_frame_id),
                        "missing": int(packet.hardware_frame_id) - previous_hardware_id - 1})
                previous_hardware_id = int(packet.hardware_frame_id)
                if previous_capture is not None \
                        and float(packet.capture_monotonic) - previous_capture \
                        > float(plan.guards.no_frame_timeout_s):
                    stop = {"status": "stopped_gap_timeout",
                            "reason": f"相邻帧间隔 {float(packet.capture_monotonic) - previous_capture:.3f} s "
                                      f"超过 {plan.guards.no_frame_timeout_s:g} s"}
                    break
                elapsed = float(packet.capture_monotonic) - phase_first
                if elapsed > float(plan.guards.max_duration_s) \
                        or elapsed > float(plan.duration_s):
                    stop = {"status": "stopped_duration",
                            "reason": f"实际采集时间 {elapsed:.3f} s 达到上限"}
                    break
                if recorder.frames_written >= int(plan.guards.max_frames):
                    stop = {"status": "stopped_frame_limit",
                            "reason": f"达到帧数上限 {plan.guards.max_frames}"}
                    break
                per_frame = int(packet.image.size * packet.image.itemsize)
                if recorder.bytes_written + per_frame > int(plan.guards.max_bytes):
                    stop = {"status": "stopped_capacity",
                            "reason": f"写入将超过容量上限 {plan.guards.max_bytes} 字节"}
                    break
                # 1) 原始帧先落盘：落盘不依赖检测结果，单帧检测失败不会连带丢掉这一帧的
                #    原始像素。注意这不是「真机也不会丢帧」——检测串在本循环内，检测变慢
                #    会影响下一次取帧，见 summary 的 backpressure 与硬件帧号缺口。
                try:
                    recorder.write(packet)
                except OSError as exc:
                    stop = {"status": "stopped_storage_error",
                            "reason": f"写入原始帧失败：{exc}",
                            "frames_preserved": recorder.frames_written}
                    break
                frames_seen += 1
                previous_capture = float(packet.capture_monotonic)
                # 2) 再测量：单帧失败只记录，不中断采集。
                try:
                    vision.localize_parallel_walls(
                        packet.image, frame_id=packet.frame_id,
                        capture_monotonic=packet.capture_monotonic)
                    payload = vision.measure_generation_zone(
                        packet.image, frame_id=packet.frame_id,
                        hardware_frame_id=packet.hardware_frame_id,
                        capture_monotonic=packet.capture_monotonic,
                        time_source=packet.time_source,
                        duct_depth_um=depth_um,
                        duct_depth_source=("declared_chip_geometry" if depth_um is not None
                                           else "unknown"))
                except Exception as exc:  # noqa: BLE001
                    processing_errors.append({"frame_id": int(packet.frame_id),
                                              "error": f"{type(exc).__name__}: {exc}"})
                    payload = {"valid": False, "reason": "processing_error"}
                payload = payload or {"valid": False, "reason": "not_evaluated"}
                # 诊断模式下不得记录物理尺寸：vision 用的是它自己的内部标尺，
                # 未经验证就把 µm 写进记录，稳态分析会把它当真值消费。
                if plan.pixel_diagnostic_only:
                    payload = strip_physical_sizes(payload)
                from .offline_campaign import row_from_payload

                writer.writerow(row_from_payload(
                    packet,
                    PhaseSpec(plan.label, plan.q1_command_ul_min, plan.q2_command_ul_min,
                              plan.guards.max_frames),
                    payload,
                    elapsed_s=elapsed,
                    time_is_proxy=packet.time_source != "camera_frame_timestamp"))
                handle.flush()
        finally:
            recorder.close()

    summary = {
        "status": stop["status"],
        "stop_reason": stop.get("reason", ""),
        "plan_label": plan.label,
        "output_dir": str(output),
        "frames_persisted": recorder.frames_written,
        "raw_bytes_persisted": recorder.bytes_written,
        "manifest": str(recorder.manifest_path),
        "raw_binary": str(recorder.binary_path),
        "frames_measured": frames_seen,
        "frames_received": frames_received,
        "hardware_frame_id_gaps": {
            "count": len(frame_id_gaps),
            "first": frame_id_gaps[:3],
            "note": ("按硬件帧号序列的缺口统计上游丢帧：缺口 > 0 说明取帧侧已经丢过帧，"
                     "记录器无法补回，也不能据此推断无缝采集"),
        },
        "backpressure": ("同步检测串在取帧循环内：真机上检测变慢会影响下一次取帧，"
                         "本记录器不保证 100 Hz 无丢帧；原始记录与后处理分离、有界缓冲"
                         "属现场适配范围"),
        "processing_errors": processing_errors,
        "processing_error_count": len(processing_errors),
        "wall_clock_s": round(time.monotonic() - started_at, 3),
        "guard_clock": {
            "wall_clock_elapsed_s": round(time.monotonic() - started_at, 3),
            "processing_budget_s": plan.guards.processing_budget_s,
            "wall_clock_stop_used": stop["status"] == "stopped_wall_clock_guard",
            "not_a_measurement_time": ("墙钟只作运行守卫；采集时间一律取帧时间戳，"
                                       "capture_span_s 同样是帧时间戳之差"),
        },
        "source_contract": {
            "cancellable_read": bool(cancellable),
            "uncancellable_acknowledged": bool(plan.uncancellable_source_acknowledged),
            "self_stop_possible_on_stalled_source": bool(cancellable),
        },
        "measurement_mode": ("pixel_diagnostic_only" if plan.pixel_diagnostic_only
                             else "physical_scale_bound"),
        "physical_sizes_withheld": (list(PHYSICAL_SIZE_KEYS)
                                    if plan.pixel_diagnostic_only else []),
        "capture_span_s": (None if phase_first is None or previous_capture is None
                           else round(previous_capture - phase_first, 3)),
        "scale_binding": scale_binding,
        "depth_um": depth_um,
        "guards": asdict(plan.guards),
        "pump_commands_sent": 0,
        "measurement_input": "raw frames persisted first; no preview or JPEG path",
        "not_a_physical_result": ("本次只验证检测链；计划时长不是达到稳态的结论"),
        "plan_report": report,
    }
    (output / "session_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"status": "executed", "summary": summary, "plan_report": report}


def plan_to_dict(plan: ShortCapturePlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["report"] = plan_report(plan)
    return payload
