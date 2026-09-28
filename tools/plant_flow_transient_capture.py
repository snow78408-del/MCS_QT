"""Capture the start-up transient of the pump and measure it while it runs.

Why this exists: the pump takes minutes, not seconds, to reach a steady flow.
A 30 minute capture at 320 fps is about 224 GB of frames, so the frames cannot
be kept; velocity is measured on the fly from a sliding window and only the
resulting series (plus optional narrow strips) is written to disk.

The channel is locked from the image itself -- no saved ROI is used, because a
saved ROI goes stale as soon as the stage moves.

两个模式必须显式二选一且互斥；**不带参数只打印用法并退出，不接触任何设备**::

    # 离线回放已保存的 stack，不需要硬件（pixel scale 必须显式给出）
    .venv/Scripts/python.exe tools/plant_flow_transient_capture.py \
        --replay output/longcap-20260918-210955/stack.npy --rate 320 \
        --scale 1.725 --out output/transient-replay

    # 实机采集，需要一份填写完毕的会话计划（字段见 docs/bench_session_template_20260920.json）
    .venv/Scripts/python.exe tools/plant_flow_transient_capture.py \
        --live --plan path/to/session_plan.json

实机路径会先完整校验计划，再按设备身份加锁、连接、回读、运行并验证停泵。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.provenance import (  # noqa: E402
    EffectiveConfig,
    ProvenanceError,
    build_bundle,
    component_leaf_count,
    declared,
    readback,
    serialize_component,
    unknown,
    unused_component,
    wall_binding,
)
from backend.vision.flow_locking import (  # noqa: E402
    activity_band,
    channel_profiles,
    channel_start,
    estimate_velocity,
    pitch_from_autocorrelation,
)
from backend.vision.offline_analysis import screen_velocity  # noqa: E402

DEFAULT_GAPS = (1, 2, 3)


def load_pixel_calibration(path: Path) -> dict:
    """加载像素标定记录——**只有它能授权物理量输出**。

    任务书 §4 要求「缺少有效像素标定，不输出物理速度」，并且显式命令行标尺不算有效标定；
    有效标定指的是项目里带 schema、标定图像哈希与不确定度的 ``CalibrationRecord``。
    """
    from backend.vision.calibration import load_calibration

    record = load_calibration(Path(path))
    record.validate()
    return {
        "um_per_px": float(record.pixel_to_micron),
        "uncertainty_um_per_px": float(record.uncertainty_um_per_px),
        "source": f"calibration_record:{record.calibration_id}",
        "image_sha256": str(record.calibration_image_sha256),
        "validated": True,
    }


def unvalidated_scale(um_per_px: float) -> dict:
    """命令行直接给出的标尺：可用于像素域诊断，但不构成有效标定。"""
    return {
        "um_per_px": float(um_per_px),
        "uncertainty_um_per_px": None,
        "source": "command_line_unsourced",
        "image_sha256": None,
        "validated": False,
    }


# The live entry point below now wires the tested lifecycle to the direct MVS
# adapter, exact device identities, OS device locks and the per-frame facts
# recorder.  New capability gaps must be added here and will close the gate.
LIVE_ACCEPTANCE_GAPS: tuple[str, ...] = ()

# 计划校验的必填项。除任务书点名的「设备、基线流量、各路上下限、最大步长、累计输送上限、
# 运行时限、无帧超时、停机重试次数」之外，另要求像素标定比例——缺它则实机采到的只能是
# 像素域结果，拿不到 mm/s，等于白跑一轮实机。
_APPARATUS_REQUIRED = (
    ("chip_id", "设备/芯片标识"),
    ("observation_site", "观测位点"),
    ("pump_port", "泵串口"),
    ("pump_address", "泵地址"),
    ("camera_unique_id", "相机唯一标识"),
    ("scale_um_per_px", "像素标定比例"),
    ("scale_evidence", "像素标定来源"),
)
_FLOW_REQUIRED = (
    ("baseline_q1_ul_min", "基线 Q1 流量"),
    ("baseline_q2_ul_min", "基线 Q2 流量"),
    ("q1_allowed_range_ul_min", "Q1 允许区间"),
    ("q2_allowed_range_ul_min", "Q2 允许区间"),
    ("max_step_ul_min", "单步最大流量变化"),
    ("max_cumulative_delivery_each_ul", "各路累计输送上限"),
)
_CAPTURE_REQUIRED = (
    ("requested_fps", "请求帧率"),
    ("requested_exposure_us", "请求曝光"),
    ("camera_backend", "相机后端"),
    ("timestamp_source_and_units", "时间戳来源与单位"),
    ("output_directory", "输出目录"),
    ("max_no_valid_frame_s", "无有效帧超时"),
    ("max_backlog_frames", "未落盘积压上限"),
)
_TERMINATION_REQUIRED = (
    ("max_session_s", "运行时限"),
    ("stop_timeout_s", "停泵超时"),
    ("max_stop_retries", "停机重试次数"),
    ("physical_stop_method", "物理停机方式"),
)
# 稳定性判据必须写进计划，不能靠脚本内置的默认 dwell/斜率阈值——§4 要求稳定结论
# 覆盖完整要求时长，且未解除混叠或时间轴无效时不得下「已稳定」结论。
_STABILITY_REQUIRED = (
    ("window_s", "稳定判定窗口"),
    ("minimum_dwell_s", "最短驻留时长"),
    ("slope_limit_with_units", "斜率上限"),
    ("spread_limit_with_units", "离散度上限"),
    ("minimum_quality_fraction", "最低有效样本比例"),
)


def load_session_plan(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("会话计划 JSON 根必须是对象")
    return payload


def _blank(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_session_plan(plan: dict) -> list[str]:
    """返回未满足项；**只有返回空列表才允许进入实机路径**。

    只检查「有没有填」不够：空白模板和 ``execution_enabled=true`` 都能通过存在性检查，
    所以这里同时校验数值类型、正负号，以及基线流量是否落在允许区间内。
    """
    unmet: list[str] = []
    if plan.get("document_type") == "planning_template_not_application_configuration":
        unmet.append("计划仍是空白模板（document_type=planning_template_not_application_configuration）")
    if plan.get("execution_enabled") is not True:
        unmet.append("execution_enabled 不是 true（注意该标志不能替代其余完整校验）")
    if plan.get("control_authorized") is not True:
        unmet.append("control_authorized 不是 true（现场条件确认前禁止启动泵）")

    def check(section: str, fields: tuple) -> dict:
        node = plan.get(section)
        if not isinstance(node, dict):
            unmet.append(f"缺少 {section} 段")
            return {}
        for key, label in fields:
            if _blank(node.get(key)):
                unmet.append(f"{label}（{section}.{key}）未填写")
        return node

    apparatus = check("apparatus", _APPARATUS_REQUIRED)
    flow = check("flow_plan", _FLOW_REQUIRED)
    capture = check("capture_plan", _CAPTURE_REQUIRED)
    termination = check("termination", _TERMINATION_REQUIRED)
    stability = check("stability_criteria", _STABILITY_REQUIRED)

    for key, label in (("baseline_q1_ul_min", "Q1"), ("baseline_q2_ul_min", "Q2")):
        value = flow.get(key)
        if not _blank(value) and not _finite(value):
            unmet.append(f"基线 {label} 流量不是有限数值：{value!r}")
    q1 = flow.get("baseline_q1_ul_min")
    q2 = flow.get("baseline_q2_ul_min")
    if _finite(q1) and _finite(q2):
        if float(q2) <= 0.0 or float(q1) <= 2.0 * float(q2):
            unmet.append(
                f"基线流量必须满足 Q2>0 且 Q1>2×Q2：q1={q1!r}, q2={q2!r}")

    address = apparatus.get("pump_address")
    if not _blank(address) and not (
        isinstance(address, int) and not isinstance(address, bool) and 1 <= address <= 31
    ):
        unmet.append(f"泵地址必须是 [1, 31] 内的整数：{address!r}")
    backend = str(capture.get("camera_backend") or "").strip().lower()
    if backend and backend != "hikrobot-direct":
        unmet.append(
            "实机采集只允许 camera_backend='hikrobot-direct'，以保证硬件帧号和主机单调时间戳")
    camera_unique_id = str(apparatus.get("camera_unique_id") or "")
    if camera_unique_id and not camera_unique_id.upper().startswith("HIKROBOT:DIRECT:"):
        unmet.append(f"相机唯一标识不是 direct MVS 设备：{camera_unique_id!r}")

    if _blank(plan.get("onsite_ready_record")):
        unmet.append("缺少现场管路已固定、无折弯/泄漏且废液出口安全的确认记录（onsite_ready_record）")

    readiness = plan.get("software_readiness")
    if not isinstance(readiness, dict):
        unmet.append("缺少 software_readiness 段")
    else:
        for key, label in (
            ("start_ack_failure_stop_test_passed", "启动确认失败后的停泵测试"),
            ("timestamp_and_raw_capture_test_passed", "时间戳与原始帧保存测试"),
            ("stop_failure_test_passed", "停泵失败测试"),
        ):
            if readiness.get(key) is not True:
                unmet.append(f"{label}未通过（software_readiness.{key}）")
    for key, label in (("q1_allowed_range_ul_min", "Q1"), ("q2_allowed_range_ul_min", "Q2")):
        span = flow.get(key)
        if _blank(span):
            continue
        if not (isinstance(span, (list, tuple)) and len(span) == 2 and all(_finite(v) for v in span)):
            unmet.append(f"{label} 允许区间必须是两个有限数值组成的 [下限, 上限]：{span!r}")
            continue
        lower, upper = float(span[0]), float(span[1])
        if not lower < upper:
            unmet.append(f"{label} 允许区间上下限顺序错误：{span!r}")
            continue
        baseline = flow.get("baseline_q1_ul_min" if key.startswith("q1") else "baseline_q2_ul_min")
        if _finite(baseline) and not (lower <= float(baseline) <= upper):
            unmet.append(f"{label} 基线流量 {baseline!r} 不在允许区间 {span!r} 内")

    for key, label in (("max_step_ul_min", "单步最大流量变化"),
                       ("max_cumulative_delivery_each_ul", "各路累计输送上限")):
        value = flow.get(key)
        if not _blank(value) and not (_finite(value) and float(value) > 0.0):
            unmet.append(f"{label} 必须是正数：{value!r}")

    value = capture.get("max_no_valid_frame_s")
    if not _blank(value) and not (_finite(value) and float(value) > 0.0):
        unmet.append(f"无有效帧超时必须为正数：{value!r}")
    backlog = capture.get("max_backlog_frames")
    if not _blank(backlog) and not (isinstance(backlog, int) and not isinstance(backlog, bool) and backlog > 0):
        unmet.append(f"未落盘积压上限必须是正整数：{backlog!r}")
    for key, label in (("requested_fps", "请求帧率"), ("requested_exposure_us", "请求曝光")):
        value = capture.get(key)
        if not _blank(value) and not (_finite(value) and float(value) > 0.0):
            unmet.append(f"{label} 必须是正数：{value!r}")

    for key, label in (("window_s", "稳定判定窗口"), ("minimum_dwell_s", "最短驻留时长")):
        value = stability.get(key)
        if not _blank(value) and not (_finite(value) and float(value) > 0.0):
            unmet.append(f"{label} 必须是正数：{value!r}")
    fraction = stability.get("minimum_quality_fraction")
    if not _blank(fraction) and not (_finite(fraction) and 0.0 < float(fraction) <= 1.0):
        unmet.append(f"最低有效样本比例必须落在 (0, 1]：{fraction!r}")

    for key, label in (("max_session_s", "运行时限"), ("stop_timeout_s", "停泵超时")):
        value = termination.get(key)
        if not _blank(value) and not (_finite(value) and float(value) > 0.0):
            unmet.append(f"{label} 必须是正数：{value!r}")
    retries = termination.get("max_stop_retries")
    if not _blank(retries) and not (isinstance(retries, int) and not isinstance(retries, bool) and retries >= 0):
        unmet.append(f"停机重试次数必须是非负整数：{retries!r}")

    return unmet


@dataclass(frozen=True)
class FrameTiming:
    """一帧的时间与身份信息。**三种时钟分开记录，绝不互相顶替。**

    * ``host_received`` —— 主机收到该帧时的单调钟读数，是唯一可靠的采集时钟；
    * ``wall_clock`` —— 适配器给的墙钟时刻（``FrameData.timestamp``），会随系统时间调整回跳；
    * ``device_ticks`` —— 设备采集时间戳 ticks。**仓库没有记录其单位与时基**，
      因此只登记不换算，也不能充当时间轴。
    * ``hardware_frame_id`` —— 设备帧号，用于检出丢帧与重复。
    """

    host_received: float | None = None
    wall_clock: float | None = None
    device_ticks: int | None = None
    hardware_frame_id: int | None = None


QUALIFY_TOLERANCE_RATIO = 0.25


def qualify_window(timings: list, *, nominal_dt: float = 0.0,
                   tolerance_ratio: float = QUALIFY_TOLERANCE_RATIO
                   ) -> tuple[bool, str, float | None, str]:
    """窗口级时间轴资格检查，返回 ``(合格, 原因, 实测间隔, 时钟来源)``。

    当前测速算法要求**均匀采样**，所以这里逐项检查：时间戳有限、**严格递增**、采样间隔
    彼此一致（不一致即视为丢帧，直接拒绝而不是用平均间隔掩盖），以及设备帧号连续。
    合格时返回由实测时间戳算出的间隔——**速度必须用它换算，不能用设定帧率**。

    没有逐帧时间戳时返回 ``assumed_user_rate``，由调用方按使用者声明的采样率处理
    （这是离线旧录像的既有路径，结论必须带条件标记）。
    """
    if not timings:
        return False, "窗口内没有帧", None, ""
    has_host = all(t.host_received is not None for t in timings)
    has_wall = all(t.wall_clock is not None for t in timings)
    if not has_host and not has_wall:
        if any(t.host_received is not None or t.wall_clock is not None for t in timings):
            return False, "窗口内部分帧缺少时间戳（时钟读数不完整）", None, ""
        return True, "", float(nominal_dt), TIMELINE_ASSUMED_USER_RATE

    for source, values in ((CLOCK_HOST_RECEIVED, [t.host_received for t in timings]),
                           (CLOCK_WALL_CLOCK, [t.wall_clock for t in timings])):
        if source == CLOCK_WALL_CLOCK and has_host:
            continue
        if any(value is None for value in values):
            continue
        numeric = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numeric):
            return False, f"{source} 时间戳存在非有限值", None, source
        if len(numeric) < 2:
            continue
        diffs = [later - earlier for earlier, later in zip(numeric, numeric[1:])]
        if any(diff <= 0.0 for diff in diffs):
            return False, f"{source} 时间戳不是严格递增（乱序或重复）", None, source
        cadence = float(np.median(diffs))
        if any(abs(diff - cadence) > cadence * tolerance_ratio for diff in diffs):
            return False, (f"{source} 采样间隔不均匀（疑似丢帧），当前算法要求均匀采样，"
                           "不用平均间隔掩盖"), None, source
        ids = [t.hardware_frame_id for t in timings]
        if all(item is not None for item in ids):
            if any(later - earlier != 1 for earlier, later in zip(ids, ids[1:])):
                return False, "设备帧号不连续（丢帧或重复）", None, source
        return True, "", cadence, source
    return False, "窗口内的逐帧时间戳不完整，无法建立时间轴", None, ""


def live_acceptance_gaps() -> list[str]:
    """尚未完成的实机验收项——非空即表示 --live 必须拒绝执行。"""
    return list(LIVE_ACCEPTANCE_GAPS)


# ---------------------------------------------------------------- 采集会话

class CaptureLifecycleError(RuntimeError):
    """采集流程失败；携带应当对外报告的结论。"""


class ChannelWriteError(CaptureLifecycleError):
    """泵参数写入未能通过回读验证。"""


class PumpStartUnverifiedError(CaptureLifecycleError):
    """启动指令已发出，但回读未确认——此时泵**可能已经在转**。"""


class CameraSetupError(CaptureLifecycleError):
    """相机设置未能通过回读确认，拒绝以警告方式继续。"""


class NoValidFrameError(CaptureLifecycleError):
    """在计划允许的时间窗内没有拿到可用帧。"""


class StorageBacklogError(CaptureLifecycleError):
    """未落盘积压超过计划上限：写盘跟不上，必须终止受影响的试验。"""


@dataclass
class StopOutcome:
    """停泵结果。``STOP_UNVERIFIED`` 表示停机未被确认，需要现场人员确认。"""

    verdict: str
    attempts: int
    verified: bool
    error: str | None = None


def live_effective_config(*, plan: dict, detector_config=None,
                          strict_localization: dict | None = None,
                          wall_source: str = "current_frame_localized",
                          wall_binding_block: dict | None = None,
                          localization_thresholds=None) -> dict:
    """由计划与**实际生效对象**组装**生效配置**。

    检测与定位组件按 dataclass 的**全部**字段序列化，不用手写白名单——白名单会让
    「改了实际使用的参数、配置指纹却不变」（复审反例：
    ``generation_edge_mad_multiplier`` 3.0→13.0 指纹完全相同）。
    入口**未使用**某组件时标「未使用」，不以默认实例冒充实际生效对象。

    设定值与实测值分开：``imaging`` 记的是计划请求的设定，回读值只在其来源（事件名）
    与实测发生后才写入，且必须带 ``readback_source``。
    """
    capture = plan.get("capture_plan") or {}
    apparatus = plan.get("apparatus") or {}
    detector = (serialize_component(detector_config, reason="detector 生效对象的全部字段")
                if detector_config is not None
                else unknown("未能取得 detector 生效对象：不接受用默认值冒充"))
    localization = (serialize_component(localization_thresholds,
                                        reason="定位器生效对象的全部字段")
                    if localization_thresholds is not None
                    else unused_component("本入口不使用逐帧自动定位器；不以默认阈值实例冒充"))
    binding = wall_binding_block if wall_binding_block is not None else wall_binding(
        source_label=wall_source,
        proposal_path=(Path(apparatus["wall_proposal"])
                       if wall_source == "reused_proposal" and apparatus.get("wall_proposal")
                       else None),
    )
    if wall_source == "reused_proposal" and "wall_lines" not in binding \
            and "proposal_content_sha256" not in binding:
        # 复用的固定墙线必须可复现：只有来源标签时，换了墙线配置指纹也不变。
        raise ValueError(
            "复用固定提议必须绑定墙线值（wall_lines）或提议文件内容指纹"
            "（proposal_content_sha256）；缺绑定不得继续组装生效配置")

    requested_exposure = capture.get("requested_exposure_us")
    requested_fps = capture.get("requested_fps")
    return EffectiveConfig(
        detector=detector,
        strict_detection_localization=dict(strict_localization or
                                           unknown("本入口未声明严格定位模式")),
        wall_source=declared(wall_source, reason="本入口实际使用的管壁来源"),
        wall_binding=binding,
        image_shape=declared([int(apparatus.get("frame_width", 0) or 0),
                              int(apparatus.get("frame_height", 0) or 0)],
                             reason="计划声明的图像尺寸（未在本次采集前实测）"),
        scale_declaration=declared(apparatus.get("scale_um_per_px"),
                                   reason="计划里的历史标尺；本入口不据此产出 µm"),
        depth_declaration=unknown("芯片深度未声明：图像不可推断"),
        imaging={
            "exposure_us_setting": declared(requested_exposure,
                                            reason="计划请求值；实测回读见 report().events"),
            "frame_rate_setting": declared(requested_fps,
                                           reason="计划请求值；实测回读见 report().events"),
            "gain_setting": declared(capture.get("requested_gain"),
                                     reason="计划请求值；实测回读见 report().events"),
            "readback_evidence": declared(
                ["camera_exposure_verified", "camera_frame_rate_verified",
                 "camera_gain_verified"],
                reason="回读证据以会话事件形式落盘；本配置块记的是设定值，不是实测值"),
        },
        sampling=declared({
            "requested_fps": requested_fps,
            "still_count": 10,
            "raw_recording_format": capture.get("raw_recording_format"),
        }, reason="计划的采样策略"),
        localization=localization,
        config_change_policy=declared(
            "forbidden_after_provenance_write",
            reason=("配置在追溯写入后不再变更；本轮无任何代码路径在运行中改写生效配置。"
                    "如需变更必须先重新写追溯并产生新的 config_version，"
                    "测量行引用写入当时的指纹")),
    ).to_dict()


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_LIVE_BLOCKED = 3
EXIT_STOP_UNVERIFIED = 4
EXIT_TASK_GOAL_UNMET = 5
"""入口退出码。

``0`` 只表示**该入口预先声明的任务目标**达成。成功停泵不等于实验通过：
测量未验收的采集可以返回 ``0``（诊断入口声明的目标就是采集），也可以返回
``EXIT_TASK_GOAL_UNMET``（试验入口声明的目标包含测量验收）。停机未被确认一律
``EXIT_STOP_UNVERIFIED``，它压过其它一切结论。``EXIT_USAGE`` 是 argparse 的约定。
所有实机入口共用这张表，不再各自解释 ``stop.verified``。
"""

TASK_GOAL_CAMPAIGN = "campaign"
TASK_GOAL_DIAGNOSTIC_CAPTURE = "diagnostic_capture"
TASK_GOALS = frozenset({TASK_GOAL_CAMPAIGN, TASK_GOAL_DIAGNOSTIC_CAPTURE})


def exit_code(report: dict | None) -> int:
    """把一次会话报告解释成退出码。

    三层结果（运行完成 / 测量验收 / 停止状态）分别取自 ``report["result_layers"]``，
    绝不从单字段反推，也绝不用 ``result.get("stop", {}).get(...)``——``stop`` 为
    ``None`` 时那样会抛 ``AttributeError``，把「没有停泵证据」变成崩溃。
    """
    payload = report if isinstance(report, dict) else {}
    layers = payload.get("result_layers")
    if not isinstance(layers, dict):
        # 没有分层信息（旧记录或非会话报告）：不得把停泵成功当成目标达成。
        return EXIT_OK if payload.get("completed") is True else EXIT_TASK_GOAL_UNMET
    stop = layers.get("stop") if isinstance(layers.get("stop"), dict) else {}
    if stop.get("state") == "STOP_UNVERIFIED":
        return EXIT_STOP_UNVERIFIED
    return EXIT_OK if layers.get("task_goal_met") is True else EXIT_TASK_GOAL_UNMET


class BoundedFrameSink:
    """有界帧接收器：积压超过上限即判定写盘跟不上并失败。

    §3 会在这里落盘逐帧采集事实（含硬件时间戳与缺口），本类先定死「有界 + 超限即失败」
    与失败时机：不允许只写警告然后继续跑完整轮采集。
    """

    def __init__(self, *, max_backlog: int) -> None:
        self.max_backlog = int(max_backlog)
        self.pending: deque = deque()
        self.accepted = 0

    def submit(self, frame: object) -> None:
        if len(self.pending) >= self.max_backlog:
            raise StorageBacklogError(
                f"未落盘积压达到上限 {self.max_backlog} 帧：写盘跟不上，必须终止受影响的试验")
        self.pending.append(frame)
        self.accepted += 1

    def flush(self) -> int:
        written = len(self.pending)
        self.pending.clear()
        return written


# ---------------------------------------------------------------- 采集事实与时间轴

FRAME_FACT_VERSION = 1

# 时间轴的来源，按可信度从高到低排列。硬件 ticks 不在其列：仓库没有记录它的单位与时基，
# 无法换算成秒，因此只能登记 ticks 本身而不能充当时间轴。
TIMELINE_HOST_MONOTONIC = "host_monotonic"
TIMELINE_ADAPTER_WALL_CLOCK = "adapter_wall_clock"
TIMELINE_ASSUMED_USER_RATE = "assumed_user_rate"

# 逐帧可用的三种时钟，来源名必须显式区分，不得互相顶替。
CLOCK_HOST_RECEIVED = "host_received"
CLOCK_WALL_CLOCK = "wall_clock"


def _measured(value: object, *, unset_when, reason: str) -> dict:
    """把一个事实字段包成 ``{value, raw, reason}``。

    取不到或未被适配器填充时 ``value`` 为 ``None`` 并附原因，同时保留 ``raw``：
    既不把 dataclass 的默认 ``0`` 当成实测值，也不把信息丢掉。
    """
    if unset_when(value):
        return {"value": None, "raw": value, "reason": reason}
    return {"value": value, "raw": value, "reason": None}


def frame_facts(frame: object, *, software_index: int, sampled_monotonic: float,
                crop: dict | None = None, block_index: int | None = None) -> dict:
    """一帧的**采集事实**，不含任何分析结果。

    硬件帧号、硬件时间戳、丢包数的 dataclass 默认值都是 ``0``，与真实值无法区分，
    因此一律把 ``0`` 视为「适配器未填充」并记 null + 原因，同时保留 raw。
    硬件时间戳只登记 ticks 本身：其单位与时基在仓库里没有说明，不得自行假定换算。
    """
    return {
        "fact_version": FRAME_FACT_VERSION,
        "software_frame_index": int(software_index),
        "sampled_host_monotonic": float(sampled_monotonic),
        "hardware_frame_id": _measured(
            int(getattr(frame, "hardware_frame_id", 0) or 0), unset_when=lambda v: v <= 0,
            reason="适配器未填充硬件帧号（默认的 legacy 路径不设置该字段）"),
        "hardware_timestamp_ticks": _measured(
            int(getattr(frame, "hardware_timestamp_ticks", 0) or 0), unset_when=lambda v: v <= 0,
            reason="适配器未填充硬件时间戳（仅 direct 路径填充）"),
        "hardware_timestamp_units": {
            "value": None, "raw": None,
            "reason": "仓库未记录时间戳 ticks 的单位与时基，§3 禁止自行假定，故不换算",
        },
        "sdk_host_timestamp_ticks": _measured(
            int(getattr(frame, "sdk_host_timestamp_ticks", 0) or 0), unset_when=lambda v: v <= 0,
            reason="适配器未填充 SDK 主机时间戳（仅 direct 路径填充）"),
        "host_monotonic_timestamp": _measured(
            float(getattr(frame, "host_monotonic_timestamp", 0.0) or 0.0),
            unset_when=lambda v: v <= 0.0,
            reason="适配器未填充单调时钟时间戳（仅 direct 路径填充）；单调时钟读数不可能恰好为 0"),
        "exposure_time_us": _measured(
            getattr(frame, "exposure_time_us", None), unset_when=lambda v: v is None,
            reason="适配器未提供曝光时间（该字段默认为 None，与曝光 0 不可混淆）"),
        "lost_packet_count": _measured(
            int(getattr(frame, "lost_packet_count", 0) or 0), unset_when=lambda v: v < 0,
            reason="适配器未填充丢包计数；默认 0 无法区分「无丢包」与「未填充」"),
        "adapter_frame_id": int(getattr(frame, "frame_id", 0) or 0),
        "adapter_timestamp": float(getattr(frame, "timestamp", 0.0) or 0.0),
        "image_ref": (
            {"value": {"kind": "block_index", "index": int(block_index)}, "reason": None}
            if block_index is not None
            else {"value": None, "raw": None,
                  "reason": "本帧未落盘原始数据块（未配置数据块索引），不伪造索引"}
        ),
        "crop": (
            {"value": dict(crop), "reason": None}
            if crop is not None
            else {"value": None, "raw": None, "reason": "本帧未做通道裁剪，裁剪信息不可用"}
        ),
    }


class FrameFactsRecorder(BoundedFrameSink):
    """落盘逐帧采集事实，并检出重复帧、缺帧与非单调时间戳。

    时间轴来源按可信度择一，并把选择结果写进摘要：

    * ``host_monotonic`` —— direct 路径提供的单调时钟秒，最可靠；
    * ``adapter_wall_clock`` —— legacy 路径只有 ``timestamp=time.time()``，可用但**非单调安全**，
      必须逐帧验证单调性，出现回退就记为异常；
    * ``assumed_user_rate`` —— 离线旧录像没有逐帧时间戳时按使用者显式给出的采样率推算，
      **必须标记为假定时间轴**，不得当作实测时序。

    有界性：每条记录写完即落盘，``pending_since_flush`` 只在写入被延后时累积；一旦超过
    ``max_backlog`` 就抛 ``StorageBacklogError``，防止漏写被无声吞掉。
    """

    def __init__(self, *, path: Path, max_backlog: int,
                 timeline_source: str = TIMELINE_HOST_MONOTONIC,
                 timeline_assumed: bool = False, flush_every: int = 1) -> None:
        super().__init__(max_backlog=max_backlog)
        self.path = Path(path)
        self.timeline_source = timeline_source
        self.timeline_assumed = bool(timeline_assumed)
        self.flush_every = max(1, int(flush_every))
        self.software_index = 0
        self.anomalies: list[dict] = []
        self.duplicate_frames = 0
        self.missing_frames = 0
        self.non_monotonic_timestamps = 0
        self.pending_since_flush = 0
        self._last_hardware_id: int | None = None
        self._last_timeline: float | None = None
        self._handle = None

    # ------------------------------------------------------------ 写入

    def submit(self, frame: object, *, crop: dict | None = None,
               block_index: int | None = None) -> None:
        if self.pending_since_flush >= self.max_backlog:
            raise StorageBacklogError(
                f"未落盘记录达到上限 {self.max_backlog} 帧（写盘被延后）："
                "写盘跟不上，必须终止受影响的试验")
        facts = frame_facts(frame, software_index=self.software_index,
                            sampled_monotonic=float(self._clock_value()), crop=crop,
                            block_index=block_index)
        self.software_index += 1
        self._detect_anomalies(facts)
        self._write_line(facts)
        self.accepted += 1
        self.pending_since_flush += 1
        if self.pending_since_flush >= self.flush_every:
            self.flush()

    def _clock_value(self) -> float:
        return time.monotonic()

    def _open(self):
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def _write_line(self, payload: dict) -> None:
        handle = self._open()
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def flush(self) -> int:
        if self._handle is not None:
            self._handle.flush()
        written = self.pending_since_flush
        self.pending_since_flush = 0
        return written

    def close(self) -> None:
        self.flush()
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # ------------------------------------------------------------ 异常检出

    def timeline_value(self, facts: dict) -> float | None:
        """按选定的时间轴取本帧的时间（秒）；取不到返回 None。"""
        if self.timeline_source == TIMELINE_HOST_MONOTONIC:
            value = facts["host_monotonic_timestamp"]["value"]
            return None if value is None else float(value)
        if self.timeline_source == TIMELINE_ADAPTER_WALL_CLOCK:
            value = float(facts["adapter_timestamp"])
            return value if value > 0.0 else None
        return None  # 假定时间轴由调用方按采样率推算，这里不提供逐帧值

    def _detect_anomalies(self, facts: dict) -> None:
        hardware_id = facts["hardware_frame_id"]["value"]
        if hardware_id is not None:
            if self._last_hardware_id is not None:
                if hardware_id == self._last_hardware_id:
                    self.duplicate_frames += 1
                    self._anomaly("duplicate_hardware_frame_id", facts, hardware_id)
                elif hardware_id > self._last_hardware_id + 1:
                    self.missing_frames += hardware_id - self._last_hardware_id - 1
                    self._anomaly("missing_hardware_frames", facts,
                                  f"{self._last_hardware_id} -> {hardware_id}")
                elif hardware_id < self._last_hardware_id:
                    self._anomaly("hardware_frame_id_regressed", facts,
                                  f"{self._last_hardware_id} -> {hardware_id}")
            self._last_hardware_id = hardware_id

        moment = self.timeline_value(facts)
        if moment is not None:
            if self._last_timeline is not None and moment <= self._last_timeline:
                self.non_monotonic_timestamps += 1
                self._anomaly("non_monotonic_timestamp", facts,
                              f"{self._last_timeline} -> {moment}")
            self._last_timeline = moment

    def _anomaly(self, kind: str, facts: dict, detail: object) -> None:
        self.anomalies.append({
            "kind": kind,
            "software_frame_index": facts["software_frame_index"],
            "detail": str(detail),
        })

    # ------------------------------------------------------------ 汇报

    def sampling_premise_ok(self) -> tuple[bool, str]:
        """当前窗口是否满足分析前提。

        前提有三条：时间轴不是假定值、逐帧时间单调、以及帧号没有缺口／回退。
        不满足就必须拒绝分析该窗口，而不是照旧给结论。
        """
        if self.timeline_source == TIMELINE_ASSUMED_USER_RATE or self.timeline_assumed:
            return False, "时间轴是按使用者给出的采样率假定的，不是逐帧实测时序"
        if self.non_monotonic_timestamps:
            return False, f"存在 {self.non_monotonic_timestamps} 个非单调时间戳"
        if self.missing_frames or self.duplicate_frames:
            return False, (f"帧号缺口 {self.missing_frames} 帧、重复 {self.duplicate_frames} 帧")
        return True, ""

    def summary(self) -> dict:
        ok, reason = self.sampling_premise_ok()
        return {
            "records": self.accepted,
            "timeline_source": self.timeline_source,
            "timeline_assumed": self.timeline_assumed,
            "duplicate_frames": self.duplicate_frames,
            "missing_frames": self.missing_frames,
            "non_monotonic_timestamps": self.non_monotonic_timestamps,
            "anomalies": list(self.anomalies),
            "sampling_premise_ok": ok,
            "sampling_premise_reason": reason,
        }



class LiveCaptureSession:
    """一次实机采集会话：从「泵可能运行」到「停泵已验证」整段锁在设备锁内。

    顺序固定为::

        校验计划 → 获取设备独占锁 → 连接并核验初始状态 → 配置相机并回读 →
        开始连续采集、保存指令前基线 → 写泵参数并验证 → 发出启动指令并验证 →
        有界运行 → 停泵并验证 → 结束采集、关闭设备 → 释放锁

    依赖全部注入（``pump`` / ``camera`` / ``lock`` / ``sink`` / ``clock``），因此每条
    失败路径都能用 mock 覆盖，不需要真机。原始异常与清理异常分开保存在
    ``original_error`` 与 ``cleanup_errors``，互不覆盖。
    """

    def __init__(self, *, plan: dict, pump, camera, locks, sink, output_dir: Path,
                 log, clock=time.monotonic,
                 task_goal: str = TASK_GOAL_CAMPAIGN,
                 provenance: dict | None = None,
                 allow_unprovenanced: bool = False) -> None:
        if task_goal not in TASK_GOALS:
            raise ValueError(f"未知的任务目标 {task_goal!r}，可选 {sorted(TASK_GOALS)}")
        self.task_goal = str(task_goal)
        # 追溯规格：{"directory": Path, "effective": dict, "root": Path|None, "config_version": int}
        self.provenance_spec = provenance
        # 只允许**无设备单元测试**经 for_device_free_tests() 打开；真机入口不得提供该开关。
        self.allow_unprovenanced = bool(allow_unprovenanced)
        self.provenance: dict | None = None
        # 是否已经碰过设备。追溯写入失败发生在硬件动作**之前**，此时必须仍为 False。
        self.hardware_touched = False
        self.plan = plan
        self.pump = pump
        self.camera = camera
        # 设备锁：与主程序共用同一批锁键（泵按规范化串口，相机按稳定设备标识）。
        self.locks = list(locks)
        self.sink = sink
        self.output_dir = Path(output_dir)
        self.log = log
        self._clock = clock
        # 启动指令一旦可能发出即置位；此后任何退出路径都必须走停泵验证。
        self.pump_may_be_running = False
        self.started = False
        self.events: list[str] = []
        self.commands: list[dict] = []
        self.output_error: str | None = None
        self.original_error: BaseException | None = None
        self.cleanup_errors: list[BaseException] = []
        self.stop_outcome: StopOutcome | None = None
        self.baseline: dict | None = None
        self.frames_seen = 0
        self.frames_accepted = 0
        # 观察序列是否真的跑到终点。与「停泵成功」「测量验收」是三件不同的事。
        self.run_finished = False

    # ------------------------------------------------------------ 主流程

    def run(self) -> dict:
        try:
            self._prepare_provenance()      # 必须在任何硬件动作之前
            self._acquire_locks()
            self._record("device_lock_acquired")
            self._connect_and_verify_initial_state()
            self._configure_camera_and_read_back()
            self._begin_capture_and_save_baseline()
            self._write_pump_parameters()
            self._start_infusion()
            self._run_bounded()
        except BaseException as exc:  # 含 KeyboardInterrupt / CancelledError
            self.original_error = exc
        finally:
            self._cleanup()
            self._close_sink()
        payload = self.report()
        self._finalize_outputs(payload)
        return payload

    @classmethod
    def for_device_free_tests(cls, **kwargs):
        """**仅供无设备单元测试**：显式声明本轮不做追溯。

        真机入口不得使用——CLI 没有对应参数，`tests/` 里有守卫断言这一点
        （`test_live_entrypoints_expose_no_provenance_bypass`）。
        这与「没有可核验的运行记录，就不允许动设备」的契约一致：默认拒绝，
        只有测试能显式、可追溯地打开。
        """
        kwargs["allow_unprovenanced"] = True
        return cls(**kwargs)

    def _prepare_provenance(self) -> None:
        """在任何硬件动作之前写下源码快照与生效配置。

        失败（或现场入口根本没给规格）即抛 :class:`ProvenanceError`：
        **没有可核验的运行记录，就不允许动设备**。
        """
        if not self.provenance_spec:
            if not self.allow_unprovenanced:
                self.provenance = {
                    "written": False,
                    "why": ("缺追溯规格：现场入口必须在任何设备动作之前提供源码快照与生效配置。"
                            "无设备单元测试请用 LiveCaptureSession.for_device_free_tests()")}
                raise ProvenanceError(self.provenance["why"])
            self.provenance = {
                "written": False,
                "why": ("经 for_device_free_tests() 显式声明的**无设备测试**会话；"
                        "本记录不具备追溯性")}
            self._record("provenance_skipped_device_free_test_interface")
            return
        spec = dict(self.provenance_spec)
        try:
            bundle = build_bundle(Path(spec["directory"]),
                                  effective=dict(spec.get("effective") or {}),
                                  root=spec.get("root"),
                                  config_version=int(spec.get("config_version", 1)))
        except ProvenanceError as exc:
            # 记录失败原因再抛出：报告里要能看到「为什么这轮没有可核验的记录」。
            self.provenance = {"written": False, "why": repr(exc)}
            raise
        self.provenance = {
            "written": True,
            "directory": str(spec["directory"]),
            "source_fingerprint": bundle["source_fingerprint"],
            "config_fingerprint": bundle["config_fingerprint"],
            "config_version": bundle["config"]["config_version"],
        }
        self._record("provenance_written_before_hardware_actions")
        # 让 sink 能把配置版本写进每一行测量记录。
        setter = getattr(self.sink, "set_provenance", None)
        if callable(setter):
            setter(dict(self.provenance))

    def _acquire_locks(self) -> None:
        """按**固定顺序**获取全部设备锁；任一失败即释放本次已取得的。

        顺序由锁键排序决定：以不同顺序请求同一组设备才会互相等待，固定顺序是防死锁的前提。
        回滚保证不会留下「拿了一半」的状态。
        """
        ordered = sorted(self.locks, key=lambda item: str(getattr(item, "key", "")))
        acquired: list = []
        try:
            for lock in ordered:
                lock.acquire()
                acquired.append(lock)
        except BaseException:
            for lock in reversed(acquired):
                try:
                    lock.release()
                except BaseException as exc:
                    self.cleanup_errors.append(exc)
            raise
        self.locks = ordered

    def _close_sink(self) -> None:
        closer = getattr(self.sink, "close", None)
        if closer is None:
            return
        try:
            closer()
        except BaseException as exc:
            self.cleanup_errors.append(exc)

    def _finalize_outputs(self, payload: dict) -> None:
        """落盘会话摘要与逐命令记录。

        落盘失败会把本轮结论**降级为失败**：记录存不下来本身就意味着这轮结果不可依赖，
        不能还对外报 ``completed=True``。降级后的结论再尝试保存一次；连失败记录都存不下时，
        在返回值里显式标出该事实并附错误，交由调用方处置——绝不静默留在「完成」状态。
        """
        try:
            self._write_json("session_summary.json", payload)
            if self.commands:
                self._write_ndjson("commands.ndjson", self.commands)
            return
        except BaseException as exc:
            self.cleanup_errors.append(exc)
            self.output_error = f"会话记录落盘失败：{exc!r}"

        payload["completed"] = False
        payload["verdict"] = "OUTPUT_WRITE_FAILED"
        payload["output_write_error"] = self.output_error
        payload["cleanup_errors"] = [repr(item) for item in self.cleanup_errors]
        try:
            self._write_json("session_summary.json", payload)
        except BaseException as exc:
            self.cleanup_errors.append(exc)
            payload["cleanup_errors"] = [repr(item) for item in self.cleanup_errors]
            payload["failure_record_saved"] = False
            payload["failure_record_error"] = f"失败记录也未能保存：{exc!r}"
        else:
            payload["failure_record_saved"] = True

    def _record(self, event: str) -> None:
        self.events.append(event)

    def _record_command(self, command_id: str, request: dict, *,
                        sent_monotonic: float | None, readback_monotonic: float | None,
                        ok: bool, detail: str | None = None) -> None:
        """逐命令记录：请求内容、发送时间、回读时间与结果。

        与逐帧事实分开保存——命令是「我们让设备做什么」，帧是「设备实际给出什么」，
        两者混在一起就无法判断是命令没生效还是取帧有问题。
        """
        self.commands.append({
            "command_id": command_id,
            "request": dict(request),
            "sent_monotonic": None if sent_monotonic is None else float(sent_monotonic),
            "readback_monotonic": None if readback_monotonic is None else float(readback_monotonic),
            "ok": bool(ok),
            "detail": detail,
        })

    def _termination(self) -> dict:
        return self.plan["termination"]

    def _capture(self) -> dict:
        return self.plan["capture_plan"]

    def _connect_and_verify_initial_state(self) -> None:
        # 从这里开始才会向设备发命令；追溯失败时这个标志保持 False。
        self.hardware_touched = True
        borrow = getattr(self.pump, "borrow_device_lock", None)
        if callable(borrow):
            expected_key = str(getattr(getattr(self.pump, "client", None), "device_lock_key", ""))
            matching = [lock for lock in self.locks if str(getattr(lock, "key", "")) == expected_key]
            if len(matching) != 1:
                raise CaptureLifecycleError(
                    f"泵设备锁未唯一匹配：expect={expected_key!r}, matches={len(matching)}")
            borrow(matching[0])
        sent = self._clock()
        state = self.pump.connect_and_probe()
        established = bool(getattr(state, "comm_established", False))
        self._record_command(
            "connect-and-probe", {"channel": "all"}, sent_monotonic=sent,
            readback_monotonic=self._clock(), ok=established,
            detail=None if established else str(getattr(state, "failed", "unknown")))
        if not established:
            raise CaptureLifecycleError(
                f"泵连接或初始状态核验失败：{getattr(state, 'failed', 'unknown')}")
        self._record("pump_connected")

    def _configure_camera_and_read_back(self) -> None:
        self.hardware_touched = True
        devices = self.camera.discover_devices()
        expected_unique_id = str(self.plan["apparatus"]["camera_unique_id"])
        matching = [
            item for item in devices
            if str(getattr(item, "unique_id", "")) == expected_unique_id
            and bool(getattr(item, "available", True))
        ]
        device = matching[0] if len(matching) == 1 else None
        if device is None:
            raise CameraSetupError(
                f"计划相机 {expected_unique_id!r} 未唯一匹配可用设备（匹配 {len(matching)}，发现 {len(devices)}）")
        self.camera.open(device)
        self._record("camera_open")
        self.camera.start_stream()
        self._record("camera_streaming")
        for name, key in (("exposure", "requested_exposure_us"), ("frame_rate", "requested_fps")):
            target = self._capture().get(key)
            if target is None:
                continue
            self._apply_feature_verified(name, float(target))
        self._prepare_live_sink()

    def _prepare_live_sink(self) -> None:
        prepare = getattr(self.sink, "prepare", None)
        if not callable(prepare):
            return
        deadline = self._clock() + max(2.0, float(self._capture()["max_no_valid_frame_s"]))
        expected_unique_id = str(self.plan["apparatus"]["camera_unique_id"])
        while self._clock() < deadline:
            frame = self.camera.read_frame(self._frame_timeout_ms())
            if not getattr(frame, "valid", False):
                continue
            if str(getattr(frame, "device_unique_id", "")) != expected_unique_id:
                raise CameraSetupError(
                    f"帧来源设备与计划不一致：expect={expected_unique_id!r}, "
                    f"actual={getattr(frame, 'device_unique_id', '')!r}")
            if int(getattr(frame, "hardware_frame_id", 0) or 0) <= 0:
                continue
            if float(getattr(frame, "host_monotonic_timestamp", 0.0) or 0.0) <= 0.0:
                continue
            prepare(frame)
            self._record("raw_archive_ready")
            return
        raise NoValidFrameError("启动泵前未取得可用于验证存储链的有效相机帧")

    def _apply_feature_verified(self, name: str, target: float, tolerance_ratio: float = 0.05) -> None:
        """设置后必须回读确认。

        适配器对未知特征名是静默 no-op（没有 else 分支），所以只调用不比对等于没配。

        NaN 与 Inf 必须在容差比较**之前**拒绝：``abs(nan - target) > tolerance`` 恒为 False，
        会让不一致检查形同失效。
        """
        if not math.isfinite(float(target)):
            raise CameraSetupError(f"相机 {name} 的设定值不是有限数：{target!r}")
        self.camera.set_feature(name, target)
        readback = self.camera.get_feature(name)
        if readback is None:
            readback = self._feature_readback_from_frames(name, target)
        try:
            actual = float(readback)
        except (TypeError, ValueError) as exc:
            raise CameraSetupError(f"相机 {name} 回读值不可解释：{readback!r}") from exc
        if not math.isfinite(actual):
            raise CameraSetupError(f"相机 {name} 回读值不是有限数：{readback!r}")
        if name == "frame_rate":
            tolerance_ratio = max(tolerance_ratio, 0.10)
        tolerance = max(abs(target) * tolerance_ratio, 1e-9)
        if abs(actual - target) > tolerance:
            raise CameraSetupError(
                f"相机 {name} 回读与设定不符：设定 {target!r}，回读 {actual!r}，容差 {tolerance!r}")
        self._record(f"camera_{name}_verified")

    def _feature_readback_from_frames(self, name: str, target: float) -> float:
        """Use direct-frame facts when the MVS feature getter is unavailable."""
        frames = []
        deadline = self._clock() + max(2.0, float(self._capture()["max_no_valid_frame_s"]))
        while self._clock() < deadline and len(frames) < 12:
            frame = self.camera.read_frame(self._frame_timeout_ms())
            if getattr(frame, "valid", False):
                frames.append(frame)
                if name == "exposure" and getattr(frame, "exposure_time_us", None) is not None:
                    return float(frame.exposure_time_us)
        if name == "frame_rate" and len(frames) >= 4:
            moments = [float(getattr(frame, "host_monotonic_timestamp", 0.0) or 0.0)
                       for frame in frames]
            if all(value > 0.0 for value in moments):
                intervals = np.diff(np.asarray(moments, dtype=float))
                if np.all(intervals > 0.0):
                    return float(1.0 / np.median(intervals))
        raise CameraSetupError(
            f"相机 {name} 设置后既无法特征回读，也无法从有效硬件帧验证")

    def _begin_capture_and_save_baseline(self) -> None:
        """保存指令前基线。

        读不到设备状态时如实记为不可用并附原因，而不是编造数字——但基线**必须落盘**，
        因此写盘失败会向上抛出。
        """
        self.hardware_touched = True
        try:
            q1, q2 = self.pump.get_current_q_state()
        except Exception as exc:
            channels = {"available": False, "reason": repr(exc)}
        else:
            channels = {"available": True, "q1": q1, "q2": q2}
        self.baseline = {"captured_monotonic": float(self._clock()), "channels": channels}
        self._write_json("pre_command_baseline.json", self.baseline)
        self._record("baseline_saved")

    def _write_pump_parameters(self) -> None:
        flow = self.plan["flow_plan"]
        q1_target = float(flow["baseline_q1_ul_min"])
        q2_target = float(flow["baseline_q2_ul_min"])
        if q2_target <= 0.0 or q1_target <= 2.0 * q2_target:
            raise ChannelWriteError(
                f"拒绝写泵：目标必须满足 Q2>0 且 Q1>2×Q2；q1={q1_target}, q2={q2_target}")

        current = (self.baseline or {}).get("channels", {})
        if not current.get("available"):
            raise ChannelWriteError("拒绝写泵：写入前无法取得当前 Q1/Q2，不能验证中间状态")
        q1_current = float(current["q1"])
        q2_current = float(current["q2"])
        if q2_current <= 0.0 or q1_current <= 2.0 * q2_current:
            raise ChannelWriteError(
                f"拒绝写泵：当前状态不满足 Q1>2×Q2；q1={q1_current}, q2={q2_current}")

        encoded = {
            1: self.pump.channel_params_for_flow(1, q1_target),
            2: self.pump.channel_params_for_flow(2, q2_target),
        }
        encoded_q1 = self.pump.flow_from_channel_params(encoded[1])
        encoded_q2 = self.pump.flow_from_channel_params(encoded[2])
        if encoded_q1 is None or encoded_q2 is None or encoded_q1 <= 2.0 * encoded_q2:
            raise ChannelWriteError(
                f"拒绝写泵：量化后的目标不能保持 Q1>2×Q2；q1={encoded_q1}, q2={encoded_q2}")

        if encoded_q1 > 2.0 * q2_current:
            order = ((1, encoded[1], encoded_q1), (2, encoded[2], encoded_q2))
        elif q1_current > 2.0 * encoded_q2:
            order = ((2, encoded[2], encoded_q2), (1, encoded[1], encoded_q1))
        else:
            raise ChannelWriteError(
                "拒绝写泵：无论先写 CH1 还是 CH2，中间组合都不能保持 Q1>2×Q2")

        for channel, params, target in order:
            sent = self._clock()
            written = self.pump.write_wsp_and_verify(channel, params)
            ok = bool(getattr(written, "ok", False))
            detail = None if ok else (getattr(written, "reason", None)
                                      or getattr(written, "error", None))
            self._record_command(
                f"wsp-ch{channel}", {"channel": channel, "target_ul_min": target},
                sent_monotonic=sent, readback_monotonic=self._clock(), ok=ok, detail=detail)
            if not ok:
                raise ChannelWriteError(f"CH{channel} 写入未通过回读验证：{detail}")
            self._record(f"ch{channel}_written_verified")

    def _start_infusion(self) -> None:
        # 启动指令即将发出：先置「泵可能运行」。若该调用抛异常或回读不通过，
        # 泵也可能已经在转，因此清理路径必须无条件尝试停泵。
        self.pump_may_be_running = True
        sent = self._clock()
        try:
            result = self.pump.start_infusion_and_verify([1, 2])
        except BaseException as exc:
            self._record_command("start-infusion", {"channels": [1, 2]}, sent_monotonic=sent,
                                 readback_monotonic=self._clock(), ok=False,
                                 detail=f"抛出异常：{exc!r}")
            raise
        ok = bool(getattr(result, "ok", False))
        detail = None if ok else (getattr(result, "reason", None) or getattr(result, "error", None))
        self._record_command("start-infusion", {"channels": [1, 2]}, sent_monotonic=sent,
                             readback_monotonic=self._clock(), ok=ok, detail=detail)
        if not ok:
            raise PumpStartUnverifiedError(f"启动指令已发出但回读未确认：{detail}")
        self.started = True
        self._record("pump_start_verified")

    def _frame_timeout_ms(self) -> int:
        return max(50, int(float(self._capture()["max_no_valid_frame_s"]) * 1000.0))

    def _run_bounded(self) -> None:
        """有界运行：会话时长、无有效帧、存储积压三项都有明确上限。"""
        capture = self._capture()
        deadline = self._clock() + float(self._termination()["max_session_s"])
        no_valid_limit = float(capture["max_no_valid_frame_s"])
        last_valid = self._clock()
        while self._clock() < deadline:
            frame = self.camera.read_frame(self._frame_timeout_ms())
            now = self._clock()
            if not getattr(frame, "valid", False):
                if now - last_valid > no_valid_limit:
                    raise NoValidFrameError(
                        f"连续 {no_valid_limit}s 未取到有效帧（上限 {capture['max_no_valid_frame_s']}），有界结束")
                continue
            last_valid = now
            self.frames_seen += 1
            self.sink.submit(frame)
            self.frames_accepted += 1
        if self.frames_accepted == 0:
            raise NoValidFrameError("整个运行窗口内没有取到任何有效帧")
        self.run_finished = True
        self._record("bounded_run_finished")

    # ------------------------------------------------------------ 清理

    def _cleanup(self) -> None:
        if self.pump_may_be_running:
            try:
                self.stop_outcome = self._stop_pump_with_retries()
            except BaseException as exc:  # 清理异常单独保存，不覆盖原始异常
                self.cleanup_errors.append(exc)
        try:
            self._teardown_camera()
        except BaseException as exc:
            self.cleanup_errors.append(exc)
        try:
            self._disconnect_pump()
        except BaseException as exc:
            self.cleanup_errors.append(exc)
        try:
            for lock in reversed(self.locks):
                lock.release()
            self._record("device_lock_released")
        except BaseException as exc:
            self.cleanup_errors.append(exc)

    def _stop_pump_with_retries(self) -> StopOutcome:
        """停泵与有限重试**全部在设备锁内**完成。"""
        budget = int(self._termination()["max_stop_retries"]) + 1
        last_error: str | None = None
        attempts = 0
        for attempt in range(1, budget + 1):
            attempts = attempt
            sent = self._clock()
            try:
                result = self.pump.stop_system_and_verify()
            except BaseException as exc:
                last_error = f"停泵调用抛出异常：{exc!r}"
                self._record_command(f"stop-attempt-{attempt}", {"channel": "all"},
                                     sent_monotonic=sent, readback_monotonic=self._clock(),
                                     ok=False, detail=last_error)
            else:
                ok = bool(getattr(result, "ok", False))
                detail = None if ok else (getattr(result, "reason", None)
                                          or getattr(result, "error", None)
                                          or "停泵回读未确认")
                self._record_command(f"stop-attempt-{attempt}", {"channel": "all"},
                                     sent_monotonic=sent, readback_monotonic=self._clock(),
                                     ok=ok, detail=detail)
                if ok:
                    self._record(f"pump_stopped_verified_attempt_{attempt}")
                    return StopOutcome("STOPPED", attempts, True)
                last_error = detail
            self.log(f"[PUMP][STOP] 第 {attempt}/{budget} 次未确认：{last_error}")
        return StopOutcome("STOP_UNVERIFIED", attempts, False, last_error)

    def _teardown_camera(self) -> None:
        self.camera.stop_stream()
        self.camera.close()
        self._record("camera_closed")

    def _disconnect_pump(self) -> None:
        self.pump.disconnect()
        self._record("pump_disconnected")

    # ------------------------------------------------------------ 汇报

    def report(self) -> dict:
        stop = self.stop_outcome
        frame_summary = self.sink.summary() if hasattr(self.sink, "summary") else None
        premise_ok = not isinstance(frame_summary, dict) or bool(
            frame_summary.get("sampling_premise_ok", True))
        completed = (self.original_error is None and self.started and stop is not None
                     and stop.verified and premise_ok)
        if stop is not None and not stop.verified:
            # 停机未被确认是最需要被看见的状态，压过其它结论。
            verdict = "STOP_UNVERIFIED"
        elif self.original_error is not None:
            verdict = "FAILED"
        elif not premise_ok:
            verdict = "PREMISE_REJECTED"
        elif not completed:
            verdict = "INCOMPLETE"
        else:
            verdict = "CAPTURE_COMPLETE"
        # 三层结果分别表达：运行完成、测量验收、停止状态。入口只按**声明的任务目标**
        # 决定退出码，所以「停泵成功」不会被读成「实验通过」。
        run_completed = bool(self.run_finished and self.original_error is None)
        # 测量验收必须由 sink **显式声明**。缺声明一律判为「未验收」并写明原因——
        # 把「没有测量证据」默认成通过，正是把「采样完整」再次当成「测量验收」的老毛病。
        if not isinstance(frame_summary, dict):
            declared_acceptance = None
            acceptance_reason = "sink_provides_no_measurement_evidence"
        else:
            declared_acceptance = frame_summary.get("measurement_accepted")
            acceptance_reason = frame_summary.get("measurement_acceptance_reason")
        if declared_acceptance is None:
            measurement_accepted = False
            if acceptance_reason is None:
                acceptance_reason = "sink_declares_no_measurement_acceptance"
        else:
            # 测量验收以「运行完成」为前提：没跑完的运行没有可验收的测量。
            measurement_accepted = bool(run_completed and premise_ok
                                        and bool(declared_acceptance))
            if acceptance_reason is None and not measurement_accepted:
                acceptance_reason = ("run_not_completed" if not run_completed
                                     else ("sampling_premise_not_ok" if not premise_ok
                                           else "sink_declared_not_accepted"))
        stop_state = ("NOT_ATTEMPTED" if stop is None
                      else ("STOPPED" if stop.verified else "STOP_UNVERIFIED"))
        stop_verified = bool(stop is not None and stop.verified)
        if self.task_goal == TASK_GOAL_DIAGNOSTIC_CAPTURE:
            task_goal_met = bool(run_completed and stop_verified)
        else:
            task_goal_met = bool(run_completed and stop_verified and measurement_accepted)
        result_layers = {
            "run_completed": run_completed,
            "measurement_accepted": measurement_accepted,
            "measurement_acceptance_reason": acceptance_reason,
            "sampling_premise_ok": bool(premise_ok),
            "stop": {"state": stop_state, "verified": stop_verified},
            "task_goal": self.task_goal,
            "task_goal_met": task_goal_met,
        }
        payload: dict = {
            "verdict": verdict,
            "completed": completed,
            "result_layers": result_layers,
            "provenance": self.provenance,
            "touched_hardware": bool(self.hardware_touched),
            "plan": self.plan.get("session_id"),
            "events": list(self.events),
            "frames_seen": self.frames_seen,
            "frames_accepted": self.frames_accepted,
            "baseline": self.baseline,
            "commands": list(self.commands),
            "device_locks": [str(getattr(item, "key", "")) for item in self.locks],
            "frame_facts": frame_summary,
            "stop": None if stop is None else {
                "verdict": stop.verdict, "attempts": stop.attempts,
                "verified": stop.verified, "error": stop.error,
            },
            "original_error": None if self.original_error is None else repr(self.original_error),
            "cleanup_errors": [repr(exc) for exc in self.cleanup_errors],
            "requires_onsite_confirmation": bool(stop is not None and not stop.verified),
        }
        if self.original_error is not None:
            payload["failure_kind"] = type(self.original_error).__name__
        if payload["requires_onsite_confirmation"]:
            payload["onsite_instruction"] = (
                "停泵未被确认：请现场人员确认泵已停止，并检查管路与注射器，"
                "在确认之前禁止继续实验或自动重启")
        return payload

    def _write_json(self, name: str, payload: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_ndjson(self, name: str, rows: list) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class LosslessFrameFactsRecorder:
    """Persist full Mono8 frames losslessly together with one-to-one facts."""

    def __init__(self, *, directory: Path, requested_fps: float, max_backlog: int) -> None:
        self.directory = Path(directory)
        self.requested_fps = float(requested_fps)
        self.facts = FrameFactsRecorder(
            path=self.directory / "frames.ndjson",
            max_backlog=max_backlog,
            timeline_source=TIMELINE_HOST_MONOTONIC,
            timeline_assumed=False,
        )
        self._writer = None
        self._frame_size: tuple[int, int] | None = None
        self._is_color = False
        self.archived_frames = 0

    @property
    def accepted(self) -> int:
        return self.facts.accepted

    def prepare(self, frame: object) -> None:
        image = getattr(frame, "image", None)
        if image is None or not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise StorageBacklogError("首帧不是可归档的 uint8 图像，拒绝在启动泵后才发现存储不可用")
        height, width = image.shape[:2]
        self._is_color = bool(image.ndim == 3 and image.shape[2] >= 3)
        self._frame_size = (int(width), int(height))
        self.directory.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.directory / "raw_frames.mkv"),
            cv2.VideoWriter_fourcc(*"FFV1"),
            self.requested_fps,
            self._frame_size,
            self._is_color,
        )
        if not writer.isOpened():
            writer.release()
            raise StorageBacklogError("无法创建 FFV1 无损原始帧文件，泵保持停止")
        self._writer = writer

    def submit(self, frame: object) -> None:
        if self._writer is None:
            self.prepare(frame)
        image = getattr(frame, "image", None)
        if image is None or tuple(reversed(image.shape[:2])) != self._frame_size:
            raise StorageBacklogError("帧尺寸在采集中改变，拒绝写入不一致的原始记录")
        self._writer.write(image)
        self.archived_frames += 1
        self.facts.submit(frame)

    def close(self) -> None:
        try:
            self.facts.close()
        finally:
            if self._writer is not None:
                self._writer.release()
                self._writer = None

    def summary(self) -> dict:
        payload = self.facts.summary()
        payload.update({
            "raw_archive": str(self.directory / "raw_frames.mkv"),
            "raw_codec": "FFV1",
            "raw_frames": self.archived_frames,
        })
        return payload


@dataclass
class TransientTracker:
    """Measure the flow continuously from a sliding window of frames."""

    dt: float
    pixel_calibration: dict | None = None
    window: int = 40
    step: int = 10
    lock_frames: int = 120
    gaps: tuple = DEFAULT_GAPS
    # 独立速度约束：留空则混叠状态只能是 ALIAS_UNRESOLVED（任务书 §4）。
    max_displacement: float | None = None
    bound_source: str = ""
    band_rows: tuple | None = None
    channel_x0: int = 0
    pitch_px: float = 0.0
    pitch_strength: float = 0.0
    direction: float | None = None
    samples: list = field(default_factory=list)
    _buffer: list = field(default_factory=list)
    _timings: list = field(default_factory=list)
    _seen: int = 0

    def push(self, frame: np.ndarray, *, timing: FrameTiming | None = None) -> dict | None:
        """Add one frame; returns a sample when a measurement was produced.

        ``timing`` 是该帧的时间与身份信息。**必须显式说明用哪个时钟**：不提供
        ``host_received``／``wall_clock`` 就只能按设定帧率推算，样本会带
        ``dt_assumed=True`` 与 ``t_source=assumed_user_rate``。

        这里刻意不提供「随便传一个时间戳就当作采集时刻」的入口——那会把墙钟或设备 ticks
        误标成实测采集时钟。三种时钟由 ``FrameTiming`` 分开记录。
        """
        self._buffer.append(np.asarray(frame, np.float32))
        self._timings.append(timing if timing is not None else FrameTiming())
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
            self._timings.pop(0)
        self._seen += 1
        if self.band_rows is None:
            if len(self._buffer) < min(self.lock_frames, self.window):
                return None
            self._lock()
        if self._seen % self.step or len(self._buffer) < self.window:
            return None
        qualified, reason, dt, source = qualify_window(self._timings, nominal_dt=self.dt)
        return self.measure(t=self._window_moment(source), t_source=source,
                            t_assumed=source == TIMELINE_ASSUMED_USER_RATE,
                            dt=dt, timing_ok=qualified, timing_reason=reason)

    def _window_moment(self, source: str) -> float:
        """窗口的时间标签。不合格窗口只给一个占位标签，速度不会由它换算。"""
        last = self._timings[-1]
        if source == CLOCK_HOST_RECEIVED and last.host_received is not None:
            return float(last.host_received)
        if source == CLOCK_WALL_CLOCK and last.wall_clock is not None:
            return float(last.wall_clock)
        return (self._seen - self.window) * self.dt

    def timing_summary(self) -> dict:
        counts: dict = {}
        sources: dict = {}
        for row in self.samples:
            counts[bool(row.get("timing_ok"))] = counts.get(bool(row.get("timing_ok")), 0) + 1
            source = row.get("dt_source")
            sources[source] = sources.get(source, 0) + 1
        return {"qualified": counts.get(True, 0), "disqualified": counts.get(False, 0),
                "dt_sources": sources,
                "timeline_usable": counts.get(False, 0) == 0 and bool(self.samples)}

    def _lock(self) -> None:
        stack = np.stack(self._buffer)
        top, bot, _ = activity_band(stack)
        self.band_rows = (top, bot)
        self.channel_x0 = channel_start(stack, top, bot)
        profiles = channel_profiles(stack, top, bot, x_lo=self.channel_x0)
        self.pitch_px, self.pitch_strength = pitch_from_autocorrelation(profiles)

    def measure(self, t: float, t_source: str = TIMELINE_ASSUMED_USER_RATE,
                t_assumed: bool = True, *, dt: float | None = None,
                timing_ok: bool = True, timing_reason: str = "") -> dict:
        stack = np.stack(self._buffer)
        top, bot = self.band_rows
        profiles = channel_profiles(stack, top, bot, x_lo=self.channel_x0)
        if self.pitch_px <= 0:
            self.pitch_px, self.pitch_strength = pitch_from_autocorrelation(profiles)
        # 速度换算一律用本窗口通过资格检查的**实测间隔**；不合格窗口不产出 px_per_second。
        effective_dt = self.dt if dt is None else float(dt)
        usable_dt = bool(timing_ok and math.isfinite(effective_dt) and effective_dt > 0.0)
        free = estimate_velocity(profiles, effective_dt, pitch=self.pitch_px, gaps=self.gaps)
        # 流向只从本段录像推断，用于把候选位移对准同一支；**不作为**解除混叠的独立证据。
        if self.direction is None and free.ok and abs(free.px_per_frame) > 1.0:
            self.direction = -1.0 if free.px_per_frame < 0 else 1.0
        if self.direction is not None:
            chosen = estimate_velocity(profiles, effective_dt, pitch=self.pitch_px, gaps=self.gaps,
                                       direction=self.direction)
            if not chosen.ok:
                chosen = free
        else:
            chosen = free
        screened = self._screen(free)
        physical = (self._physical(screened, effective_dt) if usable_dt else {
            "mm_per_second": None,
            "reason": f"时间轴不合格，不换算速度：{timing_reason or '未建立时间轴'}"})
        sample = {
            "t_s": round(float(t), 4),
            "t_source": t_source,
            "t_assumed": bool(t_assumed),
            # —— 本窗口的时间轴：来源、是否假定、是否通过资格检查、实测间隔 ——
            "dt_s": effective_dt,
            "dt_source": t_source,
            "dt_assumed": bool(t_assumed),
            "timing_ok": bool(timing_ok),
            "timing_reason": timing_reason,
            # —— 候选像素位移：纯像素域，不含任何标定换算 ——
            "px_per_frame": round(float(chosen.px_per_frame), 4),
            "px_per_second": (round(float(chosen.px_per_second), 4) if usable_dt else None),
            "ok": bool(chosen.ok),
            "confidence": chosen.confidence,
            "residual_px": float(chosen.residual_px),
            "alias_margin": float(chosen.alias_margin),
            "agreeing_gaps": list(chosen.agreeing_gaps),
            # —— 混叠状态与候选分支（复用 offline_analysis 的筛查） ——
            "aliasing": screened,
            # —— 条件选定的速度：只有条件解除时才有值 ——
            "selected_px_per_frame": screened.get("selected_px_per_frame"),
            # —— 物理量：需要有效标定且混叠已条件解除 ——
            "mm_per_second": physical["mm_per_second"],
            "mm_per_second_reason": physical["reason"],
            "pitch_px": round(float(self.pitch_px), 2),
            "band_rows": list(self.band_rows),
            "channel_x0": int(self.channel_x0),
            "direction": None if self.direction is None else float(self.direction),
        }
        self.samples.append(sample)
        return sample

    def _screen(self, estimate) -> dict:
        """复用 ``offline_analysis`` 的混叠筛查。

        ``direction`` 一律传 ``None``：本模块的流向是从**同一段录像**推断的，任务书 §4
        明确「从同一录像推断的流向，不算独立解除混叠的证据」，因此不能拿它筛掉候选分支。
        """
        if self.pitch_px <= 0:
            return {
                "status": "REJECTED", "reason": "缺少空间周期，无法筛查混叠",
                "candidates": [], "candidates_exhaustive": False,
                "selected_px_per_frame": None, "control_authorized": False,
                "independent_bound_px_per_frame": self.max_displacement,
                "bound_source": self.bound_source,
            }
        return screen_velocity(estimate, self.pitch_px, direction=None,
                               max_displacement=self.max_displacement,
                               bound_source=self.bound_source)

    def _physical(self, screened: dict, dt: float) -> dict:
        """物理速度只在「混叠已条件解除」「标定通过校验」「时间间隔可用」时才给出。

        ``dt`` 必须是**本窗口实测**的间隔（或离线明示假定的间隔），不再用设定帧率。
        """
        if screened.get("status") != "CONDITIONAL":
            return {"mm_per_second": None,
                    "reason": f"混叠状态为 {screened.get('status')}，未条件解除，不输出物理速度"}
        selected = screened.get("selected_px_per_frame")
        if selected is None:
            return {"mm_per_second": None, "reason": "没有条件选定的像素位移"}
        calibration = self.pixel_calibration or {}
        if not calibration.get("validated"):
            return {"mm_per_second": None,
                    "reason": "缺少通过校验的像素标定记录（命令行 --scale 不算有效标定），不输出物理速度"}
        if not (math.isfinite(float(dt)) and float(dt) > 0.0):
            return {"mm_per_second": None, "reason": "时间间隔不可用，不输出物理速度"}
        speed = float(selected) / float(dt) * float(calibration["um_per_px"]) / 1000.0
        return {"mm_per_second": round(speed, 5), "reason": None}

    def usable(self) -> list:
        return [row for row in self.samples if row["ok"]]


def steady_state_time(samples: list, slope_tolerance: float = 0.05, dwell_s: float = 60.0,
                      *, value_key: str = "px_per_frame") -> float | None:
    """从某一时刻起，整段 ``dwell_s`` 内取值都满足斜率容差的最早时刻。

    两条硬要求（任务书 §4）：

    * 窗口必须**覆盖完整的要求时长**——不得用 ``0.8 × dwell`` 之类的折扣顶替；
    * 窗口内**不得出现无效样本**。先滤掉无效样本再把剩下的当成连续有效窗口，
      会把「中间断了一段」伪装成一段连续平台。

    采样间隔带来的不到一个周期的端点缺口用 ``cadence`` 补偿（这是采样事实，不是折扣）。
    """
    if len(samples) < 3:
        return None
    times = np.array([float(row["t_s"]) for row in samples], dtype=np.float64)
    if not np.isfinite(times).all() or times[-1] < times[0] + dwell_s:
        return None
    cadence = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    for start in times:
        if start + dwell_s > times[-1]:
            break
        inside = (times >= start) & (times <= start + dwell_s)
        window = [row for row, flag in zip(samples, inside) if flag]
        if len(window) < 3:
            continue
        if any((not row.get("ok")) or row.get(value_key) is None for row in window):
            continue
        if (times[inside].max() - times[inside].min()) < dwell_s - cadence:
            continue
        values = np.array([float(row[value_key]) for row in window], dtype=np.float64)
        if not np.isfinite(values).all():
            continue
        slope = float(np.polyfit(times[inside], values, 1)[0])
        if abs(slope) <= slope_tolerance:
            return float(start)
    return None


def _displacement_section(rows: list) -> dict:
    """候选像素位移——纯像素域，不因标定或混叠状态而变成物理速度。"""
    if not rows:
        return {"available": False, "reason": "没有可用样本"}
    times = np.array([row["t_s"] for row in rows], dtype=np.float64)
    values = np.array([row["px_per_frame"] for row in rows], dtype=np.float64)
    slope = float(np.polyfit(times, values, 1)[0]) if len(rows) > 2 else 0.0
    return {
        "available": True,
        "unit": "px_per_frame",
        "first": float(values[0]),
        "last": float(values[-1]),
        "overall_slope_per_s": round(slope, 6),
        "max_abs_residual_px": float(max(row["residual_px"] for row in rows)),
        "max_abs_alias_margin": float(max(row["alias_margin"] for row in rows)),
        "note": "候选位移是像素域量；它不等于已确定的真实速度",
    }


def _aliasing_section(samples: list) -> dict:
    counts: dict = {}
    latest: dict = {}
    for row in samples:
        screened = row.get("aliasing") or {}
        status = screened.get("status", "REJECTED")
        counts[status] = counts.get(status, 0) + 1
        if screened.get("candidates"):
            latest = screened
    if counts.get("CONDITIONAL"):
        status = "CONDITIONAL"
    elif counts.get("ALIAS_UNRESOLVED"):
        status = "ALIAS_UNRESOLVED"
    elif counts:
        status = "REJECTED"
    else:
        status = "NO_SAMPLES"
    return {
        "status": status,
        "status_counts": counts,
        "candidates_px_per_frame": latest.get("candidates", []),
        "candidates_exhaustive": bool(latest.get("candidates_exhaustive", False)),
        "alias_period_px_per_frame": latest.get("alias_period_px_per_frame"),
        "reason": latest.get("reason"),
        "control_authorized": False,
        "note": "没有独立速度约束时状态只能是 ALIAS_UNRESOLVED，不可能为 CONDITIONAL",
    }


def _stability_section(samples: list, slope_tolerance: float, dwell_s: float,
                       aliasing_status: str, calibration_validated: bool,
                       timeline_usable: bool = True) -> dict:
    pixel_plateau = (steady_state_time(samples, slope_tolerance, dwell_s,
                                       value_key="px_per_frame")
                     if timeline_usable else None)
    physical_available = any(row.get("mm_per_second") is not None for row in samples)
    physical_plateau = (steady_state_time(samples, slope_tolerance, dwell_s,
                                          value_key="mm_per_second")
                        if (physical_available and timeline_usable) else None)
    note = ("像素域诊断保留、失败窗口不丢弃；物理域结论需要独立约束解除混叠、标定通过校验，"
            "且窗口覆盖完整驻留时长")
    if not timeline_usable:
        note += "；本轮时间轴未通过资格检查，因此不给任何稳定平台"
    return {
        "window_s": dwell_s,
        "timeline_usable": bool(timeline_usable),
        "pixel_domain_plateau_s": pixel_plateau,
        "pixel_domain_available": pixel_plateau is not None,
        "physical_plateau_s": physical_plateau,
        "physical_available": physical_available,
        "physical_plateau_available": physical_plateau is not None,
        "physical_claim_allowed": bool(physical_plateau is not None
                                       and aliasing_status == "CONDITIONAL"
                                       and calibration_validated),
        "note": note,
    }


def _verdict(report: dict, sampling_premise: tuple[bool, str] | None,
             time_axis_assumed: bool) -> str:
    if report["usable"] == 0:
        return "NO_SIGNAL"
    if sampling_premise is not None and not sampling_premise[0]:
        return "PREMISE_REJECTED"
    if not report["timing"]["timeline_usable"]:
        return "PREMISE_REJECTED"
    if report["aliasing"]["status"] != "CONDITIONAL":
        return "ALIAS_UNRESOLVED"
    if not report["pixel_calibration"].get("validated"):
        return "PIXEL_DOMAIN_ONLY"
    if not report["stability"]["physical_plateau_available"]:
        return "STILL_RISING"
    if time_axis_assumed:
        return "CONDITIONAL_STEADY"
    return "STEADY"


def analyse(samples: list, slope_tolerance_mm_s2: float = 0.05, dwell_s: float = 60.0,
            *, pixel_calibration: dict | None = None, bound_source: str = "",
            max_displacement: float | None = None,
            sampling_premise: tuple[bool, str] | None = None,
            time_axis_assumed: bool = False) -> dict:
    """把「采集事实」与「分析结果」分开汇报。

    速度输出按任务书 §4 拆成互不混淆的六段：候选像素位移、混叠状态与候选分支、
    独立约束及来源、条件选定速度、像素标定及来源、稳定性诊断。物理量只在混叠已条件
    解除**且**标定通过校验时出现，否则给 null 与原因，不以「ok」冒充实测真值。
    """
    rows = [row for row in samples if row.get("ok")]
    calibration = dict(pixel_calibration or {
        "um_per_px": None, "validated": False, "source": None,
        "reason": "未提供像素标定记录",
    })
    report: dict = {
        "samples": len(samples),
        "usable": len(rows),
        "dwell_s": dwell_s,
        "slope_tolerance": slope_tolerance_mm_s2,
    }
    report["displacement"] = _displacement_section(rows)
    report["aliasing"] = _aliasing_section(samples)
    report["independent_constraint"] = {
        "max_displacement_px_per_frame": max_displacement,
        "source": bound_source or None,
        "provided": bool(max_displacement is not None and bound_source.strip()),
        "note": None if bound_source.strip() else "从同一录像推断的流向不作为独立约束",
    }
    selected = [row["selected_px_per_frame"] for row in rows
                if row.get("selected_px_per_frame") is not None]
    report["selected_velocity"] = {
        "available": bool(selected),
        "sample_count": len(selected),
        "px_per_frame": None if not selected else float(np.median(selected)),
        "conditional": bool(selected),
        "control_authorized": False,
        "note": "选定速度只在独立约束下唯一时才存在；它不授权控制",
    }
    report["pixel_calibration"] = calibration
    disqualified = sum(1 for row in samples if row.get("timing_ok") is False)
    dt_sources: dict = {}
    for row in samples:
        key = row.get("dt_source")
        dt_sources[key] = dt_sources.get(key, 0) + 1
    timeline_usable = bool(samples) and disqualified == 0
    report["timing"] = {
        "timeline_usable": timeline_usable,
        "disqualified_windows": disqualified,
        "dt_sources": dt_sources,
        "note": ("速度按各窗口通过资格检查的实测间隔换算；不合格窗口不产出 px_per_second，"
                 "因此不会参与任何稳定结论"),
    }
    report["stability"] = _stability_section(samples, slope_tolerance_mm_s2, dwell_s,
                                             report["aliasing"]["status"],
                                             bool(calibration.get("validated")),
                                             timeline_usable)
    report["sampling_premise"] = (
        None if sampling_premise is None
        else {"ok": bool(sampling_premise[0]), "reason": sampling_premise[1]})
    report["time_axis_assumed"] = bool(time_axis_assumed)
    report["verdict"] = _verdict(report, sampling_premise, time_axis_assumed)
    return report


def write_outputs(directory: Path, tracker: TransientTracker, report: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"analysis": report, "samples": tracker.samples,
               "pitch_px": tracker.pitch_px, "pitch_strength": tracker.pitch_strength,
               "band_rows": list(tracker.band_rows or ()), "channel_x0": tracker.channel_x0,
               "pixel_calibration": tracker.pixel_calibration, "dt_s": tracker.dt}
    (directory / "series.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    with (directory / "series.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_s", "t_source", "t_assumed", "px_per_frame",
                         "selected_px_per_frame", "mm_per_second", "aliasing_status", "ok",
                         "confidence", "residual_px", "alias_margin"])
        for row in tracker.samples:
            writer.writerow([
                row["t_s"], row["t_source"], row["t_assumed"], row["px_per_frame"],
                "" if row["selected_px_per_frame"] is None else row["selected_px_per_frame"],
                "" if row["mm_per_second"] is None else row["mm_per_second"],
                (row.get("aliasing") or {}).get("status", ""),
                row["ok"], row["confidence"], row["residual_px"], row["alias_margin"],
            ])


def plot(directory: Path, tracker: TransientTracker, report: dict) -> None:
    rows = [row for row in tracker.samples if row["ok"]]
    if not rows:
        return
    width, height, margin = 1100, 560, 70
    image = np.full((height, width, 3), 255, np.uint8)
    times = np.array([row["t_s"] for row in rows], dtype=np.float64)
    values = np.array([row["px_per_frame"] for row in rows], dtype=np.float64)
    t_max = max(1e-6, float(times.max()))
    v_min, v_max = float(values.min()), float(values.max())
    span = max(1e-6, v_max - v_min)
    def project(x, y):
        return (int(margin + x / t_max * (width - 2 * margin)),
                int(height - margin - (y - v_min) / span * (height - 2 * margin)))
    previous = None
    for x, y in zip(times, values):
        point = project(x, y)
        if previous is not None:
            cv2.line(image, previous, point, (40, 120, 20), 2, cv2.LINE_AA)
        previous = point
    plateau = report.get("stability", {}).get("pixel_domain_plateau_s")
    if plateau is not None:
        line = project(float(plateau), v_min)
        cv2.line(image, (line[0], margin), (line[0], height - margin), (200, 120, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"steady after {plateau:.1f} s", (line[0] + 6, margin + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 90, 0), 1, cv2.LINE_AA)
    cv2.line(image, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)
    cv2.line(image, (margin, margin), (margin, height - margin), (0, 0, 0), 2)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        level = v_min + fraction * span
        _, y_pixel = project(0.0, level)
        cv2.line(image, (margin - 6, y_pixel), (margin, y_pixel), (0, 0, 0), 1)
        cv2.putText(image, f"{level:8.2f}", (6, y_pixel + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
        moment = fraction * t_max
        x_pixel, _ = project(moment, v_min)
        cv2.putText(image, f"{moment:.1f}", (x_pixel - 14, height - margin + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "candidate droplet displacement, pixel domain (no saved ROI)",
                (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    aliasing = report.get("aliasing", {}).get("status", "?")
    timeline = "假定时间轴" if report.get("time_axis_assumed") else "实测时间戳"
    cv2.putText(image, f"pitch {tracker.pitch_px:.0f} px   band {tracker.band_rows}   "
                       f"verdict {report.get('verdict')}   aliasing {aliasing}   {timeline}",
                (margin, height - margin + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(image, "time (s)", (width // 2 - 30, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1)
    cv2.putText(image, "px/frame", (8, margin - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(directory / "transient.png"), image)


def run_replay(path: Path, rate: float, directory: Path, window: int, step: int,
               slope_tolerance: float = 0.05, dwell: float = 60.0, *,
               pixel_calibration: dict | None = None,
               max_displacement: float | None = None, bound_source: str = "") -> dict:
    """离线回放：分析路径与实机一致，但时间轴是按采样率**假定**的。

    假定时间轴不阻止稳定判定，但结论降级为条件结论且不授权控制（任务书 §4 的取舍）。
    """
    stack = np.load(path, mmap_mode="r")
    tracker = TransientTracker(dt=1.0 / rate, pixel_calibration=pixel_calibration,
                               window=window, step=step,
                               max_displacement=max_displacement, bound_source=bound_source)
    for index in range(stack.shape[0]):
        tracker.push(np.asarray(stack[index], np.float32))
    report = analyse(tracker.samples, slope_tolerance, dwell,
                     pixel_calibration=pixel_calibration,
                     max_displacement=max_displacement, bound_source=bound_source,
                     time_axis_assumed=True)
    report["source"] = str(path)
    report["frames"] = int(stack.shape[0])
    report["time_axis"] = {
        "source": TIMELINE_ASSUMED_USER_RATE,
        "assumed": True,
        "detail": (f"离线 stack 不含逐帧时间戳，时间轴按 --rate={rate} 推算；"
                   "因此该轮结论是条件结论，不能当作实测时序，也不授权控制"),
    }
    write_outputs(directory, tracker, report)
    plot(directory, tracker, report)
    return report


def run_live(args, directory: Path) -> dict:
    """实机采集入口。

    先校验计划；只有全部通过后才延迟导入硬件模块、构造精确设备锁并运行会话。
    """
    if not args.plan:
        return {"verdict": "LIVE_BLOCKED", "touched_hardware": False,
                "unmet": ["--live 必须提供 --plan（填写完毕的会话计划）"]}
    try:
        plan = load_session_plan(Path(args.plan))
    except (OSError, ValueError) as exc:
        return {"verdict": "LIVE_BLOCKED", "touched_hardware": False,
                "unmet": [f"会话计划无法读取或解析：{exc}"]}

    unmet = validate_session_plan(plan)
    unmet += live_acceptance_gaps()
    if unmet:
        return {"verdict": "LIVE_BLOCKED", "touched_hardware": False,
                "plan": str(args.plan), "unmet": unmet}

    configured_output = Path(str(plan["capture_plan"]["output_directory"])).resolve()
    if configured_output != Path(directory).resolve():
        return {"verdict": "LIVE_BLOCKED", "touched_hardware": False,
                "plan": str(args.plan),
                "unmet": [
                    f"计划输出目录与实际 --out 不一致：plan={configured_output}, actual={Path(directory).resolve()}"
                ]}

    if bool(getattr(args, "preflight_only", False)):
        return {"verdict": "LIVE_READY", "touched_hardware": False,
                "plan": str(args.plan), "unmet": []}

    required_free = plan["capture_plan"].get("required_free_space_bytes")
    if _finite(required_free) and float(required_free) > 0.0:
        import shutil

        directory.parent.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(directory.parent).free
        if available < int(required_free):
            return {"verdict": "LIVE_BLOCKED", "touched_hardware": False,
                    "plan": str(args.plan),
                    "unmet": [f"输出盘剩余空间不足：需要 {int(required_free)}，实际 {available}"]}

    # Imports stay below every non-hardware gate so a rejected plan cannot even
    # enumerate a device.
    from backend.device_lock import DeviceLock, camera_lock_key, normalize_port_key
    from backend.pump_hardware.config import SerialConfig
    from backend.pump_hardware.service import PumpHardwareService
    from backend.vision.cameras.adapters.hikrobot_camera import HikrobotCameraAdapter

    apparatus = plan["apparatus"]
    serial = SerialConfig(
        port=str(apparatus["pump_port"]).strip().upper(),
        address=int(apparatus["pump_address"]),
        baudrate=int(apparatus.get("pump_baudrate", 1200)),
        parity=str(apparatus.get("pump_parity", "E")).strip().upper(),
    )
    pump = PumpHardwareService(serial_config=serial, logger=lambda message: print(message, flush=True))
    camera = HikrobotCameraAdapter(logger=lambda message: print(message, flush=True))
    locks = [
        DeviceLock(normalize_port_key(serial.port)),
        DeviceLock(camera_lock_key(apparatus["camera_unique_id"])),
    ]
    sink = LosslessFrameFactsRecorder(
        directory=Path(directory),
        requested_fps=float(plan["capture_plan"]["requested_fps"]),
        max_backlog=int(plan["capture_plan"]["max_backlog_frames"]),
    )
    session = LiveCaptureSession(
        plan=plan,
        pump=pump,
        camera=camera,
        locks=locks,
        sink=sink,
        output_dir=Path(directory),
        log=lambda message: print(message, flush=True),
        provenance={
            "directory": Path(directory) / "provenance",
            "effective": live_effective_config(
                plan=plan,
                strict_localization=declared(
                    True, reason="实时源：prepare_video 对 camera/realtime 来源声明严格定位"),
                wall_source="current_frame_localized"),
        },
    )
    return session.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and measure the pump start-up transient.",
        epilog="--replay 与 --live 必须显式二选一且互斥；不带参数只打印用法，不接触任何设备。",
    )
    parser.add_argument("--replay", default="", help="离线 .npy stack（显式选择离线模式）")
    parser.add_argument("--live", action="store_true", help="实机采集；必须同时提供 --plan")
    parser.add_argument("--plan", default="", help="填写完毕的会话计划 JSON（仅 --live 使用）")
    parser.add_argument("--preflight-only", action="store_true",
                        help="只校验实机计划和输出目录，不枚举或连接设备")
    parser.add_argument("--rate", type=float, default=0.0, help="帧率；必须显式给出，无默认兜底")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.0,
                        help="micron/pixel；仅作为未验证标尺记录，不能授权物理速度输出")
    parser.add_argument("--calibration", default="",
                        help="像素标定记录 JSON；只有通过校验的记录才授权物理速度输出")
    parser.add_argument("--max-displacement", type=float, default=None,
                        help="独立的最大像素位移约束（px/frame）；必须配 --bound-source")
    parser.add_argument("--bound-source", default="",
                        help="独立约束的来源说明；从同一录像推断的流向不算独立约束")
    parser.add_argument("--slope-tolerance", type=float, default=0.05, help="mm/s^2")
    parser.add_argument("--dwell", type=float, default=60.0, help="required plateau seconds")
    parser.add_argument("--out", default="")
    return parser


def _resolve_calibration(args, parser) -> dict | None:
    """把命令行给出的像素标尺解成标定元信息。

    只有通过校验的 ``CalibrationRecord`` 才标记 ``validated=True``；``--scale`` 只是
    未验证标尺，可用于像素域诊断，但不能授权物理速度输出（任务书 §4）。
    """
    if args.calibration:
        try:
            return load_pixel_calibration(Path(args.calibration))
        except Exception as exc:
            parser.error(f"--calibration 无法加载或校验：{exc}")
    if args.scale > 0.0:
        return unvalidated_scale(args.scale)
    return None


def _write_live_gate_report(directory: Path, report: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "live_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.replay and args.live:
        parser.error("--replay 与 --live 互斥：请只指定一个模式")
    if not args.replay and not args.live:
        parser.print_usage(sys.stderr)
        print("必须显式选择模式：--replay <stack.npy> 或 --live --plan <plan.json>；"
              "不带参数不会接触任何设备。", file=sys.stderr)
        return 2
    if args.plan and not args.live:
        parser.error("--plan 只能与 --live 一起使用")
    if args.preflight_only and not args.live:
        parser.error("--preflight-only 只能与 --live 一起使用")

    directory = Path(args.out) if args.out else REPO_ROOT / "output" / time.strftime("transient-%Y%m%d-%H%M%S")

    if args.replay:
        if args.rate <= 0.0:
            parser.error("--replay 需要显式 --rate（离线回放的采样率必须由使用者给出）")
        if args.max_displacement is not None and not args.bound_source.strip():
            parser.error("--max-displacement 必须配 --bound-source（独立约束必须给出来源）")
        if args.bound_source.strip() and args.max_displacement is None:
            parser.error("--bound-source 必须配 --max-displacement")
        report = run_replay(Path(args.replay), args.rate, directory, args.window, args.step,
                            args.slope_tolerance, args.dwell,
                            pixel_calibration=_resolve_calibration(args, parser),
                            max_displacement=args.max_displacement,
                            bound_source=args.bound_source)
    else:
        report = run_live(args, directory)
        _write_live_gate_report(directory, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("verdict") == "LIVE_BLOCKED":
        print("BLOCKED " + str(directory))
        return EXIT_LIVE_BLOCKED
    print("DONE " + str(directory))
    if "result_layers" in report:
        # 一次采集会话：退出码由三层结果决定，不从 verdict 反推。
        return exit_code(report)
    return 0 if report.get("verdict") in {
        "STEADY", "STILL_RISING", "CAPTURE_COMPLETE", "LIVE_READY"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
