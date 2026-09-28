"""采集脚本入口与会话计划校验的回归测试。

对应复查任务书 §1：实机入口必须先被封印。这里的核心不是「函数返回了错误码」，而是
**证明没有扫描、连接或修改设备**——所以子进程里先把设备入口（泵的连接/写入/启动/停机，
相机的发现/打开/取流/取帧）替换成会记名的哨兵，再跑一次 ``main()``，断言哨兵一次都没被触发。

``test_probe_sentinels_are_live`` 是这套做法的自检：若哪天哨兵替换失败（模块路径或函数名改了），
那些「没有被调用」的断言会变成空过，所以必须有一条测试证明哨兵确实拦得住。
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))

import plant_flow_transient_capture as tool  # noqa: E402
from test_plant_flow_transient_capture import ramp_stack  # noqa: E402

_PROBE = textwrap.dedent(
    """
    import importlib, json, sys

    sys.path.insert(0, sys.argv[1])   # repo root
    sys.path.insert(0, sys.argv[2])   # tools dir

    TARGETS = (
        ("backend.pump_hardware.service", "PumpHardwareService", "connect_and_probe"),
        ("backend.pump_hardware.service", "PumpHardwareService", "write_wsp_and_verify"),
        ("backend.pump_hardware.service", "PumpHardwareService", "start_infusion_and_verify"),
        ("backend.pump_hardware.service", "PumpHardwareService", "stop_system_and_verify"),
        ("backend.pump_hardware.service", "PumpHardwareService", "channel_params_for_flow"),
        ("backend.vision.cameras.adapters.hikrobot_camera", "HikrobotCameraAdapter", "discover_devices"),
        ("backend.vision.cameras.adapters.hikrobot_camera", "HikrobotCameraAdapter", "open"),
        ("backend.vision.cameras.adapters.hikrobot_camera", "HikrobotCameraAdapter", "start_stream"),
        ("backend.vision.cameras.adapters.hikrobot_camera", "HikrobotCameraAdapter", "set_feature"),
        ("backend.vision.cameras.adapters.hikrobot_camera", "HikrobotCameraAdapter", "read_frame"),
    )

    calls = []


    def _install():
        patched = []
        for module_name, class_name, attribute in TARGETS:
            try:
                owner = getattr(importlib.import_module(module_name), class_name)
            except Exception:
                continue

            def _make(label):
                def _sentinel(*_args, **_kwargs):
                    calls.append(label)
                    return None
                return _sentinel

            setattr(owner, attribute, _make(module_name + "." + class_name + "." + attribute))
            patched.append(class_name + "." + attribute)
        return patched


    patched = _install()
    mode = sys.argv[3]
    code = None

    if mode == "selfcheck":
        importlib.import_module("backend.pump_hardware.service").PumpHardwareService.connect_and_probe()
        importlib.import_module("backend.vision.cameras.adapters.hikrobot_camera").HikrobotCameraAdapter.discover_devices()
    else:
        import plant_flow_transient_capture as tool

        if mode == "main":
            try:
                code = tool.main(json.loads(sys.argv[4]))
            except SystemExit as exc:
                code = exc.code

    print(json.dumps({"code": code, "calls": calls, "patched": patched}))
    """
)


def _run_probe(mode: str, argv: list[str] | None = None) -> dict:
    """在子进程里装载哨兵并执行；回传退出码、被触发的设备入口、成功替换的哨兵。"""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(REPO_ROOT), str(TOOLS_DIR), mode, json.dumps(argv or [])],
        capture_output=True, encoding="utf-8", errors="replace", timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_probe_sentinels_are_live() -> None:
    """自检：真的去调两个设备入口，哨兵必须记到它们。

    没有这条，下面所有「未被调用」的断言在哨兵替换失效时会变成空过。
    """
    outcome = _run_probe("selfcheck")
    assert outcome["patched"], "没有任何设备入口被替换，探针失效"
    assert any("connect_and_probe" in item for item in outcome["calls"])
    assert any("discover_devices" in item for item in outcome["calls"])


# ---------------------------------------------------------------- 入口封印

def test_no_arguments_prints_usage_and_does_not_touch_hardware() -> None:
    """不带参数必须只打印用法退出——从前它会用默认 Q1/Q2 直接开泵跑 30 分钟。"""
    outcome = _run_probe("main", [])
    assert outcome["code"] == 2
    assert outcome["calls"] == []


def test_importing_the_tool_alone_does_not_call_device_entrypoints() -> None:
    outcome = _run_probe("import")
    assert outcome["calls"] == []


def test_live_without_plan_does_not_touch_hardware(tmp_path: Path) -> None:
    outcome = _run_probe("main", ["--live", "--out", str(tmp_path / "out")])
    assert outcome["code"] == 3
    assert outcome["calls"] == []


def test_live_with_blank_template_does_not_touch_hardware(tmp_path: Path) -> None:
    plan_path = tmp_path / "blank.json"
    plan_path.write_text(json.dumps(blank_template()), encoding="utf-8")
    outcome = _run_probe("main", ["--live", "--plan", str(plan_path), "--out", str(tmp_path / "out")])
    assert outcome["code"] == 3
    assert outcome["calls"] == []


def test_live_preflight_with_valid_plan_does_not_touch_hardware(tmp_path: Path) -> None:
    """完整计划的纯预检必须证明就绪且不碰任何设备入口。"""
    output = tmp_path / "out"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(valid_plan(output)), encoding="utf-8")
    outcome = _run_probe(
        "main", ["--live", "--preflight-only", "--plan", str(plan_path), "--out", str(output)]
    )
    assert outcome["code"] == 0
    assert outcome["calls"] == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--replay", "stack.npy", "--live", "--plan", "plan.json"],   # 模式互斥
        ["--plan", "plan.json"],                                     # --plan 缺 --live
        ["--replay", "stack.npy"],                                   # 缺 --rate
        ["--replay", "stack.npy", "--scale", "1.725"],               # 缺 --rate
        ["--replay", "stack.npy", "--rate", "320",
         "--max-displacement", "60"],                                # 独立约束缺来源
        ["--replay", "stack.npy", "--rate", "320",
         "--bound-source", "标定尺规实测上限"],                        # 来源缺独立约束
    ],
)
def test_usage_errors_exit_2_without_touching_hardware(argv: list[str]) -> None:
    outcome = _run_probe("main", argv)
    assert outcome["code"] == 2
    assert outcome["calls"] == []


def test_replay_still_works_without_any_hardware(tmp_path: Path) -> None:
    """封印实机入口不能把离线回放一起封掉。"""
    source = tmp_path / "stack.npy"
    np.save(source, ramp_stack(frames=200, ramp_frames=60))
    outcome = _run_probe("main", ["--replay", str(source), "--rate", "100", "--scale", "1.725",
                                  "--window", "20", "--step", "5", "--out", str(tmp_path / "out")])
    assert outcome["calls"] == []
    assert outcome["code"] in (0, 1)
    assert (tmp_path / "out" / "series.json").exists()


# ---------------------------------------------------------------- 计划校验

def test_blank_template_is_rejected_with_actionable_reasons() -> None:
    unmet = tool.validate_session_plan(blank_template())
    assert unmet
    joined = " | ".join(unmet)
    assert "空白模板" in joined
    assert "execution_enabled" in joined
    for fragment in ("设备/芯片标识", "基线 Q1 流量", "Q1 允许区间", "单步最大流量变化",
                     "各路累计输送上限", "运行时限", "无有效帧超时", "停机重试次数"):
        assert fragment in joined, fragment


def test_execution_enabled_alone_is_not_enough() -> None:
    """execution_enabled=true 不能替代完整校验。"""
    plan = blank_template()
    plan["execution_enabled"] = True
    assert tool.validate_session_plan(plan)


def test_valid_plan_passes_and_preflight_reports_ready(tmp_path: Path) -> None:
    output = tmp_path / "out"
    assert tool.validate_session_plan(valid_plan(output)) == []

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(valid_plan(output)), encoding="utf-8")
    report = tool.run_live(_args(plan_path, preflight_only=True), output)
    assert report["verdict"] == "LIVE_READY"
    assert report["touched_hardware"] is False
    assert report["unmet"] == []


def test_baseline_outside_allowed_range_is_rejected() -> None:
    plan = valid_plan()
    plan["flow_plan"]["baseline_q1_ul_min"] = 99.0
    assert "不在允许区间" in " | ".join(tool.validate_session_plan(plan))


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda p: p["flow_plan"].__setitem__("max_step_ul_min", 0.0), "单步最大流量变化 必须是正数"),
        (lambda p: p["flow_plan"].__setitem__("max_step_ul_min", -1.0), "单步最大流量变化 必须是正数"),
        (lambda p: p["termination"].__setitem__("max_session_s", 0.0), "运行时限 必须是正数"),
        (lambda p: p["termination"].__setitem__("stop_timeout_s", -1.0), "停泵超时 必须是正数"),
        (lambda p: p["termination"].__setitem__("max_stop_retries", -1), "停机重试次数必须是非负整数"),
        (lambda p: p["termination"].__setitem__("max_stop_retries", 2.5), "停机重试次数必须是非负整数"),
        (lambda p: p["capture_plan"].__setitem__("max_no_valid_frame_s", 0.0), "无有效帧超时必须为正数"),
        (lambda p: p["flow_plan"].__setitem__("q1_allowed_range_ul_min", [30.0, 10.0]),
         "Q1 允许区间上下限顺序错误"),
        (lambda p: p["flow_plan"].__setitem__("q1_allowed_range_ul_min", [10.0]), "Q1 允许区间必须是"),
        (lambda p: p["flow_plan"].__setitem__("baseline_q1_ul_min", float("nan")),
         "基线 Q1 流量不是有限数值"),
        (lambda p: p["apparatus"].__setitem__("scale_um_per_px", None), "像素标定比例"),
        (lambda p: p["apparatus"].__setitem__("scale_evidence", None), "像素标定来源"),
        (lambda p: p.pop("termination"), "缺少 termination 段"),
    ],
)
def test_invalid_plan_values_are_rejected(mutate, expect: str) -> None:
    plan = copy.deepcopy(valid_plan())
    mutate(plan)
    assert expect in " | ".join(tool.validate_session_plan(plan))


def test_missing_or_broken_plan_file_is_reported_not_crashed(tmp_path: Path) -> None:
    report = tool.run_live(_args(tmp_path / "missing.json"), tmp_path / "out")
    assert report["verdict"] == "LIVE_BLOCKED"
    assert report["touched_hardware"] is False
    assert "无法读取或解析" in report["unmet"][0]

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    report = tool.run_live(_args(broken), tmp_path / "out")
    assert report["verdict"] == "LIVE_BLOCKED"
    assert "无法读取或解析" in report["unmet"][0]


def test_live_acceptance_gap_list_is_empty_after_wiring() -> None:
    assert tool.live_acceptance_gaps() == []


# ---------------------------------------------------------------- 固定装置

def valid_plan(output_directory: Path | None = None) -> dict:
    """一份填写完毕的计划——通过校验所需的最小集合。"""
    return {
        "document_type": "bench_session_plan",
        "execution_enabled": True,
        "control_authorized": True,
        "session_id": "test-session",
        "onsite_ready_record": "operator confirmed fixed tubing, no leak, safe waste outlet",
        "apparatus": {
            "chip_id": "chip-01", "observation_site": "generation-zone",
            "pump_port": "COM12", "pump_address": 1, "pump_baudrate": 1200,
            "pump_parity": "E", "camera_unique_id": "HIKROBOT:DIRECT:0",
            "scale_um_per_px": 1.725, "scale_evidence": "calibration.json sha256:abc123",
        },
        "flow_plan": {
            "baseline_q1_ul_min": 17.5, "baseline_q2_ul_min": 5.0,
            "q1_allowed_range_ul_min": [10.0, 30.0], "q2_allowed_range_ul_min": [2.0, 10.0],
            "max_step_ul_min": 2.0, "max_cumulative_delivery_each_ul": 500.0,
        },
        "capture_plan": {
            "requested_fps": 320.0, "requested_exposure_us": 80.0,
            "camera_backend": "hikrobot-direct",
            "timestamp_source_and_units": "hardware ticks；单位未在仓库中说明，按 §3 记 null 并附原因",
            "output_directory": str(output_directory or "D:/MCS_QT_Data/capture"),
            "max_no_valid_frame_s": 5.0, "max_backlog_frames": 64,
        },
        "termination": {
            "max_session_s": 1800.0, "stop_timeout_s": 30.0,
            "max_stop_retries": 3, "physical_stop_method": "stop_system_and_verify",
        },
        "stability_criteria": {
            "window_s": 60.0, "minimum_dwell_s": 60.0,
            "slope_limit_with_units": "0.05 mm/s^2", "spread_limit_with_units": "0.1 mm/s",
            "minimum_quality_fraction": 0.8,
        },
        "software_readiness": {
            "start_ack_failure_stop_test_passed": True,
            "timestamp_and_raw_capture_test_passed": True,
            "stop_failure_test_passed": True,
        },
    }


def blank_template() -> dict:
    return json.loads(
        (REPO_ROOT / "docs" / "bench_session_template_20260920.json").read_text(encoding="utf-8")
    )


def _args(plan_path: Path, *, preflight_only: bool = False):
    from types import SimpleNamespace

    return SimpleNamespace(plan=str(plan_path), preflight_only=preflight_only)
