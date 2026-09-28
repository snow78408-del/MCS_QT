"""Run the first four R1 observations with uninterrupted camera capture.

This is deliberately a narrow bench entry point.  It reuses the audited
single-session lifecycle, but keeps frame acquisition on its own thread while
the slow 1200-baud pump link performs the three between-segment writes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.device_lock import DeviceLock, camera_lock_key, normalize_port_key
from backend.provenance import wall_binding_from_file
from backend.pump_hardware.config import SerialConfig
from backend.pump_hardware.service import PumpHardwareService
from backend.vision.cameras.adapters.hikrobot_camera import HikrobotCameraAdapter
from tools.plant_flow_transient_capture import (
    CaptureLifecycleError,
    ChannelWriteError,
    LiveCaptureSession,
    LosslessFrameFactsRecorder,
    NoValidFrameError,
    StopOutcome,
    EXIT_LIVE_BLOCKED,
    declared,
    exit_code,
    live_effective_config,
    load_session_plan,
    validate_session_plan,
)


CAMPAIGN_PROFILES = {
    "r1_front4": [
        ("C_baseline", 70.0, 20.0, False),
        ("C_same_value_write", 70.0, 20.0, True),
        ("L", 60.0, 20.0, True),
        ("C_return", 70.0, 20.0, True),
    ],
    "v1_B1": [
        ("C_baseline", 70.0, 20.0, False),
        ("Q1_50", 50.0, 20.0, True),
        ("C_return_from_50", 70.0, 20.0, True),
        ("Q1_80", 80.0, 20.0, True),
        ("C_return_from_80", 70.0, 20.0, True),
    ],
    "requested_first_two_20260926": [
        ("Q1_50", 50.0, 20.0, False),
        ("Q1_100", 100.0, 20.0, True),
    ],
    "requested_last_20260926": [
        ("Q1_120", 120.0, 20.0, False),
    ],
    "requested_q2_steps_20260926": [
        ("Q2_10", 100.0, 10.0, False),
        ("Q2_30", 100.0, 30.0, True),
        ("Q2_40", 100.0, 40.0, True),
    ],
    "requested_q2_10_only_20260926": [("Q2_10", 100.0, 10.0, False)],
    "requested_q2_30_40_20260926": [
        ("Q2_30", 100.0, 30.0, False),
        ("Q2_40", 100.0, 40.0, True),
    ],
    "requested_q2_30_only_20260926": [("Q2_30", 100.0, 30.0, False)],
    "requested_q2_40_only_20260926": [("Q2_40", 100.0, 40.0, False)],
    "requested_q2_10_30_repeat_20260926": [
        ("Q2_10", 100.0, 10.0, False),
        ("Q2_30", 100.0, 30.0, True),
    ],
    "requested_q1_60_70_80_90_20260926": [
        ("Q1_60", 60.0, 20.0, False),
        ("Q1_70", 70.0, 20.0, True),
        ("Q1_80", 80.0, 20.0, True),
        ("Q1_90", 90.0, 20.0, True),
    ],
}


class R1FrontFourSession(LiveCaptureSession):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.segment_records: list[dict] = []
        self._reader_stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._reader_error: BaseException | None = None

    def run(self) -> dict:
        try:
            self._prepare_provenance()
            self._acquire_locks()
            self._record("device_lock_acquired")
            self._connect_and_verify_initial_state()
            self._configure_camera_and_read_back()
            self._begin_capture_and_save_baseline()
            self._start_reader()
            self._write_pump_parameters()
            self._start_infusion()
            self._run_segments()
        except BaseException as exc:
            self.original_error = exc
        finally:
            self._cleanup_campaign()
            self._close_sink()
        payload = self.report()
        payload["segments"] = list(self.segment_records)
        payload["measurement_chain_status"] = {
            "physical_scale_validated": False,
            "ten_point_automatic_detection_passed": False,
            "use": "pixel_and_raw_timeline_evidence_only",
        }
        self._finalize_outputs(payload)
        return payload

    def _connect_and_verify_initial_state(self) -> None:
        super()._connect_and_verify_initial_state()
        state = self.pump.read_rse()
        parsed = getattr(state, "parsed_reply", None)
        if not bool(getattr(state, "ok", False)) or parsed is None:
            raise CaptureLifecycleError("unable to verify the initial pump run state")
        running = bool(getattr(parsed, "system_running", False)) or any(
            bool(item) for item in getattr(parsed, "channel_running", [])
        )
        if not running:
            self._record("initial_pump_stopped_verified")
            return
        stopped = self.pump.stop_system_and_verify()
        self._record_command(
            "unexpected-initial-running-stop",
            {"channel": "all"},
            sent_monotonic=None,
            readback_monotonic=self._clock(),
            ok=bool(getattr(stopped, "ok", False)),
            detail=None if getattr(stopped, "ok", False) else str(
                getattr(stopped, "reason", None) or getattr(stopped, "error", None)
            ),
        )
        if not getattr(stopped, "ok", False):
            raise CaptureLifecycleError(
                "pump was unexpectedly running at entry and safe stop was not verified"
            )
        raise CaptureLifecycleError(
            "pump was unexpectedly running at entry; it was stopped and this attempt was aborted"
        )

    def _configure_camera_and_read_back(self) -> None:
        super()._configure_camera_and_read_back()
        gain = self.plan["capture_plan"].get("requested_gain")
        if gain is not None:
            self._apply_feature_verified("gain", float(gain))

    def _start_reader(self) -> None:
        self._reader = threading.Thread(
            target=self._reader_loop,
            name="r1-continuous-frame-reader",
            daemon=True,
        )
        self._reader.start()
        self._record("continuous_capture_started_before_pump_write")

    def _write_pump_parameters(self) -> None:
        """Optionally establish the safe ratio while the verified pump is stopped.

        This is restricted to the explicit two-step plan.  The current bench
        can be left at 20:20 after a manual refill; no pump is started until
        CH1 has been written and the safe ratio has been read back.
        """
        current = (self.baseline or {}).get("channels", {})
        q1_current = float(current.get("q1", 0.0)) if current.get("available") else 0.0
        q2_current = float(current.get("q2", 0.0)) if current.get("available") else 0.0
        if q2_current > 0.0 and q1_current <= 2.0 * q2_current:
            flow = self.plan["flow_plan"]
            if (flow.get("campaign_profile") != "requested_first_two_20260926"
                    or flow.get("allow_stopped_safe_initialization") is not True):
                raise ChannelWriteError("当前泵流量比不安全，计划未允许停泵状态下的安全初始化")
            target = float(flow["baseline_q1_ul_min"])
            if target <= 2.0 * q2_current:
                raise ChannelWriteError("安全初始化目标仍不满足 Q1>2×当前 Q2")
            state = self.pump.read_rse()
            parsed = getattr(state, "parsed_reply", None)
            if not getattr(state, "ok", False) or parsed is None or \
                    bool(getattr(parsed, "system_running", False)) or \
                    any(getattr(parsed, "channel_running", [])):
                raise ChannelWriteError("不能确认泵已停止，拒绝初始化流量")
            source = self.pump.read_rsp(1)
            original = getattr(source, "parsed_reply", None)
            if not getattr(source, "ok", False) or original is None:
                raise ChannelWriteError("无法回读 CH1 原参数，拒绝初始化流量")
            params = self.pump._channel_params_preserving_profile(original, target)
            encoded = self.pump.flow_from_channel_params_strict(params)
            if encoded is None or encoded <= 2.0 * q2_current:
                raise ChannelWriteError("量化后的 CH1 初始化值不满足 Q1>2×当前 Q2")
            # Even a parameter write must leave a verified stopped state.
            self.pump_may_be_running = True
            sent = self._clock()
            result = self.pump.write_wsp_and_verify(1, params)
            self._record_command(
                "stopped-safe-initialization-ch1",
                {"channel": 1, "target_ul_min": target},
                sent_monotonic=sent, readback_monotonic=self._clock(),
                ok=bool(getattr(result, "ok", False)),
                detail=None if getattr(result, "ok", False) else str(
                    getattr(result, "reason", None) or getattr(result, "error", None)))
            if not getattr(result, "ok", False):
                raise ChannelWriteError("停泵状态 CH1 初始化写入/回读失败")
            after = self.pump.read_rse()
            parsed_after = getattr(after, "parsed_reply", None)
            if not getattr(after, "ok", False) or parsed_after is None or \
                    bool(getattr(parsed_after, "system_running", False)) or \
                    any(getattr(parsed_after, "channel_running", [])):
                raise ChannelWriteError("初始化后未能确认泵仍处于停止状态")
            new_q1, new_q2 = self.pump.get_current_q_state()
            if new_q2 <= 0.0 or new_q1 <= 2.0 * new_q2:
                raise ChannelWriteError("初始化回读未达到安全流量比")
            self.baseline["channels"] = {"available": True, "q1": new_q1, "q2": new_q2}
            self._record("stopped_safe_initialization_verified")
        super()._write_pump_parameters()

    def _reader_loop(self) -> None:
        no_valid_limit = float(self._capture()["max_no_valid_frame_s"])
        last_valid = self._clock()
        try:
            while not self._reader_stop.is_set():
                frame = self.camera.read_frame(250)
                now = self._clock()
                if not getattr(frame, "valid", False):
                    if now - last_valid > no_valid_limit:
                        raise NoValidFrameError(
                            f"continuous capture had no valid frame for {no_valid_limit}s"
                        )
                    continue
                last_valid = now
                self.frames_seen += 1
                self.sink.submit(frame)
                self.frames_accepted += 1
        except BaseException as exc:
            self._reader_error = exc
            self._reader_stop.set()

    def _raise_reader_error(self) -> None:
        if self._reader_error is not None:
            raise CaptureLifecycleError(
                f"continuous capture failed: {self._reader_error!r}"
            ) from self._reader_error

    def _wait_segment(self, duration_s: float) -> None:
        deadline = self._clock() + duration_s
        while True:
            self._raise_reader_error()
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                return
            self._reader_stop.wait(min(0.25, remaining))

    def _run_segments(self) -> None:
        steps = self.plan["flow_plan"]["steps"]
        profile = str(self.plan["flow_plan"].get("campaign_profile", "r1_front4"))
        expected = CAMPAIGN_PROFILES.get(profile)
        if expected is None or len(steps) != len(expected):
            raise CaptureLifecycleError(f"unsupported or malformed campaign profile: {profile}")
        for index, (step, exp) in enumerate(zip(steps, expected), start=1):
            label, q1, q2, requires_write = exp
            if (
                step.get("label") != label
                or not math.isclose(float(step.get("q1_ul_min")), q1)
                or not math.isclose(float(step.get("q2_ul_min")), q2)
                or not math.isclose(float(step.get("duration_s")), 300.0)
            ):
                raise CaptureLifecycleError(f"step {index} does not match frozen {profile} plan")
            command_started = None
            command_finished = None
            if requires_write:
                result = self.pump.update_flow_while_running(q1, q2)
                command_started = getattr(result, "command_started_monotonic", None)
                command_finished = getattr(result, "readback_completed_monotonic", None)
                ok = bool(getattr(result, "ok", False)) and bool(
                    getattr(result, "still_running", False)
                )
                if profile.startswith("requested_"):
                    ok = (ok and bool(getattr(result, "stop_verified_before_write", False))
                          and bool(getattr(result, "restart_verified", False)))
                self._record_command(
                    f"segment-{index}-{label}",
                    {"q1_ul_min": q1, "q2_ul_min": q2,
                     "same_value": profile == "r1_front4" and index == 2,
                     "stop_verified_before_write": bool(getattr(result, "stop_verified_before_write", False)),
                     "restart_verified": bool(getattr(result, "restart_verified", False))},
                    sent_monotonic=command_started,
                    readback_monotonic=command_finished,
                    ok=ok,
                    detail=None if ok else str(getattr(result, "reason", "update failed")),
                )
                if not ok:
                    raise CaptureLifecycleError(
                        f"segment {index} pump write/readback failed: "
                        f"{getattr(result, 'reason', 'unknown')}"
                    )
            start = self._clock()
            record = {
                "index": index,
                "label": label,
                "q1_ul_min": q1,
                "q2_ul_min": q2,
                "same_value_write": profile == "r1_front4" and index == 2,
                "command_started_monotonic": command_started,
                "command_readback_monotonic": command_finished,
                "observation_started_monotonic": start,
                "planned_duration_s": 300.0,
                "observation_finished_monotonic": None,
            }
            self.segment_records.append(record)
            self._record(f"segment_{index}_{label}_started")
            self._wait_segment(300.0)
            record["observation_finished_monotonic"] = self._clock()
            self._record(f"segment_{index}_{label}_finished")
        self.run_finished = True
        self._record(f"campaign_{profile}_finished")

    def _cleanup_campaign(self) -> None:
        if self.pump_may_be_running:
            try:
                self.stop_outcome = self._stop_pump_with_retries()
            except BaseException as exc:
                self.cleanup_errors.append(exc)
        self._reader_stop.set()
        if self._reader is not None:
            self._reader.join(timeout=3.0)
            if self._reader.is_alive():
                self.cleanup_errors.append(RuntimeError("camera reader did not stop within 3s"))
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


def validate_r1_plan(plan: dict) -> list[str]:
    unmet = validate_session_plan(plan)
    flow = plan.get("flow_plan") or {}
    steps = flow.get("steps")
    if not isinstance(steps, list) or not steps:
        unmet.append("flow_plan.steps must contain a frozen campaign sequence")
        return unmet
    profile = str(flow.get("campaign_profile", "r1_front4"))
    expected = CAMPAIGN_PROFILES.get(profile)
    if expected is None or len(steps) != len(expected):
        unmet.append(f"unsupported or malformed campaign profile: {profile}")
        return unmet
    for index, (step, (label, q1, q2, _)) in enumerate(zip(steps, expected), start=1):
        if not isinstance(step, dict):
            unmet.append(f"step {index} must be an object")
            continue
        try:
            matching = (step.get("label") == label
                        and math.isclose(float(step.get("q1_ul_min")), q1)
                        and math.isclose(float(step.get("q2_ul_min")), q2)
                        and math.isclose(float(step.get("duration_s")), 300.0))
        except (TypeError, ValueError):
            matching = False
        if not matching:
            unmet.append(f"step {index} does not match frozen {profile} plan")
    if unmet:
        return unmet
    if profile.startswith("requested_"):
        proposal = Path(str((plan.get("apparatus") or {}).get("wall_proposal") or ""))
        try:
            _, walls = wall_binding_from_file(
                proposal, source_label="preacquisition_visual_review")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            unmet.append(f"wall proposal is unavailable: {exc}")
        else:
            if len(walls) != 2:
                unmet.append("wall proposal must contain exactly two lines")
    if unmet:
        return unmet
    try:
        q1_budget = sum(float(step["q1_ul_min"]) * float(step["duration_s"]) / 60.0 for step in steps)
        q2_budget = sum(float(step["q2_ul_min"]) * float(step["duration_s"]) / 60.0 for step in steps)
        duration = sum(float(step["duration_s"]) for step in steps)
    except (KeyError, TypeError, ValueError):
        unmet.append("campaign steps must contain numeric q1_ul_min, q2_ul_min and duration_s")
        return unmet
    limit = float(flow.get("max_cumulative_delivery_each_ul", 0))
    if limit < max(q1_budget, q2_budget):
        unmet.append(
            f"cumulative delivery limit {limit} uL does not cover campaign maximum {max(q1_budget, q2_budget)} uL"
        )
    if float((plan.get("termination") or {}).get("max_session_s", 0)) < duration:
        unmet.append(f"session limit must cover {duration} seconds of observations")
    return unmet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--provenance-dir", type=Path,
                        help="追溯目录；缺省为 <plan 输出目录>/provenance。写入失败即在动设备前拒绝启动")
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = load_session_plan(plan_path)
    unmet = validate_r1_plan(plan)
    output = Path(plan["capture_plan"]["output_directory"]).resolve()
    required_free = int(plan["capture_plan"]["required_free_space_bytes"])
    available = shutil.disk_usage(output.parent).free
    if available < required_free:
        unmet.append(f"free space {available} is below required {required_free}")
    if unmet:
        print(json.dumps({"verdict": "LIVE_BLOCKED", "unmet": unmet}, ensure_ascii=False))
        return EXIT_LIVE_BLOCKED
    if args.preflight_only:
        print(json.dumps({"verdict": "LIVE_READY", "output": str(output)}, ensure_ascii=False))
        return 0

    apparatus = plan["apparatus"]
    serial = SerialConfig(
        port=str(apparatus["pump_port"]).strip().upper(),
        address=int(apparatus["pump_address"]),
        baudrate=int(apparatus["pump_baudrate"]),
        parity=str(apparatus["pump_parity"]).strip().upper(),
    )
    pump = PumpHardwareService(serial_config=serial, logger=lambda m: print(m, flush=True))
    camera = HikrobotCameraAdapter(logger=lambda m: print(m, flush=True))
    locks = [
        DeviceLock(normalize_port_key(serial.port)),
        DeviceLock(camera_lock_key(apparatus["camera_unique_id"])),
    ]
    sink = LosslessFrameFactsRecorder(
        directory=output,
        requested_fps=float(plan["capture_plan"]["requested_fps"]),
        max_backlog=int(plan["capture_plan"]["max_backlog_frames"]),
    )
    profile = str(plan["flow_plan"].get("campaign_profile", "r1_front4"))
    wall_source = "current_frame_localized"
    wall_block = None
    if profile.startswith("requested_"):
        proposal_path = Path(str(apparatus.get("wall_proposal") or ""))
        if not proposal_path.is_file():
            raise ValueError("requested step campaign requires a preacquisition wall proposal")
        wall_block, wall_lines = wall_binding_from_file(
            proposal_path, source_label="preacquisition_visual_review")
        if len(wall_lines) != 2:
            raise ValueError("wall proposal must contain exactly two lines")
        wall_source = "preacquisition_visual_review_not_applied_to_raw_capture"
    provenance_spec = {
        "directory": args.provenance_dir or (output / "provenance"),
        "effective": live_effective_config(
            plan=plan,
            strict_localization=declared(
                True, reason="实时源：prepare_video 对 camera/realtime 来源声明严格定位"),
            wall_source=wall_source,
            wall_binding_block=wall_block),
    }
    session = R1FrontFourSession(
        plan=plan,
        pump=pump,
        camera=camera,
        locks=locks,
        sink=sink,
        output_dir=output,
        log=lambda m: print(m, flush=True),
        provenance=provenance_spec,
    )
    report = session.run()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # 退出码由三层结果决定：0 只表示该入口声明的任务目标（含测量验收）达成。
    print(json.dumps({"result_layers": report.get("result_layers")}, ensure_ascii=False))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
