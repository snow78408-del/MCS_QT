"""Bounded 70:20 baseline; no video and no automatic steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.device_lock import DeviceLock, camera_lock_key, normalize_port_key
from backend.pump_hardware.config import SerialConfig
from backend.pump_hardware.service import PumpHardwareService
from backend.runtime_paths import user_data_dir
from backend.vision.cameras.adapters.hikrobot_camera import HikrobotCameraAdapter
from backend.provenance import sha256_file, wall_binding_from_file  # noqa: E402
from backend.vision.config import DetectorConfig, DebugConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_measurement import (
    FrameEvidence,
    ScaleEvidence,
    measure_generation_plugs,
    verify_reused_walls,
)
from tools.plant_flow_transient_capture import (
    TASK_GOAL_DIAGNOSTIC_CAPTURE,
    declared,
    exit_code,
    live_effective_config,
)
from tools.run_r1_front4_live import R1FrontFourSession


class MeasurementSink:
    """像素域诊断 sink。

    走**生产测量入口** ``measure_generation_plugs``：不绕过公开接口、不直接调用
    detector 的私有方法、不用 ``min(gray.shape)`` 猜参考宽度。管壁是复用的目视提议，
    因此每帧用 ``verify_reused_walls`` 在当前画面上核验；未声明独立标尺，所以
    ``valid`` 预期为 False，物理单位一律不产出，只保留像素域候选与拒绝理由。
    """

    def __init__(self, directory: Path, walls: list[dict], duration_s: int = 90) -> None:
        self.directory = directory
        self.walls = walls
        tuning = json.loads((user_data_dir() / "config/vision_tuning_parameters.json").read_text(encoding="utf-8"))
        config = DetectorConfig(**tuning["detector"])
        config.measurement_mode = "generation_plug"
        config.generation_min_length_ratio = 0.5
        self.detector = DropletDetector(config, DebugConfig())
        self.rows: list[dict] = []
        self.last_sample = 0.0
        self.start = time.monotonic()
        self.stills = 0
        self.still_interval_s = duration_s / 9.0
        # 复用的目视提议：只声明像素域，不声明任何 µm 标尺。
        self.scale = ScaleEvidence(um_per_px=None, source="configured_optical", validated=False,
                                  detail="像素域诊断：未声明独立标尺")
        self.profiles: list[dict] = []
        self.provenance: dict | None = None
        self.still_index: list[dict] = []
        self.session_id = directory.name
        self.capture_id = f"{directory.name}#diagnostic"

    def set_provenance(self, payload: dict) -> None:
        """由会话在**硬件动作之前**注入源码/配置指纹，逐行引用它绑定配置版本。"""
        self.provenance = dict(payload)

    def _provenance_block(self) -> dict:
        if not self.provenance:
            return {"written": False, "why": "本次未提供追溯规格"}
        return {"written": bool(self.provenance.get("written")),
                "config_fingerprint": self.provenance.get("config_fingerprint"),
                "config_version": self.provenance.get("config_version"),
                "source_fingerprint": self.provenance.get("source_fingerprint")}

    def submit(self, packet) -> None:
        stamp = float(packet.host_monotonic_timestamp or 0)
        if stamp <= 0:
            raise RuntimeError("Missing acquisition timestamp")
        if stamp - self.last_sample < 0.2:
            return
        self.last_sample = stamp
        frame_id = int(packet.hardware_frame_id or 0)
        consistency = verify_reused_walls(packet.image, self.walls, frame_id=frame_id)
        trace: dict = {}
        measurement = measure_generation_plugs(
            packet.image,
            detector=self.detector,
            wall_lines=self.walls,
            scale=self.scale,
            frame_evidence=FrameEvidence(
                frame_id=frame_id, hardware_frame_id=frame_id, capture_monotonic=stamp,
                localization_frame_id=frame_id, time_source="host_clock_proxy"),
            duct_depth_um=None, duct_depth_source="unknown", duct_depth_validated=False,
            wall_source="reused", wall_consistency=consistency,
            trace=trace,
        )
        summary = measurement.detection_trace_summary()
        row = {
            "hardware_frame_id": frame_id,
            "capture_monotonic": stamp,
            "valid": bool(measurement.valid),
            "reason": measurement.reason,
            "geometry_source": "visually_selected_session_proposal",
            "wall_verification_status": consistency.get("status"),
            "wall_verification_frame_id": consistency.get("verified_frame_id"),
            "rectified_shape": list(measurement.rectified_shape or ()),
            "reference_width_px": summary.get("reference_width_px"),
            "reference_width_source": summary.get("reference_width_source"),
            "minimum_length_px": summary.get("minimum_length_px"),
            "detector_config_channel_px": summary.get("config_channel_px"),
            "selected_pixels": summary.get("selected_intervals", []),
            "outline_checks": summary.get("raw_outline_contrast_checks", []),
            "physical_scale_validated": False,
            # 逐帧媒体：本次诊断不保存短视频（video_saved=false），所以**逐行**没有
            # 可解码的图像引用。这里如实标缺失，不用邻近静帧或文件序号猜配对。
            "source_frame_ref": {
                "version": 2, "available": False,
                "missing_reason": ("本次诊断未保存短视频，逐行没有可解码图像引用；"
                                   "只有 10 张静帧，见 stills_index.json"),
                "session_id": self.session_id, "capture_id": self.capture_id,
                "session_root": str(self.directory),
                "hardware_frame_id": frame_id,
                "capture_monotonic": stamp,
                "software_frame_index": None,
            },
            "provenance": self._provenance_block(),
        }
        self.rows.append(row)
        self.profiles.append({"reason": measurement.reason, "valid": measurement.valid,
                              "reference_width_source": summary.get("reference_width_source")})
        if self.stills < 10 and time.monotonic() - self.start >= self.stills * self.still_interval_s:
            name = f"check_{self.stills:02d}.png"
            path = self.directory / name
            if not cv2.imwrite(str(path), packet.image):
                raise RuntimeError("Diagnostic still write failed")
            self.still_index.append({
                "still_path": name,
                "session_root": str(self.directory),
                "session_id": self.session_id,
                "capture_id": self.capture_id,
                "hardware_frame_id": frame_id,
                "capture_monotonic": stamp,
                "still_sha256": sha256_file(path),
                "provenance": self._provenance_block(),
            })
            self.stills += 1

    def close(self) -> None:
        with (self.directory / "measurements.ndjson").open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + "\n")
        # 静帧索引：每张带文件哈希与采集身份，可离线核验（这是本入口真正可绑定的图像证据）。
        with (self.directory / "stills_index.json").open("w", encoding="utf-8") as stream:
            json.dump({"session_id": self.session_id, "capture_id": self.capture_id,
                       "provenance": self._provenance_block(),
                       "stills": self.still_index}, stream, ensure_ascii=False, indent=2)

    def summary(self) -> dict:
        reasons: dict = {}
        for item in self.profiles:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        return {"sampled_frames": len(self.rows),
                "frames_with_complete_plugs": sum(bool(r["selected_pixels"]) for r in self.rows),
                "video_saved": False, "diagnostic_stills": self.stills,
                # 「采样前提」指实验级验收，本诊断入口从未声明通过；它是**声明值**，
                # 不是由帧号缺口计算出来的（见 report() 的 result_layers 分层）。
                "sampling_premise_ok": False,
                "measurement_accepted": False,
                "measurement_acceptance_reason": "diagnostic_pixel_domain_only",
                "measurement_reason_counts": reasons,
                "reason": "Manual geometry and physical scale require review before step experiments"}


class ValidationSession(R1FrontFourSession):
    def __init__(self, *, duration_s: int = 90, **kwargs):
        super().__init__(**kwargs)
        if not 90 <= duration_s <= 300:
            raise ValueError("Baseline duration must be 90..300 seconds")
        self.duration_s = duration_s

    def _run_segments(self) -> None:
        start = self._clock()
        deadline = start + self.duration_s
        self._record("baseline_started")
        self.segment_records.append({"label": "C_baseline", "q1_ul_min": 70,
                                     "q2_ul_min": 20, "observation_started_monotonic": start,
                                     "planned_duration_s": self.duration_s})
        while self._clock() < deadline:
            self._wait_segment(min(30.0, deadline - self._clock()))
            if self._clock() >= deadline:
                break
            sent = self._clock()
            result = self.pump.read_rse()
            running, reason = self.pump.are_required_channels_running(
                [1, 2], run_state=getattr(result, "parsed_reply", None)) if result.ok else (False, "read_failed")
            self._record_command("baseline-run-state", {"channels": [1, 2]},
                                 sent_monotonic=sent, readback_monotonic=self._clock(),
                                 ok=bool(running), detail=reason)
            if not running:
                raise RuntimeError(f"Baseline run state not verified: {reason}")
            self.log(json.dumps({"baseline_elapsed_s": round(self._clock() - start, 1),
                                 "run_state_verified": True, **self.sink.summary()}))
        self.segment_records[-1]["observation_finished_monotonic"] = self._clock()
        self.run_finished = True
        self._record("baseline_finished")

    def _configure_camera_and_read_back(self) -> None:
        super()._configure_camera_and_read_back()
        self._apply_feature_verified("gain", 6.0)


def load_walls_and_binding(proposal_path: Path, *, source_label: str) -> tuple[list[dict], dict]:
    """**一次**读取提议文件，返回 ``(给 sink 用的墙线, 给追溯用的绑定)``。

    两者必须来自同一次读取：先解析墙线、再单独读一次算指纹，两次之间文件被改就会
    得到「指纹属于 A、墙线属于 B」的自相矛盾记录。这里是入口的**配置组装路径**，
    测试直接调它，而不是只测 ``wall_binding`` 辅助函数。
    """
    block, walls = wall_binding_from_file(proposal_path, source_label=source_label)
    if len(walls) != 2:
        raise ValueError(f"提议文件必须给出恰好两条墙线，得到 {len(walls)} 条")
    return walls, block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration-seconds", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path,
                        help="追溯目录；缺省为 <output>/provenance。追溯写入失败即在动设备前拒绝启动")
    parser.add_argument("--walls-proposal", type=Path,
                        help="固定复用的目视管壁提议文件；其内容指纹与墙线值一起进追溯")
    args = parser.parse_args()
    if not 90 <= args.duration_seconds <= 300:
        parser.error("duration must be 90..300 seconds")
    plan = json.loads(Path("output/v1-b1-20260923/session_plan.json").read_text(encoding="utf-8"))
    duration = args.duration_seconds
    plan["session_id"] = f"current-channel-baseline-{duration}s"
    plan["onsite_ready_record"] = "User reported about 1 mL each before two completed 90 s checks; subsequent remaining volume is estimated."
    plan["flow_plan"].update(remaining_volume_each_ul=700, max_cumulative_delivery_each_ul=400,
                             baseline_q1_ul_min=70, baseline_q2_ul_min=20, steps=[])
    plan["flow_plan"]["remaining_volume_evidence"] = "Conservative working estimate from user report and logged command durations; not a physical readback"
    plan["flow_plan"]["max_plunger_travel_each_mm"] = None
    plan["termination"].update(max_session_s=duration, max_segment_s=duration)
    plan["capture_plan"]["output_directory"] = str(args.output.resolve())
    plan["capture_plan"]["raw_recording_format"] = "No video; sampled measurements and at most ten diagnostic stills"
    plan["apparatus"]["load_condition"] = "Two short checks since reported refill; see volume estimate"
    plan["apparatus"]["syringe_and_code_evidence"] = "Preserve current pump syringe configuration; do not reconfigure syringe"
    if not args.execute:
        print(json.dumps({"duration_s": duration, "q1_ul_min": 70, "q2_ul_min": 20,
                          "nominal_consumption_ul": [70 * duration / 60, 20 * duration / 60], "video_saved": False}))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "session_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    # 固定复用的目视提议：墙线值与内容指纹来自**同一次读取**，两者都进追溯。
    walls, wall_binding_block = load_walls_and_binding(
        Path(args.walls_proposal or "output/current-wall-fullspan-20260923/proposal.json"),
        source_label="reused_proposal")
    serial = SerialConfig(port="COM12", address=1, baudrate=1200, parity="E")
    camera = HikrobotCameraAdapter()
    pump = PumpHardwareService(serial_config=serial)
    sink = MeasurementSink(args.output, walls, duration_s=duration)
    provenance_spec = {
        "directory": args.provenance_dir or (args.output / "provenance"),
        "effective": live_effective_config(
            plan=plan, detector_config=sink.detector.config,
            strict_localization=declared(
                False, reason="诊断入口走复用提议，不启用实时严格定位门控"),
            wall_source="reused_proposal",
            wall_binding_block=wall_binding_block),
    }
    session = ValidationSession(duration_s=duration, plan=plan, pump=pump, camera=camera,
        locks=[DeviceLock(normalize_port_key(serial.port)), DeviceLock(camera_lock_key(plan["apparatus"]["camera_unique_id"]))],
        sink=sink, output_dir=args.output,
        log=lambda message: print(message, flush=True),
        task_goal=TASK_GOAL_DIAGNOSTIC_CAPTURE,
        provenance=provenance_spec)
    result = session.run()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    # 三层结果分别打印：采集完成 / 测量验收 / 停止状态。退出码 0 只表示该入口声明的
    # 诊断采集目标达成，不表示测量通过。
    print(json.dumps({"result_layers": result.get("result_layers")}, ensure_ascii=False),
          flush=True)
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
