"""Fixed 70/25, 60-second raw pilot; hardware only with --live.

This is not the formal SB control block. All pump, camera, geometry and stop
operations use the existing ResponseSession lifecycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_q2_lab_capture import LabRecorder, live
from tools.run_q1_response import ResponseSession
from tools.plant_flow_transient_capture import CaptureLifecycleError, declared, live_effective_config

PROFILE = "q2_sb_pilot_70_25_60s_20261023"
STAGE = dict(label="SB_pilot_60s", q1_ul_min=70., q2_ul_min=25., duration_s=60.)


class PilotRecorder(LabRecorder):
    def full_rate(self, name: str, elapsed: float) -> bool:
        return elapsed < 30 if name == STAGE["label"] else super().full_rate(name, elapsed)


class SBPilotSession(ResponseSession):
    def _run_segments(self) -> None:
        if not self._geometry_registered:
            raise CaptureLifecycleError("No current geometry registration")
        self._check_geometry()
        self._update_delivery()
        if any(used + rate + 10 > self._delivery_limit_ul
               for used, rate in zip(self._delivery_ul, (70.5, 25.5))):
            raise CaptureLifecycleError("Insufficient remaining pilot reserve")
        row = dict(STAGE, index=1, planned_duration_s=60., pilot_not_formal_sb=True,
                   same_value_write=False, same_setting_transaction=False,
                   command_started_monotonic=None, command_readback_monotonic=None,
                   observation_started_monotonic=self._clock(), observation_finished_monotonic=None)
        self.segment_records.append(row)
        self._record("segment_1_SB_pilot_60s_started")
        self._wait_segment(60.)
        self._end_segment(row)
        self.run_finished = True


def validate(plan: dict) -> list[str]:
    errors = live.validate_session_plan(plan)
    flow = plan.get("flow_plan", {})
    if (flow.get("campaign_profile") != PROFILE
            or (flow.get("baseline_q1_ul_min"), flow.get("baseline_q2_ul_min")) != (70., 25.)
            or flow.get("steps") != [] or plan.get("identification_stages") != [STAGE]
            or plan.get("capture_mode") != "pilot_60s_not_formal_sb"):
        errors.append("Only the fixed 70/25, single 60s pilot is permitted")
    if (flow.get("q1_allowed_range_ul_min"), flow.get("q2_allowed_range_ul_min")) != ([69.5, 70.5], [24.5, 25.5]):
        errors.append("Only fixed 70/25 tolerance bounds are permitted")
    cap = flow.get("max_cumulative_delivery_each_ul", 0)
    if (isinstance(cap, bool) or not isinstance(cap, (float, int))
            or not math.isfinite(cap) or not 115.75 <= cap <= 900):
        errors.append("Pilot delivery cap must cover startup, 60s and reserve, within 900uL")
    confirmed = flow.get("usable_confirmed_ul", {})
    for name in ("oil", "water"):
        value = confirmed.get(name)
        if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                or not isinstance(cap, (float, int)) or value < cap + 10):
            errors.append(f"Fresh confirmed {name} refill is insufficient")
    if (not Path(str(plan.get("geometry_reference_image", ""))).is_file()
            or len(plan.get("reference_walls", [])) != 2):
        errors.append("Current-chip geometry reference required")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    errors = validate(plan)
    output = Path(plan["capture_plan"]["output_directory"])
    if output.exists():
        errors.append("Output exists; refuse overwrite")
    if shutil.disk_usage(output.parent).free < plan["capture_plan"]["required_free_space_bytes"]:
        errors.append("Insufficient storage")
    print(json.dumps(dict(preflight_gaps=errors, hardware_requested=args.live)), flush=True)
    if errors:
        return 2
    if not args.live:
        return 0
    output.mkdir(parents=True)
    sources = [Path(__file__), Path(__file__).with_name("run_q1_response.py"), args.plan]
    identity = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (output / "explicit_source_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    apparatus = plan["apparatus"]
    serial = live.SerialConfig(port=apparatus["pump_port"], address=apparatus["pump_address"],
                               baudrate=apparatus["pump_baudrate"], parity=apparatus["pump_parity"])
    pump = live.PumpHardwareService(serial_config=serial, logger=lambda message: print(message, flush=True))
    camera = live.HikrobotCameraAdapter(logger=lambda message: print(message, flush=True))
    locks = [live.DeviceLock(live.normalize_port_key(serial.port)),
             live.DeviceLock(live.camera_lock_key(apparatus["camera_unique_id"]))]
    sink = PilotRecorder(directory=output, interval_s=1., walls=[])
    provenance = dict(directory=output / "provenance", effective=live_effective_config(
        plan=plan, strict_localization=declared(True, reason="60s raw pilot, not formal SB or model data"),
        wall_source="registered_same_chip_reference"))
    report = SBPilotSession(plan=plan, pump=pump, camera=camera, locks=locks, sink=sink,
                           output_dir=output, log=lambda message: print(message, flush=True), provenance=provenance).run()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return live.exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
