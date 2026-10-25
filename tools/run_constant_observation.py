"""Single bounded70/20 raw observation; default is device-free preflight."""
from __future__ import annotations

import argparse
import json
import math
import queue
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_a_first_three import AFirstThreeSession, RawSurveyRecorder
from tools import run_r1_front4_live as live
from tools.plant_flow_transient_capture import CaptureLifecycleError, live_effective_config, declared

PROFILE = 'constant_70_20_20261016'


def sampling(elapsed: float) -> tuple[int, bool]:
    """Fixed20Hz slots and three30s full-rate windows; one write per input frame."""
    slot = int(math.floor(elapsed * 20 + 1e-7))
    burst = any(start <= elapsed < start + 30 for start in (0, 180, 540))
    return slot, burst


class ConstantRecorder(RawSurveyRecorder):
    def full_rate(self, name: str, elapsed: float) -> bool:
        return name == 'storage_pilot' or (name == 'H' and sampling(elapsed)[1])

    def submit(self, frame: object) -> None:
        self.check()
        image = frame.image
        stamp = float(frame.host_monotonic_timestamp)
        identifier = int(frame.hardware_frame_id)
        if not math.isfinite(stamp) or stamp <= 0 or identifier <= 0:
            raise ValueError('Invalid frame identity or timestamp')
        if self._last_stamp is not None and (stamp <= self._last_stamp or identifier <= self._last_id):
            raise ValueError('Regressed frame identity or timestamp')
        if image.shape != self._shape or image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError('Original image format changed')
        gap = max(0, identifier - self._last_id - 1) if self._last_id is not None else 0
        self.hardware_gaps += gap
        self._last_stamp, self._last_id = stamp, identifier
        software_id = self.received
        self.received += 1
        with self._lock:
            if self._phase is None:
                return
            name, start = self._phase
            elapsed = stamp - start
            if elapsed < 0:
                return
            slot, _ = sampling(elapsed)
            burst = self.full_rate(name, elapsed)
            grid = slot > self._slot
            if grid:
                self._slot = slot
            if not grid and not burst:
                return
        row = dict(phase=name, elapsed_s=elapsed, sample_slot=slot if grid else None,
                   grid_slot=slot // 20 if grid and slot % 20 == 0 else None,
                   burst=burst, software_frame_id=software_id, hardware_frame_id=identifier,
                   host_monotonic_s=stamp, wall_clock_s=float(getattr(frame,'timestamp',0)),
                   hardware_timestamp_ticks=int(getattr(frame,'hardware_timestamp_ticks',0)),
                   sdk_host_timestamp_ticks=int(getattr(frame,'sdk_host_timestamp_ticks',0)),
                   lost_packet_count=int(getattr(frame,'lost_packet_count',0)),
                   hardware_gap_before_frame=gap, raw_shape=list(image.shape),
                   image=f'frames/{name}_sw{software_id:08d}_hw{identifier:08d}.png',
                   endpoint_validated=False)
        try:
            self._queue.put_nowait((image.copy(),row))
        except queue.Full as exc:
            raise CaptureLifecycleError('Bounded PNG queue full') from exc


class ConstantSession(AFirstThreeSession):
    def _run_segments(self) -> None:
        duration=float(self.plan['observation_s'])
        start=self._clock()
        record=dict(index=1,label='H',q1_ul_min=70.,q2_ul_min=20.,same_value_write=False,
                    command_started_monotonic=None,command_readback_monotonic=None,
                    observation_started_monotonic=start,planned_duration_s=duration,
                    observation_finished_monotonic=None)
        self.segment_records.append(record)
        self._record('segment_1_H_started')
        self._wait_segment(duration)
        record['observation_finished_monotonic']=self._clock()
        self._record('segment_1_H_finished')
        self.run_finished=True


def validate(plan: dict) -> list[str]:
    unmet=live.validate_session_plan(plan)
    flow=plan.get('flow_plan',{})
    if flow.get('campaign_profile')!=PROFILE or (flow.get('baseline_q1_ul_min'),flow.get('baseline_q2_ul_min'))!=(70.,20.):
        unmet.append('Only the constant70/20 profile is supported')
    duration=plan.get('observation_s')
    if not isinstance(duration,(int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or not 60<=duration<=720:
        unmet.append('observation_s must be finite within60..720')
        return unmet
    limit=flow.get('max_cumulative_delivery_each_ul',0)
    if not isinstance(limit,(int,float)) or isinstance(limit,bool) or not math.isfinite(limit) or not 0<limit<=900:
        unmet.append('Delivery cap must be within0..900uL')
        return unmet
    if limit < 70.5*(duration+30)/60:
        unmet.append('Delivery cap cannot cover observation and30s startup reserve')
    confirmed=flow.get('usable_confirmed_ul',{})
    if not all(isinstance(confirmed.get(k),(float,int)) and not isinstance(confirmed[k],bool)
               and math.isfinite(confirmed[k]) and confirmed[k]>=rate*(duration+30)/60+10
               for k,rate in [('oil',70.5),('water',20.5)]):
        unmet.append('Fresh confirmed usable oil/water volume is insufficient')
    if flow.get('steps')!=[]:
        unmet.append('No additional flow steps permitted')
    return unmet


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    unmet=validate(plan)
    output=Path(plan['capture_plan']['output_directory'])
    if output.exists():
        unmet.append('Output directory already exists; refuse overwrite')
    if shutil.disk_usage(output.parent).free < plan['capture_plan']['required_free_space_bytes']:
        unmet.append('Insufficient disk space')
    print(json.dumps(dict(preflight_gaps=unmet,hardware_requested=args.live),ensure_ascii=False),flush=True)
    if unmet:
        return 2
    if not args.live:
        return 0
    app=plan['apparatus']
    serial=live.SerialConfig(port=app['pump_port'],address=app['pump_address'],baudrate=app['pump_baudrate'],parity=app['pump_parity'])
    pump=live.PumpHardwareService(serial_config=serial,logger=lambda m:print(m,flush=True))
    camera=live.HikrobotCameraAdapter(logger=lambda m:print(m,flush=True))
    locks=[live.DeviceLock(live.normalize_port_key(serial.port)),live.DeviceLock(live.camera_lock_key(app['camera_unique_id']))]
    recorder=ConstantRecorder(directory=output,interval_s=1.,walls=[])
    provenance=dict(directory=output/'provenance',effective=live_effective_config(plan=plan,strict_localization=declared(True,reason='raw capture; geometry reviewed separately'),wall_source='new_chip_raw_no_reused_walls'))
    session=ConstantSession(plan=plan,pump=pump,camera=camera,locks=locks,sink=recorder,
                            output_dir=output,log=lambda m:print(m,flush=True),provenance=provenance)
    report=session.run()
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return live.exit_code(report)


if __name__=='__main__':
    raise SystemExit(main())
