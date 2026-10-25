"""Frozen two-stage raw diagnostic capture; hardware only with --live."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import run_r1_front4_live as live
from tools.run_q1_response import ResponseRecorder,ResponseSession
from tools.run_a_first_three import AFirstThreeSession
from tools.plant_flow_transient_capture import CaptureLifecycleError,live_effective_config,declared

PROFILE='q2_70_20_to70_25_20261016'
STAGES=[dict(label='B1_70_20',q1_ul_min=70.,q2_ul_min=20.,duration_s=300.),
        dict(label='B2_70_25',q1_ul_min=70.,q2_ul_min=25.,duration_s=300.)]
LOW_Q1_PROFILE='q1_60_20_to50_20_20261016'
HOLDOUT_PROFILE='holdout_60_20_to65_22_20261016'
REVERSE_Q1_PROFILE='q1_reverse_50_20_to60_20_20261016'
LOW_Q2_PROFILE='q2_70_20_to70_18_20261016'
SHAM_LOW_PROFILE='sham_70_20_20261023'
SHAM_HIGH_PROFILE='sham_70_25_20261023'
REVERSE_Q2_PROFILE='q2_reverse_70_25_to70_20_20261023'
BLOCKS={PROFILE:STAGES,
    LOW_Q1_PROFILE:[dict(label='C1_60_20',q1_ul_min=60.,q2_ul_min=20.,duration_s=300.),
                    dict(label='C2_50_20',q1_ul_min=50.,q2_ul_min=20.,duration_s=300.)],
    HOLDOUT_PROFILE:[dict(label='D1_60_20',q1_ul_min=60.,q2_ul_min=20.,duration_s=300.),
                     dict(label='D2_65_22',q1_ul_min=65.,q2_ul_min=22.,duration_s=300.)],
    REVERSE_Q1_PROFILE:[dict(label='E1_50_20',q1_ul_min=50.,q2_ul_min=20.,duration_s=300.),
                        dict(label='E2_60_20',q1_ul_min=60.,q2_ul_min=20.,duration_s=300.)],
    LOW_Q2_PROFILE:[dict(label='F1_70_20',q1_ul_min=70.,q2_ul_min=20.,duration_s=300.),
                    dict(label='F2_70_18',q1_ul_min=70.,q2_ul_min=18.,duration_s=300.)],
    SHAM_LOW_PROFILE:[dict(label='SA_pre',q1_ul_min=70.,q2_ul_min=20.,duration_s=300.),
                      dict(label='SA_post',q1_ul_min=70.,q2_ul_min=20.,duration_s=300.)],
    SHAM_HIGH_PROFILE:[dict(label='SB_pre',q1_ul_min=70.,q2_ul_min=25.,duration_s=300.),
                       dict(label='SB_post',q1_ul_min=70.,q2_ul_min=25.,duration_s=300.)],
    REVERSE_Q2_PROFILE:[dict(label='RQ2_pre',q1_ul_min=70.,q2_ul_min=25.,duration_s=300.),
                        dict(label='RQ2_post',q1_ul_min=70.,q2_ul_min=20.,duration_s=300.)]}


def target_reserve(used:list[float],cap:float,current=(70.,20.),target=(70.,25.))->bool:
    return all(u+(max(old,new)+.5)*.5+(new+.5)*5+10<=cap
        for u,old,new in zip(used,current,target))


class LabRecorder(ResponseRecorder):
    def full_rate(self,name:str,elapsed:float)->bool:
        if name in {stage['label'] for block in BLOCKS.values() for stage in block}:
            return elapsed<30 or 180<=elapsed<210 or 270<=elapsed<300
        return super().full_rate(name,elapsed)


class Q2LabSession(ResponseSession):
    @property
    def raw_only(self)->bool:
        return self.plan.get('capture_plan',{}).get('raw_only') is True

    def _write_pump_parameters(self)->None:
        if not self.raw_only:
            return super()._write_pump_parameters()
        self._write_json('raw_only_policy.json',dict(
            geometry_registered=False,geometry_guard_enabled=False,
            measurement_accepted=False,
            reason='Explicit raw-only acquisition; pump, volume, camera and storage protections retained'))
        AFirstThreeSession._write_pump_parameters(self)

    def _check_geometry(self)->None:
        if not self.raw_only:
            super()._check_geometry()

    @property
    def stages(self)->list[dict]:
        return BLOCKS[self.plan['flow_plan']['campaign_profile']]

    def _begin_lab_stage(self,index:int,command=None)->dict:
        row=dict(self.stages[index],index=index+1,same_value_write=False,
            # A same-setting control still executes the verified stop/restart
            # transaction. The pump may skip redundant parameter writes.
            same_setting_transaction=bool(index and all(
                self.stages[0][key] == self.stages[1][key]
                for key in ('q1_ul_min', 'q2_ul_min'))),
            command_started_monotonic=getattr(command,'command_started_monotonic',None),
            command_readback_monotonic=getattr(command,'readback_completed_monotonic',None),
            observation_started_monotonic=self._clock(),planned_duration_s=300.,observation_finished_monotonic=None)
        self.segment_records.append(row)
        self._record(f"segment_{index+1}_{row['label']}_started")
        return row

    def _run_segments(self)->None:
        if not self.raw_only and not self._geometry_registered:raise CaptureLifecycleError('No current geometry registration')
        baseline=self._begin_lab_stage(0)
        self._wait_segment(300.);self._end_segment(baseline)
        self._update_delivery();self._check_geometry()
        current=tuple(self.stages[0][key] for key in ['q1_ul_min','q2_ul_min'])
        target_q=tuple(self.stages[1][key] for key in ['q1_ul_min','q2_ul_min'])
        if not target_reserve(self._delivery_ul,self._delivery_limit_ul,current,target_q):
            raise CaptureLifecycleError('Insufficient same-stroke target reserve')
        self.sink.set_phase('transition',self._clock())
        # Charge the whole transaction at each channel's larger platform
        # rate, including any uncertain stop/write/readback/restart time.
        self._upper_rates=tuple(max(old,new)+.5 for old,new in zip(current,target_q))
        command=self.pump.update_flow_while_running(*target_q)
        self._update_delivery()
        ok=bool(getattr(command,'ok',False) and getattr(command,'still_running',False)
            and getattr(command,'stop_verified_before_write',False) and getattr(command,'restart_verified',False))
        self._record_command(self.stages[0]['label']+'_to_'+self.stages[1]['label'],
            dict(q1_ul_min=target_q[0],q2_ul_min=target_q[1],stop_verified_before_write=getattr(command,'stop_verified_before_write',False),restart_verified=getattr(command,'restart_verified',False)),
            sent_monotonic=getattr(command,'command_started_monotonic',None),
            readback_monotonic=getattr(command,'readback_completed_monotonic',None),ok=ok)
        if not ok:raise CaptureLifecycleError('Target transaction failed')
        q1,q2=self.pump.get_current_q_state()
        if any(not expected-.5<=actual<=expected+.5 for actual,expected in zip([q1,q2],target_q)):
            raise CaptureLifecycleError('Target readback out of range')
        self._upper_rates=tuple(q+.5 for q in target_q)
        self._check_geometry()
        if any(u+rate*5+10>self._delivery_limit_ul for u,rate in zip(self._delivery_ul,self._upper_rates)):
            raise CaptureLifecycleError('Target reserve invalid after transaction')
        target=self._begin_lab_stage(1,command)
        self._wait_segment(300.);self._end_segment(target);self.run_finished=True


def validate(plan:dict)->list[str]:
    errors=live.validate_session_plan(plan);flow=plan.get('flow_plan',{})
    block=BLOCKS.get(flow.get('campaign_profile'))
    if block is None:errors.append('Unknown fixed lab profile');block=STAGES
    current=tuple(block[0][key] for key in ['q1_ul_min','q2_ul_min'])
    target=tuple(block[1][key] for key in ['q1_ul_min','q2_ul_min'])
    if (flow.get('baseline_q1_ul_min'),flow.get('baseline_q2_ul_min'))!=current:errors.append('Baseline differs from fixed block')
    if plan.get('identification_stages')!=block or flow.get('steps')!=[]:errors.append('Frozen two5min stages required')
    cap=flow.get('max_cumulative_delivery_each_ul',0)
    if not isinstance(cap,(int,float)) or isinstance(cap,bool) or not math.isfinite(cap) or not 0<cap<=900:
        errors.append('Invalid delivery cap')
    elif not target_reserve([(q+.5)*5 for q in current],cap,current,target):errors.append('Cap cannot cover whole block')
    usable=flow.get('usable_confirmed_ul',{})
    if any(not isinstance(usable.get(k),(int,float)) or isinstance(usable[k],bool) or not math.isfinite(usable[k]) or usable[k]<cap+10 for k in ['oil','water']):
        errors.append('Fresh refill bounds insufficient')
    raw_only=plan.get('capture_plan',{}).get('raw_only') is True
    if raw_only and plan.get('capture_plan',{}).get('detector_executed') is not False:
        errors.append('Raw-only capture must explicitly disable measurement')
    if not raw_only and (not Path(str(plan.get('geometry_reference_image',''))).is_file() or len(plan.get('reference_walls',[]))!=2):
        errors.append('Same-chip geometry reference required')
    if any(flow.get(key)!=[min(old,new)-.5,max(old,new)+.5]
           for key,old,new in zip(['q1_allowed_range_ul_min','q2_allowed_range_ul_min'],current,target)):
        errors.append('Only declared Q1/Q2 tolerance bounds allowed')
    return errors


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--live',action='store_true');args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'));errors=validate(plan)
    output=Path(plan['capture_plan']['output_directory'])
    if output.exists():errors.append('Output exists')
    if shutil.disk_usage(output.parent).free<plan['capture_plan']['required_free_space_bytes']:errors.append('Insufficient storage')
    print(json.dumps(dict(preflight_gaps=errors,hardware_requested=args.live)),flush=True)
    if errors:return 2
    if not args.live:return 0
    output.mkdir(parents=True)
    identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('run_q1_response.py'),Path(args.plan)]}
    (output/'explicit_source_identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    a=plan['apparatus'];serial=live.SerialConfig(port=a['pump_port'],address=a['pump_address'],baudrate=a['pump_baudrate'],parity=a['pump_parity'])
    pump=live.PumpHardwareService(serial_config=serial,logger=lambda m:print(m,flush=True));camera=live.HikrobotCameraAdapter(logger=lambda m:print(m,flush=True))
    locks=[live.DeviceLock(live.normalize_port_key(serial.port)),live.DeviceLock(live.camera_lock_key(a['camera_unique_id']))]
    sink=LabRecorder(directory=output,interval_s=1,walls=[])
    raw_only=plan.get('capture_plan',{}).get('raw_only') is True
    provenance=dict(directory=output/'provenance',effective=live_effective_config(plan=plan,strict_localization=declared(not raw_only,reason='explicit raw-only acquisition' if raw_only else 'raw diagnostic capture, no fit'),wall_source='none_raw_only' if raw_only else 'registered_same_chip_reference'))
    result=Q2LabSession(plan=plan,pump=pump,camera=camera,locks=locks,sink=sink,output_dir=output,log=lambda m:print(m,flush=True),provenance=provenance).run()
    print(json.dumps(result,ensure_ascii=False),flush=True);return live.exit_code(result)

if __name__=='__main__':raise SystemExit(main())
