"""Budget-gated70/20 to60/20 response experiment; no devices without --live."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import run_r1_front4_live as live
from tools.run_a_first_three import AFirstThreeSession
from tools.run_constant_observation import ConstantRecorder
from tools.plant_flow_transient_capture import CaptureLifecycleError,live_effective_config,declared
from backend.vision.config import DetectorConfig,DebugConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_roi import rectify_channel_frame, wall_line_quad
from backend.vision.rectified_measurement import detect_rectified_generation_plugs

PROFILE='q1_response_70_to60_20261016'


def registration_patch(plan:dict,reference:np.ndarray)->tuple[int,int,int,int]:
    """Use a declared static feature for a new field, preserving the old default."""
    custom=plan.get('geometry_guard',{}).get('reference_patch_px')
    roi=[330,150,105,90] if custom is None else custom
    if (not isinstance(roi,(list,tuple)) or len(roi)!=4
            or any(isinstance(value,bool) or not isinstance(value,int) for value in roi)):
        raise CaptureLifecycleError('Static reference patch must contain four integer pixel values')
    x,y,width,height=roi
    if (x<0 or y<0 or width<20 or height<20
            or x+width>reference.shape[1] or y+height>reference.shape[0]):
        raise CaptureLifecycleError('Static reference patch outside image or too small')
    if custom is not None:
        geometry=wall_line_quad(reference.shape[1],reference.shape[0],plan.get('reference_walls',[]))
        if geometry is None:
            raise CaptureLifecycleError('Static reference patch requires declared channel walls')
        rectangle=np.array([[x,y],[x+width,y],[x+width,y+height],[x,y+height]],np.float32)
        area,_=cv2.intersectConvexConvex(rectangle,geometry[0])
        if area>0:
            raise CaptureLifecycleError('Static reference patch must lie outside the channel')
    return x,y,width,height


def response_schedule(plan:dict)->tuple[list[int],float]:
    response=plan.get('response_plan',{})
    policy=response.get('baseline_policy','qualified')
    checks=response.get('baseline_check_s',[120,180,240])
    duration=response.get('target_duration_s',540)
    if policy=='fixed_diagnostic':
        baseline=response.get('baseline_duration_s')
        if baseline in [120,300] and checks==[baseline] and duration==300:
            return checks,300.
        raise ValueError('Unsupported fixed diagnostic schedule')
    if policy!='qualified':raise ValueError('Unsupported baseline policy')
    if checks==[120,180,240] and duration==540:
        return checks,540.
    if checks==[120,180,240,300,360,420] and duration==300:
        return checks,300.
    raise ValueError('Unsupported baseline/target schedule')

def admission(used: list[float], cap: float, duration: float=540.) -> bool:
    return all(u+old*.5+target*duration/60+10<=cap
               for u,old,target in zip(used,[70.5,20.5],[60.5,20.5]))

def stable_check(samples: list[tuple[float,float|None]], end: float) -> dict:
    # Cube-root length has exactly the same relative screens as D for a fixed
    # cross-section and scale; do not invent a physical scale in the gate.
    bins=[]
    for start in np.arange(end-120,end,10):
        v=[float(np.cbrt(y)) for t,y in samples if start<=t<start+10 and y is not None]
        bins.append((float(start),float(np.median(v)) if len(v)>=5 else None))
    available=(all(sum(y is not None for _,y in group)>=5 for group in [bins[:6],bins[6:]])
               and sum(y is not None for _,y in bins)>=10
               and not any(a[1] is None and b[1] is None for a,b in zip(bins,bins[1:])))
    result=dict(end_s=end,available=available,passed=False)
    if not available:return result
    windows=[[float(np.cbrt(y)) for t,y in samples if a<=t<a+60 and y is not None] for a in [end-120,end-60]]
    a,b=windows;ma,mb=float(np.median(a)),float(np.median(b))
    wa=float(np.percentile(a,95)-np.percentile(a,5));wb=float(np.percentile(b,95)-np.percentile(b,5))
    center=abs(mb-ma)/ma*100
    width=abs(wb-wa)/wa*100 if wa>1e-9 else None
    points=[(t,y) for t,y in bins if y is not None]
    slopes=[(yb-ya)/(tb-ta)*60 for i,(ta,ya) in enumerate(points) for tb,yb in points[i+1:]]
    slope=float(np.median(slopes))/float(np.median([y for _,y in points]))*100
    result.update(center_change_pct=center,width_change_pct=width,slope_pct_per_min=slope,
                  passed=bool(center<=1 and width is not None and width<=30 and abs(slope)<=.5))
    return result


class ResponseRecorder(ConstantRecorder):
    def __init__(self,**kwargs)->None:
        super().__init__(**kwargs)
        self._guard_sample=None

    def submit(self,frame:object)->None:
        super().submit(frame)
        stamp=float(frame.host_monotonic_timestamp)
        with self._lock:
            if self._guard_sample is None or stamp-self._guard_sample[1]>=.5:
                self._guard_sample=(frame.image.copy(),stamp)

    def guard_frame(self):
        # One bounded snapshot; never open latest.png while the writer replaces
        # it. Windows readers can otherwise prevent atomic replacement.
        with self._lock:
            return self._guard_sample

    def full_rate(self,name: str,elapsed: float)->bool:
        if name=='P1_target':return elapsed<60 or 420<=elapsed<450
        if name=='P1_baseline':return elapsed<30 or elapsed>=120
        if name=='transition':return True
        return super().full_rate(name,elapsed)


class ResponseSession(AFirstThreeSession):
    def __init__(self,**kwargs)->None:
        super().__init__(**kwargs)
        self._geometry_registered=False
        self._guard_template=None
        self._guard_anchor=None

    def _register_geometry(self)->list:
        self._geometry_registered=False
        self._guard_template=None
        self._guard_anchor=None
        reference=cv2.imread(self.plan['geometry_reference_image'],0)
        image=cv2.imread(str(self.output_dir/'pre_start.png'),0)
        if reference is None or image is None or image.shape!=reference.shape:
            raise CaptureLifecycleError('Current/reference image unavailable')
        # A static, out-of-channel speck identifies field translation. Never
        # use the droplet prediction or flow label to optimize geometry.
        ref_x,ref_y,ref_width,ref_height=registration_patch(self.plan,reference)
        template=reference[ref_y:ref_y+ref_height,ref_x:ref_x+ref_width]
        if float(np.std(template))<1e-6:
            raise CaptureLifecycleError('Static reference patch has no texture; no pump start')
        scores=cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        dx,dy=x-ref_x,y-ref_y
        if score<.85 or abs(dx)>100 or abs(dy)>150:
            raise CaptureLifecycleError('Field registration unresolved; no pump start')
        w=[dict(line) for line in self.plan['reference_walls']]
        for line in w:
            for key in ['x1','x2']:line[key]+=dx/image.shape[1]
            for key in ['y1','y2']:line[key]+=dy/image.shape[0]
        if any(not 0<=v<=1 for line in w for v in line.values()):
            raise CaptureLifecycleError('Registered channel outside frame')
        payload=dict(walls=w,dx=dx,dy=dy,correlation=score,reference=self.plan['geometry_reference_image'],
                     reference_patch_px=[ref_x,ref_y,ref_width,ref_height],
                     reference_sha256=hashlib.sha256(Path(self.plan['geometry_reference_image']).read_bytes()).hexdigest(),measurement_accepted=False)
        self._write_json('registered_geometry.json',payload)
        self._guard_template=image[y:y+template.shape[0],x:x+template.shape[1]].copy()
        self._guard_anchor=(x,y)
        self._geometry_registered=True
        return w

    def _check_geometry(self)->None:
        if self._guard_template is None:return
        sample=self.sink.guard_frame()
        if sample is None:
            if self._delivery_last is not None or self.segment_records:
                raise CaptureLifecycleError('Current geometry guard frame unavailable')
            image=cv2.imread(str(self.output_dir/'pre_start.png'),0)
        else:
            image,stamp=sample
            if self._clock()-stamp>2.:
                raise CaptureLifecycleError('Current geometry guard frame stale')
        if image is None:raise CaptureLifecycleError('Current geometry guard frame unavailable')
        x,y=self._guard_anchor;h,w=self._guard_template.shape
        x0,y0=max(0,x-20),max(0,y-20)
        region=image[y0:min(image.shape[0],y+h+20),x0:min(image.shape[1],x+w+20)]
        if region.shape[0]<h or region.shape[1]<w:raise CaptureLifecycleError('Geometry guard region outside frame')
        _,score,_,(mx,my)=cv2.minMaxLoc(cv2.matchTemplate(region,self._guard_template,cv2.TM_CCOEFF_NORMED))
        dx,dy=mx+x0-x,my+y0-y
        check=dict(monotonic_s=self._clock(),dx=dx,dy=dy,correlation=score,passed=score>=.85 and abs(dx)<=2 and abs(dy)<=2)
        self._write_json('geometry_guard.json',check)
        if not check['passed']:
            raise CaptureLifecycleError(f'Geometry moved/unresolved:dx={dx},dy={dy},correlation={score:.3f}')

    def _wait_segment(self,duration_s:float)->None:
        if self._guard_template is None:
            return super()._wait_segment(duration_s)
        deadline=self._clock()+duration_s
        last_status=self._clock()
        self._check_geometry()
        while self._clock()<deadline:
            super()._wait_segment(min(2.,max(0.,deadline-self._clock())))
            self._check_geometry()
            if self._clock()-last_status>=5.:
                self._record('capture_heartbeat')
                last_status=self._clock()

    def _write_pump_parameters(self)->None:
        self._register_geometry()
        super()._write_pump_parameters()

    def _measure_baseline(self,walls:list)->list[tuple[float,float|None]]:
        # Keep Python vision work in another process: inline baseline analysis
        # coincided with hardware-frame gaps in run01 at each checkpoint.
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--analyze-baseline',str(self.output_dir)],
                              capture_output=True,text=True,encoding='utf-8',timeout=15,check=False)
        if result.returncode:
            raise CaptureLifecycleError(f'Baseline analysis failed:{result.stderr[-1000:]}')
        return [(float(t),float(y) if y is not None else None) for t,y in json.loads(result.stdout)]

    @staticmethod
    def analyze_baseline(directory:Path,walls:list)->list[tuple[float,float|None]]:
        rows=[]
        for line in (directory/'frame_manifest.ndjson').read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['phase']=='P1_baseline' and r['grid_slot'] is not None:rows.append(r)
        det=DropletDetector(DetectorConfig(measurement_mode='generation_plug',generation_phase_mode='transverse_bounded',generation_phase_support_fraction=.02),DebugConfig())
        result=[]
        for r in rows:
            image=cv2.imread(str(directory/r['image']),0)
            if image is None:raise CaptureLifecycleError('Persisted frame unreadable')
            rect=rectify_channel_frame(image,walls);_,trace=detect_rectified_generation_plugs(rect,detector=det)
            ii=trace['selected_intervals']
            result.append((r['elapsed_s'],float(np.median([v[2] for v in ii])) if ii else None))
        return result

    def _begin_segment(self,label:str,q1:float,duration:float,command=None)->dict:
        record=dict(index=len(self.segment_records)+1,label=label,q1_ul_min=q1,q2_ul_min=20.,same_value_write=False,
                    command_started_monotonic=getattr(command,'command_started_monotonic',None),
                    command_readback_monotonic=getattr(command,'readback_completed_monotonic',None),
                    observation_started_monotonic=self._clock(),planned_duration_s=duration,observation_finished_monotonic=None)
        self.segment_records.append(record)
        self._record(f"segment_{record['index']}_{label}_started")
        return record

    def _end_segment(self,r:dict)->None:
        r['observation_finished_monotonic']=self._clock()
        self._record(f"segment_{r['index']}_{r['label']}_finished")

    def _run_segments(self)->None:
        if not self._geometry_registered:raise CaptureLifecycleError('No current geometry registration')
        walls=json.loads((self.output_dir/'registered_geometry.json').read_text())['walls']
        check_times,target_duration=response_schedule(self.plan)
        diagnostic=self.plan.get('response_plan',{}).get('baseline_policy')=='fixed_diagnostic'
        baseline=self._begin_segment('P1_baseline',70.,float(check_times[-1]))
        checks=[];qualified=False
        for end in check_times:
            self._wait_segment(max(0,baseline['observation_started_monotonic']+end-self._clock()))
            if diagnostic:
                self._check_geometry()
                self._write_json('baseline_checks.json',dict(checks=[],qualified_numeric=False,
                    assessment='not_an_online_gate; offline assessment pending',policy='fixed_diagnostic',
                    delivery_ul=self._delivery_ul))
                break
            samples=self._measure_baseline(walls)
            self._check_geometry()
            check=stable_check(samples,end);checks.append(check)
            self._update_delivery()
            qualified=len(checks)>=2 and checks[-1]['passed'] and checks[-2]['passed']
            self._write_json('baseline_checks.json',dict(checks=checks,qualified_numeric=qualified,delivery_ul=self._delivery_ul))
            print(json.dumps(dict(baseline_check=check,qualified_numeric=qualified)),flush=True)
            if qualified:break
        self._end_segment(baseline)
        self._update_delivery()
        if not qualified and not diagnostic:
            raise CaptureLifecycleError(f'Baseline not qualified by{check_times[-1]}s; target not started')
        if not admission(self._delivery_ul,self._delivery_limit_ul,target_duration):
            raise CaptureLifecycleError('Insufficient same-stroke target reserve; target not started')
        # Current phase and high upper rate span the entire update transaction.
        self.sink.set_phase('transition',self._clock())
        result=self.pump.update_flow_while_running(60.,20.)
        self._update_delivery()
        ok=bool(getattr(result,'ok',False) and getattr(result,'still_running',False)
                and getattr(result,'stop_verified_before_write',False) and getattr(result,'restart_verified',False))
        self._record_command('P1_70_to60',dict(q1_ul_min=60.,q2_ul_min=20.,stop_verified_before_write=getattr(result,'stop_verified_before_write',False),restart_verified=getattr(result,'restart_verified',False)),
                             sent_monotonic=getattr(result,'command_started_monotonic',None),readback_monotonic=getattr(result,'readback_completed_monotonic',self._clock()),ok=ok)
        if not ok:raise CaptureLifecycleError('Target transaction failed')
        q1,q2=self.pump.get_current_q_state()
        if not 59.5<=q1<=60.5 or not 19.5<=q2<=20.5:
            raise CaptureLifecycleError('Target readback out of range')
        self._upper_rates=(60.5,20.5)
        if any(u+rate*target_duration/60+10>self._delivery_limit_ul for u,rate in zip(self._delivery_ul,self._upper_rates)):
            raise CaptureLifecycleError('Target reserve invalid after transaction; stopping')
        target=self._begin_segment('P1_target',60.,target_duration,result)
        self._wait_segment(target_duration)
        self._end_segment(target);self.run_finished=True


def validate(plan:dict)->list[str]:
    errors=live.validate_session_plan(plan);flow=plan.get('flow_plan',{})
    try:
        check_times,duration=response_schedule(plan)
    except ValueError as exc:
        errors.append(str(exc));check_times,duration=[120,180,240],540.
    if flow.get('campaign_profile')!=PROFILE or (flow.get('baseline_q1_ul_min'),flow.get('baseline_q2_ul_min'))!=(70.,20.):errors.append('Only70/20 to60/20 profile allowed')
    if flow.get('steps')!=[]:errors.append('Additional steps forbidden')
    cap=flow.get('max_cumulative_delivery_each_ul',0)
    if not isinstance(cap,(int,float)) or isinstance(cap,bool) or not math.isfinite(cap) or not 0<cap<=900:errors.append('Invalid cap')
    elif not admission([70.5*180/60,20.5*180/60],cap,duration):errors.append('Cap cannot cover shortest qualified baseline and target')
    elif duration==300 and not admission([70.5*check_times[-1]/60,20.5*check_times[-1]/60],cap,duration):
        errors.append('Cap cannot cover extended baseline and target')
    usable=flow.get('usable_confirmed_ul',{})
    if any(not isinstance(usable.get(k),(int,float)) or isinstance(usable[k],bool) or not math.isfinite(usable[k]) or usable[k]<cap+10 for k in ['oil','water']):errors.append('Fresh refill bounds insufficient')
    if not Path(str(plan.get('geometry_reference_image',''))).is_file() or len(plan.get('reference_walls',[]))!=2:errors.append('Current geometry reference required')
    return errors


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--live',action='store_true');parser.add_argument('--analyze-baseline');args=parser.parse_args()
    if args.analyze_baseline:
        if args.live or args.plan:parser.error('Offline baseline analysis cannot request hardware or a plan')
        directory=Path(args.analyze_baseline)
        cv2.setNumThreads(1)
        walls=json.loads((directory/'registered_geometry.json').read_text())['walls']
        print(json.dumps(ResponseSession.analyze_baseline(directory,walls)),flush=True);return 0
    if not args.plan:parser.error('--plan required')
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'));errors=validate(plan)
    output=Path(plan['capture_plan']['output_directory'])
    if output.exists():errors.append('Output exists')
    if shutil.disk_usage(output.parent).free<plan['capture_plan']['required_free_space_bytes']:errors.append('Insufficient free space')
    print(json.dumps(dict(preflight_gaps=errors,hardware_requested=args.live),ensure_ascii=False),flush=True)
    if errors:return 2
    if not args.live:return 0
    a=plan['apparatus'];serial=live.SerialConfig(port=a['pump_port'],address=a['pump_address'],baudrate=a['pump_baudrate'],parity=a['pump_parity'])
    pump=live.PumpHardwareService(serial_config=serial,logger=lambda m:print(m,flush=True));camera=live.HikrobotCameraAdapter(logger=lambda m:print(m,flush=True))
    locks=[live.DeviceLock(live.normalize_port_key(serial.port)),live.DeviceLock(live.camera_lock_key(a['camera_unique_id']))]
    sink=ResponseRecorder(directory=output,interval_s=1,walls=[])
    provenance=dict(directory=output/'provenance',effective=live_effective_config(plan=plan,strict_localization=declared(True,reason='raw camera capture; numeric baseline screen'),wall_source='current_image_registered_from_previous_same_chip'))
    result=ResponseSession(plan=plan,pump=pump,camera=camera,locks=locks,sink=sink,output_dir=output,log=lambda m:print(m,flush=True),provenance=provenance).run()
    print(json.dumps(result,ensure_ascii=False),flush=True);return live.exit_code(result)

if __name__=='__main__':raise SystemExit(main())
