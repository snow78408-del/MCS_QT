"""Frozen-reference convolution wall diagnostics and fault replay; offline only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_convolution import digest
from tools.replay_measurement_audit import first_per_second


@dataclass(frozen=True)
class GeometrySettings:
    template_radius_px: int = 16
    search_px: int = 16
    drift_px: float = 3.
    width_change_px: float = 3.
    tilt_change_px: float = 3.
    minimum_correlation: float = .65
    minimum_energy_ratio: float = .35


def profile(image: np.ndarray, wall: dict, half: int, radius: int) -> np.ndarray:
    h,w=image.shape
    x0,x1=wall['x1']*w,wall['x2']*w
    if x1<=x0 or half not in (0,1):
        raise ValueError('Invalid wall span')
    xx=np.linspace(x0+(x1-x0)*(half*.5+.03),x0+(x1-x0)*(half*.5+.47),64)
    yy=np.interp(xx,[x0,x1],[wall['y1']*h,wall['y2']*h])
    grid_y=yy[None,:]+np.arange(-radius,radius+1)[:,None]
    grid_x=np.broadcast_to(xx,grid_y.shape)
    if grid_y.min()<0 or grid_y.max()>h-1 or grid_x.min()<0 or grid_x.max()>w-1:
        raise ValueError('Wall template extends outside image')
    band=cv2.remap(image,grid_x.astype(np.float32),grid_y.astype(np.float32),cv2.INTER_LINEAR)
    # Derivative of a Gaussian suppresses illumination offset; median limits moving droplet texture.
    return np.median(gaussian_filter1d(band.astype(float),1.,axis=0,order=1),axis=1)


class FrozenWallMonitor:
    def __init__(self, image: np.ndarray, walls: list[dict], settings: GeometrySettings | None=None):
        if image.ndim!=2 or image.dtype!=np.uint8 or len(walls)!=2:
            raise ValueError('Mono8 and two wall seeds required')
        self.settings=settings or GeometrySettings(); self.walls=walls; self.shape=image.shape
        self.templates=[profile(image,wall,half,self.settings.template_radius_px)
                        for wall in walls for half in (0,1)]

    def check(self, image: np.ndarray) -> dict:
        cfg=self.settings
        if image.shape!=self.shape or image.dtype!=np.uint8:
            return dict(state='geometry_unresolved',reason='frame_format_changed',eligible=False)
        details=[]
        for k,(wall,half) in enumerate((w,h) for w in self.walls for h in (0,1)):
            ref=self.templates[k]; ref=ref-ref.mean(); energy=float(np.linalg.norm(ref))
            if energy<1.:
                return dict(state='geometry_unresolved',reason='reference_has_weak_wall_evidence',eligible=False)
            try:
                current=profile(image,wall,half,cfg.template_radius_px+cfg.search_px)
            except ValueError as error:
                return dict(state='geometry_unresolved',reason=str(error),eligible=False)
            choices=[]
            for shift in range(-cfg.search_px,cfg.search_px+1):
                segment=current[cfg.search_px+shift:cfg.search_px+shift+len(ref)]
                segment=segment-segment.mean(); norm=float(np.linalg.norm(segment))
                corr=float(ref@segment/(energy*norm)) if norm>1e-9 else 0.
                choices.append((corr,shift,norm/energy))
            corr,shift,ratio=max(choices,key=lambda c:c[0])
            details.append(dict(wall=k//2,half=half,correlation=corr,shift_px=shift,energy_ratio=ratio))
        shifts=np.array([r['shift_px'] for r in details]).reshape(2,2)
        width_change=float(np.max(abs(shifts[1]-shifts[0])))
        tilt_change=float(np.max(abs(shifts[:,1]-shifts[:,0])))
        if any(r['correlation']<cfg.minimum_correlation or r['energy_ratio']<cfg.minimum_energy_ratio
               or abs(r['shift_px'])==cfg.search_px for r in details):
            state='wall_unresolved'
        elif width_change>cfg.width_change_px:
            state='wall_spacing_changed'
        elif tilt_change>cfg.tilt_change_px:
            state='wall_tilt_changed'
        elif np.max(abs(shifts))>cfg.drift_px:
            state='view_drift'
        else:
            state='stable_candidate'
        return dict(state=state,eligible=state=='stable_candidate',details=details,
                    width_change_px=width_change,tilt_change_px=tilt_change,
                    interpretation='offline template consistency, not verified physical walls')


class QualityGate:
    """Replay policy only; no pump/lifecycle dependency or device actions."""
    def __init__(self) -> None:
        self.last_stamp=None; self.last_frame=None; self.bad_streak=0

    def check(self, stamp: float, frame_id: int, state: str) -> str:
        stale=(not np.isfinite(stamp) or frame_id<=0 or self.last_stamp is not None
               and (stamp<=self.last_stamp or frame_id<=self.last_frame))
        if not stale:
            self.last_stamp=stamp; self.last_frame=frame_id
        valid=not stale and state=='stable_candidate'
        self.bad_streak=0 if valid else self.bad_streak+1
        if self.bad_streak>=3:
            return 'hold_and_alert_persistent_invalid'
        return 'candidate_only' if valid else 'hold_stale' if stale else 'hold_invalid'


def replay(manifest: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing overwrite')
    output.mkdir(parents=True); cv2.setNumThreads(1)
    source=json.loads(manifest.read_text(encoding='utf8')); groups={}
    for s in source['stages']:
        if s['date']=='20261023':
            groups.setdefault(s['block'],[]).append(s)
    rows=[]; summaries={}; faults=[]
    for block,members in groups.items():
        grids=[first_per_second(s['frames']) for s in members]
        known_missing=block in ('R1','R2'); monitor=None; refs=[]; reference_error=None
        if not known_missing:
            refs=[r for r in grids[0][:10] if r is not None]
            if len(refs)<8:
                reference_error='insufficient_reference_frames'
            else:
                images=[cv2.imread(r['image'],0) for r in refs]
                if any(im is None for im in images):
                    raise ValueError('Unreadable reference')
                reference=np.median(np.stack(images),axis=0).astype(np.uint8)
                try:
                    monitor=FrozenWallMonitor(reference,members[0]['walls'])
                except ValueError as error:
                    reference_error=str(error)
        tally=Counter(); single_tally=Counter()
        for stage_index,(stage,grid) in enumerate(zip(members,grids)):
            for second in range(0,300,5):
                raw=grid[second]
                if known_missing:
                    result=dict(state='field_report_geometry_unconfirmed',eligible=False)
                elif stage_index==0 and second<10:
                    result=dict(state='reference_warmup',eligible=False)
                elif raw is None:
                    result=dict(state='missing_raw',eligible=False)
                elif monitor is None:
                    result=dict(state='geometry_unresolved',reason=reference_error,eligible=False)
                elif second<4 or any(r is None for r in grid[second-4:second+1]):
                    result=dict(state='temporal_window_unavailable',eligible=False)
                else:
                    window=grid[second-4:second+1]
                    images=[cv2.imread(r['image'],0) for r in window]
                    if any(im is None for im in images):
                        raise ValueError('Unreadable original')
                    single=monitor.check(images[-1]); single_tally[single['state']]+=1
                    image=np.median(np.stack(images),axis=0).astype(np.uint8)
                    result=monitor.check(image)
                    result['single_frame_state']=single['state']
                    result['causal_window']=[dict(t=r['t'],image=r['image'],sha256=digest(Path(r['image']))) for r in window]
                    if second==120:
                        h,w=image.shape
                        shifted=cv2.warpAffine(image,np.float32([[1,0,0],[0,1,8]]),(w,h),borderMode=cv2.BORDER_REFLECT_101)
                        tilted=cv2.warpAffine(image,np.float32([[1,0,0],[.02,1,-.01*w]]),(w,h),borderMode=cv2.BORDER_REFLECT_101)
                        erased=image.copy(); wall=members[0]['walls'][1]
                        for x in range(w):
                            y=round(np.interp(x,[wall['x1']*w,wall['x2']*w],[wall['y1']*h,wall['y2']*h]))
                            lo,hi=max(0,y-40),min(h,y+41)
                            erased[lo:hi,x]=int(np.median(image[lo:hi,x]))
                        for scenario,altered in [('shift_8px',shifted),('tilt',tilted),('erased_wall',erased),
                                                 ('flat_frame',np.full_like(image,int(np.median(image))))]:
                            faults.append(dict(block=block,stage=stage_index+1,scenario=scenario,
                                original_state=result['state'],result=monitor.check(altered),synthetic=True))
                tally[result['state']]+=1
                rows.append(dict(block=block,stage=stage_index+1,second=second,source=raw,result=result,
                                 sha256=digest(Path(raw['image'])) if raw else None))
        summaries[block]=dict(states=dict(tally),single_frame_states=dict(single_tally),frozen_reference=[dict(image=r['image'],sha256=digest(Path(r['image']))) for r in refs],
                              reference_error=reference_error,physical_walls_verified=False)
    gate=QualityGate()
    gate_replay=[gate.check(t,f,s) for t,f,s in [(1,1,'stable_candidate'),(1,1,'stable_candidate'),
        (2,2,'missing_raw'),(3,3,'wall_unresolved'),(4,4,'stable_candidate')]]
    result=dict(blocks=summaries,faults=faults,gate_replay=gate_replay,settings=asdict(GeometrySettings()),
        manifest_sha256=digest(manifest),script_sha256=digest(Path(__file__)),hardware_requested=False,
        control_authorized=False,measurement_accepted=False,
        limitations=['Fixed first-10-second reference, never automatically rebased; itself not human-verified.',
            'R1/R2 field report is an external exclusion, not an automatically detected missing wall.',
            'Observed alarm counts are not false-positive rates without human truth.',
            'Fault injections and gate trace are offline diagnostics; lifecycle freeze/stop is not integrated.',
            'Five past 1Hz frames form a causal median; includes up to four seconds of history, never future frames.',
            '5-second sampling cannot guarantee prompt real-time detection.'])
    for name,value in [('summary.json',result),('rows.json',rows)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(replay(a.manifest,a.output)['blocks'],ensure_ascii=False,indent=2))
