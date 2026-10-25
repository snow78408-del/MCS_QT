"""Package already fitted parameters for a Q2 validation gate; no fit or hardware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.q2_prediction_gate import (BASELINE_METHOD, SOURCE_FILES, ROOT, atomic_json,
                                     object_hash, predict, sha256, validate_bundle)


def freeze(plan_path: Path, candidate_path: Path, output: Path) -> str:
    if output.exists():
        raise ValueError('Refusing overwrite of frozen bundle')
    plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
    candidate=json.loads(candidate_path.read_text(encoding='utf-8-sig'))
    if candidate.get('baseline_method') != BASELINE_METHOD:
        raise ValueError('Fit must explicitly use the same fresh-pipeline tail baseline method')
    if plan['session_id'] in candidate['training_block_ids']:
        raise ValueError('Validation session is listed as training')
    bundle=dict(schema_version=1,study_id=plan['study_id'],chip_id=plan['apparatus']['chip_id'],
        baseline_method=BASELINE_METHOD,q1_fixed=70,q2_range=[20,25],
        models=candidate['models'],primary_model=candidate['primary_model'],training_block_ids=candidate['training_block_ids'],
        measurement_source_hashes={name:sha256(ROOT/name) for name in SOURCE_FILES},
        baseline_worker_sha256=sha256(ROOT/'tools/q2_prediction_gate.py'),
        reference_image_sha256=sha256(Path(plan['geometry_reference_image'])),
        reference_walls_sha256=object_hash(plan['reference_walls']),
        camera_settings=dict(camera_unique_id=plan['apparatus']['camera_unique_id'],
            exposure_us=plan['capture_plan']['requested_exposure_us'],gain=plan['capture_plan']['requested_gain']),
        candidate_sha256=sha256(candidate_path),created_utc=datetime.now(timezone.utc).isoformat(),
        control_authorized=False,measurement_accepted=False)
    validate_bundle(bundle)
    predict(bundle,[1.]*6,[70,20],[70,25]);predict(bundle,[1.]*6,[70,25],[70,20])
    atomic_json(output,bundle)
    return sha256(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','candidate','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();print(freeze(args.plan,args.candidate,args.output))
