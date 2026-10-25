"""Operator acknowledgement/stop files only; never connects to hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.q2_prediction_gate import atomic_json


def acknowledge(directory: Path, operator: str, *, both_walls_visible: bool,
                complete_droplet_endpoints_visible: bool) -> None:
    request = json.loads((directory/'readiness_request.json').read_text(encoding='utf-8'))
    now = time.monotonic()
    if (not operator.strip() or not both_walls_visible or not complete_droplet_endpoints_visible
            or request['status'] != 'waiting' or not request['started_monotonic'] <= now < request['deadline_monotonic']
            or (directory/'STOP').exists() or (directory/'PAUSE').exists()):
        raise ValueError('Readiness not confirmed or request expired/stopped')
    response = dict(session_id=request['session_id'],nonce=request['nonce'],operator=operator,
        confirmed_monotonic=now,both_walls_visible=True,complete_droplet_endpoints_visible=True)
    atomic_json(directory/'readiness_confirmation.json', response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--action',choices=('ready','stop','pause'),required=True)
    parser.add_argument('--operator',default='')
    parser.add_argument('--walls-visible',action='store_true')
    parser.add_argument('--complete-droplets',action='store_true')
    args = parser.parse_args()
    if args.action == 'ready':
        acknowledge(args.session,args.operator,both_walls_visible=args.walls_visible,
                    complete_droplet_endpoints_visible=args.complete_droplets)
        print('Readiness acknowledgement saved; capture session verifies it before starting the clock.')
    else:
        # Pausing this bounded experiment means stopping; continuing needs a new run/refill record.
        (args.session/'STOP').touch(exist_ok=True)
        print('Stop requested through existing capture lifecycle; wait for verified stop status.')
