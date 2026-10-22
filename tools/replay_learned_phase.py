"""Compare a learned proposal model on a saved, explicitly registered image.

No camera, pump or controller is opened. Geometry JSON contains quad (four
sensor-coordinate corners), nx and ny. Output is diagnostic pixel proposals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.vision.capsule_profile import calibrated_phase_intervals
from backend.vision.learned_phase import LearnedPhaseProposer


def channel_strip(image: np.ndarray, geometry: dict) -> np.ndarray:
    q = np.asarray(geometry['quad'], dtype=np.float32)
    nx, ny = int(geometry['nx']), int(geometry['ny'])
    if (q.shape != (4, 2) or not np.isfinite(q).all() or not 2 <= nx <= 10000 or not 2 <= ny <= 10000
            or not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 1):
        raise ValueError('Invalid registered channel geometry')
    if image.ndim != 2 or image.size == 0:
        raise ValueError('Expected a grayscale image')
    if (q[:,0].min()<0 or q[:,1].min()<0 or q[:,0].max()>image.shape[1]-1 or q[:,1].max()>image.shape[0]-1):
        raise ValueError('Geometry extends outside image')
    t = np.linspace(0, 1, nx)[None,:,None]
    s = np.linspace(0, 1, ny)[:,None,None]
    xy = (1-s)*((1-t)*q[0]+t*q[1])+s*((1-t)*q[3]+t*q[2])
    return cv2.remap(image, xy[:,:,0].astype(np.float32), xy[:,:,1].astype(np.float32), cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--geometry', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    image = (np.load(args.image, allow_pickle=False) if args.image.suffix.lower()=='.npy' else
             cv2.imdecode(np.frombuffer(args.image.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE))
    if image is None:
        parser.error('Could not decode image')
    geometry = json.loads(args.geometry.read_text(encoding='utf-8'))
    strip = channel_strip(image, geometry)
    result = dict(existing=calibrated_phase_intervals(strip, strip.shape[0]-1),
                  learned=LearnedPhaseProposer(args.model).predict(strip),
                  coordinates='registered strip pixels', production_authorized=False,
                  warning='Rough boundary supervision; proposals do not certify droplet detachment or physical size')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(args.output)


if __name__ == '__main__':
    main()
