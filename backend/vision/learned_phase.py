"""Opt-in learned phase proposals for registered grayscale channel strips.

Rough human boundary supervision supports proposals, not certified physical
measurements or proof that a thin neck has detached. PyTorch is optional and
is imported only when constructing/loading this offline candidate model.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def normalized_strip(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2 or min(gray.shape) < 2 or not np.isfinite(gray).all():
        raise ValueError('Expected finite grayscale channel strip')
    strip = cv2.resize(gray.astype(np.float32), (720, 24), interpolation=cv2.INTER_LINEAR)
    return np.clip((strip - strip.mean()) / max(float(strip.std()), 5.), -5., 5.)


def build_network():
    import torch.nn as nn

    layers = []
    channels = 24
    for out, kernel, dilation in [(32, 7, 1), (32, 5, 2), (32, 5, 4), (32, 5, 8), (16, 5, 1)]:
        layers.extend([nn.Conv1d(channels, out, kernel, padding=dilation*(kernel//2), dilation=dilation), nn.GELU()])
        channels = out
    layers.append(nn.Conv1d(channels, 2, 1))
    return nn.Sequential(*layers)


def probability_intervals(probability: np.ndarray, *, threshold: float = .5) -> list[tuple[int, int]]:
    p = np.asarray(probability)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or not 0 < threshold < 1:
        raise ValueError('Expected finite axial probabilities and threshold in (0, 1)')
    active = p >= threshold
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)-1
    # Edge-truncated bodies are not complete droplets.
    return [(int(a), int(b)) for a, b in zip(starts, ends) if a > 0 and b < len(p)-1 and b-a >= 12]


def refine_existing_intervals(
    gray: np.ndarray,
    existing: list[tuple[int, int]],
    proposed: list[tuple[float, float]],
    reference_width_px: float,
) -> list[tuple[int, int]]:
    """Bounded one-to-one correction; never create, merge or remove a body.

    Original phase candidates remain authoritative. Ambiguous associations,
    large moves or newly unsupported carrier flanks retain original endpoints.
    """
    height, width = gray.shape
    channel = float(reference_width_px)
    if not np.isfinite(channel) or not 14 <= channel <= height:
        raise ValueError('Invalid rectified channel width')
    candidates = [(float(a),float(b)) for a,b in proposed
                  if np.isfinite([a,b]).all() and 0 <= a < b < width]
    margin = max(3, int(np.ceil(channel*.2)))
    upper = np.mean(gray[max(1,int(height*.08)):int(height*.25)],axis=0)
    lower = np.mean(gray[int(height*.76):int(height*.94)],axis=0)
    center = np.mean(gray[int(height*.35):int(height*.65)],axis=0)
    curvature = center-(upper+lower)/2
    overlap = lambda a,b,c,d: max(0.,min(b,d)-max(a,c))/max(1.,max(b,d)-min(a,c))
    result = []
    for index,(a,b) in enumerate(existing):
        matches = [(c,d) for c,d in candidates if overlap(a,b,c,d)>=.6]
        chosen = (a,b)
        if len(matches)==1:
            c,d = matches[0]
            unique = sum(overlap(e,f,c,d)>=.6 for e,f in existing)==1
            left,right = int(round(c)),int(round(d))
            previous = existing[index-1][1] if index else -1
            following = existing[index+1][0] if index+1<len(existing) else width
            if (unique and max(abs(left-a),abs(right-b))<=channel*.25
                    and margin<=left<right<width-margin and channel<right-left<channel*4
                    and left>previous and right<following
                    and max(abs(float(np.median(curvature[max(0,left-10):left]))),
                            abs(float(np.median(curvature[right:min(width,right+10)]))))<=12.):
                chosen = (left,right)
        result.append(chosen)
    # Simultaneous outward moves must not make two originally separate bodies overlap.
    bad = {j for i in range(len(result)-1) if result[i][1]>=result[i+1][0] for j in [i,i+1]}
    return [existing[i] if i in bad else pair for i,pair in enumerate(result)]


class LearnedPhaseProposer:
    """Explicitly loaded research proposals; never authorizes control."""

    def __init__(self, model_path: str | Path) -> None:
        import torch

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if checkpoint.get('schema') != 'rough-phase-cnn-v1':
            raise ValueError('Unsupported learned phase model')
        if checkpoint.get('production_authorized') is not False:
            raise ValueError('Unexpected model authorization metadata')
        self._torch = torch
        self._network = build_network()
        self._network.load_state_dict(checkpoint['state_dict'], strict=True)
        self._network.eval()
        self.threshold = float(checkpoint['threshold'])
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError('Invalid learned probability threshold')

    def predict(self, gray: np.ndarray) -> list[tuple[float, float]]:
        if gray.ndim != 2:
            raise ValueError('Expected grayscale channel strip')
        height, width = gray.shape
        if height < 18 or width < max(100, height*3):
            return []
        # Match the existing phase evidence floor: a network trained on
        # droplet-containing images must not invent bodies on blank/noise input.
        # Reject on the raw signal before any resize/normalization so degenerate
        # frames cost no tensor work; the output semantics are unchanged.
        upper = np.mean(gray[max(1, int(height*.08)):int(height*.25)], axis=0)
        lower = np.mean(gray[int(height*.76):int(height*.94)], axis=0)
        signal = cv2.GaussianBlur((lower-upper).astype(np.float32).reshape(1,-1), (0,0), 2.).ravel()
        low, high = np.percentile(signal, [10,90])
        if high-low < 15.:
            return []
        x = normalized_strip(gray)
        with self._torch.inference_mode():
            p = self._network(self._torch.from_numpy(x)[None]).sigmoid()[0, 0].numpy()
        scale = (gray.shape[1]-1)/719
        margin = max(3, int(np.ceil((height-1)*.2)))
        return [(a*scale, b*scale) for a, b in probability_intervals(p, threshold=self.threshold)
                if a*scale >= margin and b*scale < width-margin
                and height-1 < (b-a)*scale < 4*(height-1)]

    def refine(self, gray: np.ndarray, existing: list[tuple[int,int]], reference_width_px: float) -> list[tuple[int,int]]:
        """Return raw learned proposals; the caller applies the single bounded
        ``refine_existing_intervals`` at the detector boundary.

        The detector re-check already runs :func:`refine_existing_intervals` on
        whatever a refiner returns, so bounding here too would apply the same
        one-to-one correction twice. Returning proposals keeps one canonical
        bound and leaves the detector as the authority over every final endpoint.
        """
        if not existing:
            return []
        return self.predict(gray)
