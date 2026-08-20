from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

try:
    from .config import BeadConfig, DebugConfig
    from .tracker import DropletTrack
except ImportError:
    from config import BeadConfig, DebugConfig
    from tracker import DropletTrack


@dataclass
class DropletBead:
    droplet_id: int
    bead_positions: List[np.ndarray] = field(default_factory=list)
    bead_count: int = 0


@dataclass
class BeadResult:
    droplets: List[DropletBead]
    total_beads: int
    debug_image: np.ndarray
    candidate_mask: np.ndarray


class BeadCounter:
    def __init__(self, config: BeadConfig, debug: DebugConfig) -> None:
        self._config = config
        self._debug = debug

    def count(
        self,
        active_droplets: Sequence[DropletTrack],
        gray_frame: np.ndarray,
        helper_mask: Optional[np.ndarray] = None,
    ) -> BeadResult:
        gray = self._ensure_gray(gray_frame)
        candidate_mask = self._build_candidate_mask(gray, helper_mask)

        if self._config.mode == "connected":
            droplets = self._count_connected_mode(active_droplets, candidate_mask)
        else:
            droplets = self._count_intensity_mode(active_droplets, gray)

        total_beads = 0
        for droplet in droplets:
            total_beads += droplet.bead_count

        return BeadResult(
            droplets=droplets,
            total_beads=total_beads,
            debug_image=np.empty((0, 0, 3), dtype=np.uint8),
            candidate_mask=candidate_mask,
        )

    def _count_intensity_mode(
        self,
        active_droplets: Sequence[DropletTrack],
        gray: np.ndarray,
    ) -> List[DropletBead]:
        droplets: List[DropletBead] = []
        for track in active_droplets:
            radius = (
                float(track.radius)
                if float(track.radius) > 0.0
                else float(self._config.default_droplet_radius)
            )
            mask = self._circle_mask(gray.shape[:2], track.position, radius * self._config.inner_radius_ratio)

            roi_values = gray[mask > 0]
            if roi_values.size == 0:
                droplets.append(DropletBead(droplet_id=track.id))
                continue

            threshold = float(np.percentile(roi_values, self._config.dark_percentile))
            local_binary = np.zeros_like(gray, dtype=np.uint8)
            local_binary[(gray <= threshold) & (mask > 0)] = 255

            blur_k = self._odd(self._config.blur_kernel)
            local_binary = cv2.GaussianBlur(local_binary, (blur_k, blur_k), 0)
            _, local_binary = cv2.threshold(local_binary, 127, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_OPEN, kernel)

            droplets.append(
                DropletBead(
                    droplet_id=track.id,
                    bead_positions=self._extract_beads_from_binary(local_binary, track.position, radius),
                )
            )
            droplets[-1].bead_count = len(droplets[-1].bead_positions)

        return droplets

    def _count_connected_mode(
        self,
        active_droplets: Sequence[DropletTrack],
        candidate_mask: np.ndarray,
    ) -> List[DropletBead]:
        droplet_map: Dict[int, DropletBead] = {
            track.id: DropletBead(droplet_id=track.id) for track in active_droplets
        }

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (candidate_mask > 0).astype(np.uint8),
            8,
            cv2.CV_32S,
        )

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self._config.area_min or area > self._config.area_max:
                continue

            ys, xs = np.where(labels == label)
            if xs.size == 0:
                continue

            points = np.column_stack((xs, ys)).astype(np.float32)
            matched_id: Optional[int] = None

            for track in active_droplets:
                radius = (
                    float(track.radius)
                    if float(track.radius) > 0.0
                    else float(self._config.default_droplet_radius)
                )
                inner_radius = max(1.0, radius * self._config.inner_radius_ratio)
                dx = points[:, 0] - float(track.position[0])
                dy = points[:, 1] - float(track.position[1])
                if np.all((dx * dx + dy * dy) <= ((inner_radius - self._config.border_margin) ** 2)):
                    matched_id = track.id
                    break

            if matched_id is None:
                continue

            cx, cy = centroids[label]
            droplet_map[matched_id].bead_positions.append(np.array([cx, cy], dtype=np.float32))

        droplets = list(droplet_map.values())
        for droplet in droplets:
            droplet.bead_count = len(droplet.bead_positions)
        return droplets

    def _extract_beads_from_binary(
        self,
        binary: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> List[np.ndarray]:
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8),
            8,
            cv2.CV_32S,
        )

        beads: List[np.ndarray] = []
        inner_radius = max(1.0, radius * self._config.inner_radius_ratio)
        max_dist2 = (inner_radius - self._config.border_margin) ** 2

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self._config.area_min or area > self._config.area_max:
                continue

            cx, cy = centroids[label]
            dx = cx - float(center[0])
            dy = cy - float(center[1])
            if dx * dx + dy * dy > max_dist2:
                continue

            beads.append(np.array([cx, cy], dtype=np.float32))

        return beads

    def _build_candidate_mask(self, gray: np.ndarray, helper_mask: Optional[np.ndarray]) -> np.ndarray:
        if helper_mask is not None and helper_mask.size > 0:
            return (helper_mask > 0).astype(np.uint8) * 255

        threshold = float(np.percentile(gray, self._config.dark_percentile))
        _, candidate = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        return candidate

    def _circle_mask(self, shape: tuple, center: np.ndarray, radius: float) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (int(center[0]), int(center[1])), int(max(1, radius)), 255, -1)
        return mask

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _odd(self, value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1
