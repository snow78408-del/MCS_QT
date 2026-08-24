from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _pixel_line(line: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.array(
        [float(line.get("x1", 0.0)) * width, float(line.get("y1", 0.0)) * height],
        dtype=np.float32,
    )
    p2 = np.array(
        [float(line.get("x2", 1.0)) * width, float(line.get("y2", 0.0)) * height],
        dtype=np.float32,
    )
    return p1, p2


def wall_line_quad(
    width: int,
    height: int,
    wall_lines: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[np.ndarray, int, int] | None:
    if width < 2 or height < 2 or len(wall_lines) != 2:
        return None
    raw = [_pixel_line(dict(line), width, height) for line in wall_lines]
    directions: list[np.ndarray] = []
    for p1, p2 in raw:
        direction = p2 - p1
        norm = float(np.linalg.norm(direction))
        if norm < 20.0:
            return None
        direction /= norm
        if directions and float(np.dot(direction, directions[0])) < 0.0:
            p1, p2 = p2, p1
            direction = -direction
        directions.append(direction)
    common = directions[0] + directions[1]
    common_norm = float(np.linalg.norm(common))
    if common_norm < 1e-6:
        return None
    common /= common_norm
    normal = np.array([-common[1], common[0]], dtype=np.float32)

    ordered: list[tuple[np.ndarray, np.ndarray]] = []
    for (p1, p2), direction in zip(raw, directions):
        if float(np.dot(direction, common)) < 0.0:
            p1, p2 = p2, p1
        ordered.append((p1, p2))
    ordered.sort(key=lambda pair: float(np.dot((pair[0] + pair[1]) * 0.5, normal)))
    first, second = ordered

    first_range = sorted((float(np.dot(first[0], common)), float(np.dot(first[1], common))))
    second_range = sorted((float(np.dot(second[0], common)), float(np.dot(second[1], common))))
    start = max(first_range[0], second_range[0])
    end = min(first_range[1], second_range[1])
    if end - start < 30.0:
        start = min(first_range[0], second_range[0])
        end = max(first_range[1], second_range[1])

    def point_at(pair: tuple[np.ndarray, np.ndarray], target: float) -> np.ndarray:
        p1, p2 = pair
        direction = p2 - p1
        denom = float(np.dot(direction, common))
        if abs(denom) < 1e-6:
            return p1.copy()
        return p1 + direction * ((target - float(np.dot(p1, common))) / denom)

    a0, a1 = point_at(first, start), point_at(first, end)
    b0, b1 = point_at(second, start), point_at(second, end)
    source = np.array([a0, a1, b1, b0], dtype=np.float32)
    output_width = max(32, int(round(0.5 * (np.linalg.norm(a1 - a0) + np.linalg.norm(b1 - b0)))))
    output_height = max(16, int(round(0.5 * (np.linalg.norm(b0 - a0) + np.linalg.norm(b1 - a1)))))
    if output_width > width * 2 or output_height > height * 2:
        return None
    return source, output_width, output_height


def rectify_channel_frame(
    frame: np.ndarray,
    wall_lines: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> np.ndarray | None:
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    height, width = frame.shape[:2]
    geometry = wall_line_quad(width, height, wall_lines)
    if geometry is None:
        return None
    source, output_width, output_height = geometry
    destination = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        frame,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def wall_separation_px(
    width: int,
    height: int,
    wall_lines: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> float | None:
    geometry = wall_line_quad(width, height, wall_lines)
    return None if geometry is None else float(geometry[2])


def wall_lines_bbox(
    wall_lines: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    margin_ratio: float = 0.01,
) -> dict[str, float] | None:
    if len(wall_lines) != 2:
        return None
    xs = [float(line[key]) for line in wall_lines for key in ("x1", "x2")]
    ys = [float(line[key]) for line in wall_lines for key in ("y1", "y2")]
    margin = max(0.0, float(margin_ratio))
    return {
        "x_start_ratio": max(0.0, min(xs) - margin),
        "y_start_ratio": max(0.0, min(ys) - margin),
        "x_end_ratio": min(1.0, max(xs) + margin),
        "y_end_ratio": min(1.0, max(ys) + margin),
    }
