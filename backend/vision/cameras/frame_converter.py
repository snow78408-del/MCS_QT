from __future__ import annotations

import time
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None

from .models import FrameData


def normalize_to_uint8(raw: np.ndarray, bits: int = 8) -> np.ndarray:
    if np is None:
        raise ValueError("图像转换需要安装 numpy")
    arr = np.asarray(raw)
    if arr.dtype == np.uint8 and bits <= 8:
        return arr
    max_value = float((1 << int(bits)) - 1)
    if max_value <= 0:
        max_value = float(np.max(arr) or 1)
    scaled = np.clip((arr.astype(np.float32) / max_value) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def convert_image(raw: Any, width: int, height: int, pixel_format: str) -> np.ndarray:
    fmt = str(pixel_format or "").strip()
    normalized = fmt.replace(" ", "").replace("_", "").upper()
    arr = np.asarray(raw)

    if normalized in {"MONO8", "MONO"}:
        return arr.reshape((height, width)).astype(np.uint8, copy=False)
    if normalized in {"MONO10", "MONO10PACKED"}:
        return normalize_to_uint8(arr.reshape((height, width)), bits=10)
    if normalized in {"MONO12", "MONO12PACKED"}:
        return normalize_to_uint8(arr.reshape((height, width)), bits=12)

    if normalized in {"RGB8", "RGB8PACKED"}:
        rgb = arr.reshape((height, width, 3)).astype(np.uint8, copy=False)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if normalized in {"BGR8", "BGR8PACKED"}:
        return arr.reshape((height, width, 3)).astype(np.uint8, copy=False)

    bayer_codes = {
        "BAYERRG8": getattr(cv2, "COLOR_BayerRG2BGR", None) if cv2 is not None else None,
        "BAYERGB8": getattr(cv2, "COLOR_BayerGB2BGR", None) if cv2 is not None else None,
        "BAYERGR8": getattr(cv2, "COLOR_BayerGR2BGR", None) if cv2 is not None else None,
        "BAYERBG8": getattr(cv2, "COLOR_BayerBG2BGR", None) if cv2 is not None else None,
    }
    if normalized in bayer_codes:
        if cv2 is None or bayer_codes[normalized] is None:
            raise ValueError("Bayer图像转换需要安装 opencv-contrib-python")
        bayer = arr.reshape((height, width)).astype(np.uint8, copy=False)
        return cv2.cvtColor(bayer, bayer_codes[normalized])

    if normalized.startswith("YUV") or normalized in {"YUYV", "YUY2"}:
        if cv2 is None:
            raise ValueError("YUV图像转换需要安装 opencv-contrib-python")
        yuv = arr.reshape((height, width, 2)).astype(np.uint8, copy=False)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)

    if arr.ndim == 2:
        return normalize_to_uint8(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8, copy=False)
    raise ValueError(f"不支持的像素格式: {pixel_format}")


def make_frame_data(
    raw: Any,
    width: int,
    height: int,
    pixel_format: str,
    frame_id: int,
    source_backend: str,
    device_unique_id: str,
    timestamp: float | None = None,
) -> FrameData:
    host_monotonic = time.monotonic()
    image = convert_image(raw, int(width), int(height), pixel_format)
    return FrameData(
        image=image,
        frame_id=int(frame_id),
        timestamp=float(timestamp or time.time()),
        width=int(image.shape[1]) if image is not None else int(width),
        height=int(image.shape[0]) if image is not None else int(height),
        pixel_format=str(pixel_format or ""),
        source_backend=source_backend,
        device_unique_id=device_unique_id,
        valid=image is not None and int(image.size) > 0,
        host_monotonic_timestamp=host_monotonic,
    )
