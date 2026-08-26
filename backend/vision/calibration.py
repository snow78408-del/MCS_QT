from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    schema_version: int
    calibration_id: str
    created_at: str
    magnification: str
    view_id: str
    pixel_to_micron: float
    uncertainty_um_per_px: float
    calibration_image_sha256: str
    cross_view_cv_percent: float | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "CalibrationRecord":
        record = cls(
            schema_version=int(values.get("schema_version", 0)),
            calibration_id=str(values.get("calibration_id", "")).strip(),
            created_at=str(values.get("created_at", "")).strip(),
            magnification=str(values.get("magnification", "")).strip(),
            view_id=str(values.get("view_id", "")).strip(),
            pixel_to_micron=float(values.get("pixel_to_micron", 0.0)),
            uncertainty_um_per_px=float(values.get("uncertainty_um_per_px", -1.0)),
            calibration_image_sha256=str(values.get("calibration_image_sha256", "")).lower().strip(),
            cross_view_cv_percent=(
                None
                if values.get("cross_view_cv_percent") is None
                else float(values["cross_view_cv_percent"])
            ),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported calibration schema_version; expected 1")
        if not self.calibration_id or not self.magnification or not self.view_id:
            raise ValueError("calibration_id, magnification and view_id are required")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("calibration created_at must be ISO-8601") from exc
        if not math.isfinite(self.pixel_to_micron) or self.pixel_to_micron <= 0.0:
            raise ValueError("calibration pixel_to_micron must be finite and positive")
        if not math.isfinite(self.uncertainty_um_per_px) or self.uncertainty_um_per_px < 0.0:
            raise ValueError("calibration uncertainty_um_per_px must be finite and non-negative")
        digest = self.calibration_image_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("calibration_image_sha256 must be a 64-character SHA-256 digest")
        if self.cross_view_cv_percent is not None and (
            not math.isfinite(self.cross_view_cv_percent) or self.cross_view_cv_percent < 0.0
        ):
            raise ValueError("cross_view_cv_percent must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def image_sha256(image_path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(image_path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def load_calibration(path: str | Path) -> CalibrationRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration file root must be an object")
    return CalibrationRecord.from_mapping(payload)
