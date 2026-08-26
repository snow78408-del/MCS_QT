from __future__ import annotations

import pytest

from backend.orchestrator.models import SystemConfig
from backend.vision.calibration import CalibrationRecord


def _record() -> dict:
    return {
        "schema_version": 1,
        "calibration_id": "cal-2026-08-26-a",
        "created_at": "2026-08-26T10:00:00+08:00",
        "magnification": "20x",
        "view_id": "chip-a/inlet",
        "pixel_to_micron": 1.25,
        "uncertainty_um_per_px": 0.02,
        "calibration_image_sha256": "a" * 64,
        "cross_view_cv_percent": 1.8,
    }


def test_versioned_calibration_requires_provenance_and_uncertainty() -> None:
    record = CalibrationRecord.from_mapping(_record())
    assert record.schema_version == 1
    assert record.uncertainty_um_per_px == 0.02


def test_runtime_scale_must_agree_with_calibration_record() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        SystemConfig(
            target_diameter=50.0,
            pixel_to_micron=2.0,
            video_source_type="video",
            video_source="sample.mp4",
            initial_q1=50.0,
            initial_q2=20.0,
            control_interval_ms=500,
            calibration=_record(),
        )
