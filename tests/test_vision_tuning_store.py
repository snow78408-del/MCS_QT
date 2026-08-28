from __future__ import annotations

import json

from backend.vision.config import ChannelRegionConfig, DetectorConfig
from frontend.vision_tuning_store import (
    TuningLoadStatus,
    VISION_TUNING_SCHEMA_VERSION,
    VisionTuningSettingsStore,
)


def test_first_load_creates_user_parameter_file_with_defaults(tmp_path) -> None:
    path = tmp_path / "config" / "vision_tuning_parameters.json"
    store = VisionTuningSettingsStore(path)

    result = store.load_or_create()

    assert result.status is TuningLoadStatus.CREATED
    assert result.detector == DetectorConfig()
    assert result.channel_region == ChannelRegionConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == VISION_TUNING_SCHEMA_VERSION
    assert payload["detector"]["sensitivity"] == DetectorConfig().sensitivity


def test_save_replaces_previous_algorithm_parameters(tmp_path) -> None:
    path = tmp_path / "vision_tuning_parameters.json"
    store = VisionTuningSettingsStore(path)
    store.load_or_create()

    detector = DetectorConfig(sensitivity=0.72, min_radius=20.0)
    channel = ChannelRegionConfig(canny_low=40)
    store.save(detector, channel)

    result = store.load_or_create()
    assert result.status is TuningLoadStatus.LOADED
    assert result.detector.sensitivity == 0.72
    assert result.detector.min_radius == 20.0
    assert result.channel_region.canny_low == 40
    assert not path.with_suffix(".json.tmp").exists()


def test_wrong_schema_or_fields_are_reported_without_overwriting_old_file(tmp_path) -> None:
    path = tmp_path / "vision_tuning_parameters.json"
    old_payload = {"schema_version": 0, "detector": {}, "channel_region": {}}
    path.write_text(json.dumps(old_payload), encoding="utf-8")
    store = VisionTuningSettingsStore(path)

    result = store.load_or_create()

    assert result.status is TuningLoadStatus.INVALID
    assert json.loads(path.read_text(encoding="utf-8")) == old_payload

    valid_payload = {
        "schema_version": VISION_TUNING_SCHEMA_VERSION,
        "detector": vars(DetectorConfig()),
        "channel_region": vars(ChannelRegionConfig()),
    }
    del valid_payload["detector"]["sensitivity"]
    path.write_text(json.dumps(valid_payload), encoding="utf-8")
    assert store.load_or_create().status is TuningLoadStatus.INVALID


def test_delete_old_parameters_and_recreate_defaults(tmp_path) -> None:
    path = tmp_path / "vision_tuning_parameters.json"
    path.write_text("old-format", encoding="utf-8")
    store = VisionTuningSettingsStore(path)

    detector, channel = store.delete_and_create_defaults()

    assert detector == DetectorConfig()
    assert channel == ChannelRegionConfig()
    assert store.load_or_create().status is TuningLoadStatus.LOADED
