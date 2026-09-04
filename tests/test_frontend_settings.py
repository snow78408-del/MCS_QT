from __future__ import annotations

import json

from frontend.config import (
    DEFAULT_BO_Q1_RANGE,
    DEFAULT_BO_Q2_RANGE,
    DEFAULT_CONTROL_INTERVAL_MS,
    MIN_CONTROL_INTERVAL_MS,
)
from frontend.settings_store import FrontendSettingsStore


def test_bo_defaults_match_commissioning_flow_envelope() -> None:
    assert DEFAULT_BO_Q1_RANGE == (20.0, 200.0)
    assert DEFAULT_BO_Q2_RANGE == (5.0, 25.0)


def test_realtime_control_period_matches_verified_pump_transaction_budget() -> None:
    assert DEFAULT_CONTROL_INTERVAL_MS == 7500
    assert MIN_CONTROL_INTERVAL_MS == 7500


def test_settings_round_trip(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = FrontendSettingsStore(path)

    store.save({"target_diameter": 60.0, "camera_parameters": {"gain": 2.5}})

    assert store.load() == {"target_diameter": 60.0, "camera_parameters": {"gain": 2.5}}
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 3
    assert not path.with_suffix(".json.tmp").exists()


def test_missing_or_invalid_settings_fall_back_to_empty(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = FrontendSettingsStore(path)
    assert store.load() == {}

    path.write_text("not-json", encoding="utf-8")
    assert store.load() == {}

    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    assert store.load() == {}


def test_legacy_settings_migrate_and_backup_recovers(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"target_diameter": 55.0}), encoding="utf-8")
    store = FrontendSettingsStore(path)
    assert store.load() == {"target_diameter": 55.0}

    store.save({"target_diameter": 60.0})
    store.save({"target_diameter": 65.0})
    path.write_text("corrupt", encoding="utf-8")
    assert store.load() == {"target_diameter": 60.0}


def test_old_observation_width_is_preserved_as_history_not_current_geometry(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "settings": {
                    "recognition_roi": {
                        "enabled": True,
                        "channel_width_um": 430.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    settings = FrontendSettingsStore(path).load()

    roi = settings["recognition_roi"]
    assert roi["channel_width_um"] == 50.0
    assert roi["generation_channel_height_um"] == 50.0
    assert roi["generation_channel_width_um"] == 50.0
    assert roi["previous_channel_width_um"] == 430.0


def test_user_square_dimension_is_kept_as_current_geometry(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "settings": {
                    "recognition_roi": {
                        "channel_width_um": 70.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    roi = FrontendSettingsStore(path).load()["recognition_roi"]

    assert roi["channel_width_um"] == 70.0
    assert roi["generation_channel_height_um"] == 70.0
    assert roi["generation_channel_width_um"] == 70.0
    assert "previous_channel_width_um" not in roi


def test_explicit_generation_width_and_depth_remain_independent(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "settings": {
                    "recognition_roi": {
                        "channel_width_um": 70.0,
                        "generation_channel_height_um": 45.0,
                        "generation_channel_width_um": 70.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    roi = FrontendSettingsStore(path).load()["recognition_roi"]

    assert roi["channel_width_um"] == 70.0
    assert roi["generation_channel_height_um"] == 45.0
    assert roi["generation_channel_width_um"] == 70.0
    assert "previous_channel_width_um" not in roi


def test_mixed_430_height_and_50_width_resolves_to_current_50_square(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "settings": {
                    "recognition_roi": {
                        "channel_width_um": 430.0,
                        "generation_channel_height_um": 430.0,
                        "generation_channel_width_um": 50.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    roi = FrontendSettingsStore(path).load()["recognition_roi"]

    assert roi["generation_channel_height_um"] == 50.0
    assert roi["generation_channel_width_um"] == 50.0
    assert roi["previous_channel_width_um"] == 430.0


def test_user_entered_430_in_current_schema_remains_current(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "settings": {
                    "recognition_roi": {
                        "channel_width_um": 430.0,
                        "generation_channel_height_um": 430.0,
                        "generation_channel_width_um": 430.0,
                        "previous_channel_width_um": 50.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    roi = FrontendSettingsStore(path).load()["recognition_roi"]

    assert roi["channel_width_um"] == 430.0
    assert roi["generation_channel_height_um"] == 430.0
    assert roi["generation_channel_width_um"] == 430.0
    assert roi["previous_channel_width_um"] == 50.0
