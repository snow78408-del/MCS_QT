from __future__ import annotations

import json

from frontend.settings_store import FrontendSettingsStore


def test_settings_round_trip(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = FrontendSettingsStore(path)

    store.save({"target_diameter": 60.0, "camera_parameters": {"gain": 2.5}})

    assert store.load() == {"target_diameter": 60.0, "camera_parameters": {"gain": 2.5}}
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
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
