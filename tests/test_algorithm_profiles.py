from __future__ import annotations

import json

import pytest

from backend.orchestrator.vision_adapter import PipelineVisionService
from backend.vision.algorithm_profiles import AlgorithmProfileStore, BUILTIN_PROFILE_ID
from backend.vision.algorithms import get_algorithm, list_algorithms


def test_builtin_algorithm_is_registered_and_protected(tmp_path) -> None:
    store = AlgorithmProfileStore(tmp_path / "algorithms.json")

    assert get_algorithm("hybrid_v1") in list_algorithms()
    assert store.active_profile_id == BUILTIN_PROFILE_ID
    assert store.active_profile().protected is True
    with pytest.raises(ValueError, match="只读"):
        store.update_parameters(BUILTIN_PROFILE_ID, {"hough_param2": 10})
    with pytest.raises(ValueError, match="不能删除"):
        store.delete(BUILTIN_PROFILE_ID)


def test_named_algorithm_has_independent_parameters_and_persists(tmp_path) -> None:
    path = tmp_path / "algorithms.json"
    store = AlgorithmProfileStore(path)
    created = store.create("高浓度液滴", "hybrid_v1")
    changed = dict(created.parameters)
    changed["hough_param2"] = 17.0
    store.update_parameters(created.profile_id, changed)
    store.activate(created.profile_id)

    reloaded = AlgorithmProfileStore(path)
    assert reloaded.active_profile().name == "高浓度液滴"
    assert reloaded.active_profile().parameters["hough_param2"] == 17.0
    assert reloaded.get(BUILTIN_PROFILE_ID).parameters["hough_param2"] == 28.0


def test_profile_export_import_creates_a_new_editable_profile(tmp_path) -> None:
    first = AlgorithmProfileStore(tmp_path / "first.json")
    created = first.create("算法 A", "hybrid_v1", {"hough_param2": 19.0})
    exported = tmp_path / "algorithm.json"
    first.export_profile(created.profile_id, exported)

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["profile"]["plugin_id"] == "hybrid_v1"

    second = AlgorithmProfileStore(tmp_path / "second.json")
    imported = second.import_profile(exported)
    assert imported.profile_id != created.profile_id
    assert imported.protected is False
    assert imported.parameters["hough_param2"] == 19.0


def test_runtime_uses_custom_profile_parameters_without_legacy_override() -> None:
    service = PipelineVisionService()
    service.configure_algorithm(
        {
            "name": "新算法",
            "plugin_id": "hybrid_v1",
            "parameters": {"hough_param2": 16.0, "candidate_full_circle_ratio": 0.91},
            "protected": False,
        }
    )
    service.configure_detection_scale(60.0, 1.0)

    pipeline = service._ensure_pipeline()
    assert pipeline.algorithm_id == "hybrid_v1"
    assert pipeline.config.detector.hough_param2 == 16.0
    assert pipeline.config.detector.candidate_full_circle_ratio == 0.91
