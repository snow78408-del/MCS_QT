from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from backend.orchestrator.models import SystemConfig
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState


def _service(state: SystemState = SystemState.RUNNING) -> OrchestratorService:
    service = OrchestratorService(vision_service=SimpleNamespace())
    service._cfg = SystemConfig(50.0, 1.0, "file", "sample.mp4", 60.0, 30.0, 500)
    service._state = state
    return service


def test_running_target_can_be_changed_without_restarting_session() -> None:
    service = _service()
    generation = service._lifecycle_generation

    result = service.update_target_diameter(65.5)

    assert result == {
        "previous_target_diameter_um": 50.0,
        "target_diameter_um": 65.5,
        "target_revision": 1,
        "state": "RUNNING",
    }
    assert service._cfg.target_diameter == pytest.approx(65.5)
    assert service._lifecycle_generation == generation


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_target_update_rejects_invalid_values(value: float) -> None:
    service = _service()

    with pytest.raises(ValueError, match="finite and positive"):
        service.update_target_diameter(value)


@pytest.mark.parametrize("state", [SystemState.IDLE, SystemState.INITIALIZING, SystemState.STOPPING, SystemState.ERROR])
def test_target_update_rejects_unsafe_lifecycle_states(state: SystemState) -> None:
    service = _service(state)

    with pytest.raises(RuntimeError, match="state does not allow target update"):
        service.update_target_diameter(60.0)
