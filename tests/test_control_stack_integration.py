from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from backend.optimization import BayesianOptimizationConfig, OperatingPoint, OptimizationStatus
from backend.orchestrator.models import SystemConfig
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.pid_control import DiameterPIDController, PIDConfig, PIDInput


def _bo_config(**overrides) -> BayesianOptimizationConfig:
    values = {
        "target_diameter_um": 60.0,
        "q1_min": 20.0,
        "q1_max": 60.0,
        "q2_min": 10.0,
        "q2_max": 25.0,
        "measured_response_delay_ms": 1200.0,
        "settling_time_ms": 2000.0,
        "response_delay_source": "step experiment 2026-08-27",
        "response_delay_uncertainty_ms": 200.0,
    }
    values.update(overrides)
    return BayesianOptimizationConfig(**values)


def test_start_optimization_registers_the_only_active_writer_and_measured_delay() -> None:
    service = object.__new__(OrchestratorService)
    service._lock = threading.RLock()
    service._cfg = SystemConfig(60.0, 1.0, "camera", "0", 50.0, 20.0, 500)
    service.pid_config = PIDConfig()
    service._pump_state = SimpleNamespace(
        pump_response_delay_ms=None,
        pump_response_measurement_status="unmeasured",
    )
    captured = []
    service.start_with_mode = captured.append

    service.start_optimization(_bo_config())

    assert captured == [SystemState.OPTIMIZING]
    assert service._optimizer is not None
    assert service._pump_state.pump_response_delay_ms == 1400.0
    assert service._pump_state.pump_response_measurement_status.startswith("step experiment")


def test_start_optimization_rejects_bounds_weaker_than_pid_safety() -> None:
    service = object.__new__(OrchestratorService)
    service._lock = threading.RLock()
    service._cfg = SystemConfig(60.0, 1.0, "camera", "0", 50.0, 20.0, 500)
    service.pid_config = PIDConfig(q1_min=1.0)
    service._pump_state = SimpleNamespace()

    with pytest.raises(ValueError, match="Q1 bounds"):
        service.start_optimization(_bo_config(q1_min=0.5))


def test_start_optimization_is_disabled_for_local_video() -> None:
    service = object.__new__(OrchestratorService)
    service._lock = threading.RLock()
    service._cfg = SystemConfig(60.0, 1.0, "file", "sample.mp4", 50.0, 20.0, 500)
    service.pid_config = PIDConfig()
    service._pump_state = SimpleNamespace()

    with pytest.raises(RuntimeError, match="realtime"):
        service.start_optimization(_bo_config())


def test_stabilizing_handoff_resets_pid_at_bo_operating_point() -> None:
    best = OperatingPoint(52.0, 18.0, 60.1, None, 2.0, 0.01, 5)
    status = OptimizationStatus(completed=True, best_operating_point=best)
    service = object.__new__(OrchestratorService)
    service._lock = threading.RLock()
    service._state = SystemState.STABILIZING
    service._message = ""
    service._error = ""
    service._log = lambda _message: None
    service._optimizer = SimpleNamespace(
        status=lambda: status,
        config=SimpleNamespace(minimum_valid_droplets=2),
    )
    service._stabilizing_until_monotonic = 0.0
    service._pump_state = SimpleNamespace(q1_actual=52.0, q2_actual=18.0)
    service._pid_controller = DiameterPIDController(PIDConfig())
    service._last_control_frame_id = None
    service._last_control_period_id = None
    service._last_control_ts = None
    service._cfg = SystemConfig(60.0, 1.0, "file", "sample.mp4", 50.0, 20.0, 500)
    snapshots = []
    service._update_control_snapshot = snapshots.append
    rec = SimpleNamespace(valid_for_control=True, frame_diameters=[59.9, 60.1], frame_id=7, control_period_id=3)

    service._run_stabilizing_step(rec=rec, now=10.0, monotonic_now=20.0)

    assert service._state == SystemState.RUNNING
    assert service._pid_controller.operating_point == (52.0, 18.0)
    assert snapshots[-1].control_owner == "HOLD"

    command = service._pid_controller.update_input(
        PIDInput(60.0, 60.0, 52.0, 18.0, 0.5, 8, True, True)
    )
    assert command.q1 == pytest.approx(52.0)
    assert command.q2 == pytest.approx(18.0)
    assert command.control_owner == "PID"


def test_same_invalid_vision_period_is_counted_only_once_by_bo() -> None:
    rejected = []
    optimizer = SimpleNamespace(
        config=SimpleNamespace(settling_time_ms=100.0, candidate_timeout_ms=5000.0),
        reject_current=rejected.append,
        status=lambda: SimpleNamespace(failed=False, reason=""),
    )
    service = object.__new__(OrchestratorService)
    service._lock = threading.RLock()
    service._optimizer = optimizer
    service._optimization_candidate = SimpleNamespace(candidate_id=1)
    service._state = SystemState.OPTIMIZING
    service._optimization_candidate_applied_monotonic = 10.0
    service._optimization_candidate_period_id = 3
    service._log = lambda _message: None

    service._reject_optimization_window_if_due("invalid vision", 11.0, 4)
    service._reject_optimization_window_if_due("invalid vision", 11.1, 4)
    service._reject_optimization_window_if_due("invalid vision", 11.2, 5)

    assert rejected == ["invalid vision", "invalid vision"]
