from __future__ import annotations

import json
from dataclasses import replace

import pytest

from backend.orchestrator.config import OrchestratorConfig
from backend.orchestrator.response_guard import ResponseGuard
from backend.orchestrator.state import SystemState
from test_feedback_hold import make_service


def observe(guard, at, diameter=50.0, **changes):
    values = dict(now=at, start=at-2, end=at-1, diameter=diameter,
                  std=0.2, tolerance=4.0, recovery_s=15, timeout_s=120,
                  error=10.0, deadband=1.0, max_attempts=3)
    values.update(changes)
    return guard.observe(**values)


def test_no_response_is_normal_but_successful_attempts_are_bounded():
    guard = ResponseGuard()
    assert not observe(guard, 10).reason
    for i in range(3):
        guard.record_command(50, 0.5)
        decision = observe(guard, 20 + 10*i)
        assert guard.unresponsive_commands == i + 1
        assert bool(decision.reason) == (i == 2)
    assert not decision.stop
    assert not observe(guard, 60, diameter=55).reason  # a late response can clear the hold
    assert guard.unresponsive_commands == 0


def test_repeated_observation_cannot_count_another_attempt():
    guard = ResponseGuard()
    guard.record_command(50, 1)
    observe(guard, 10)
    observe(guard, 10)
    observe(guard, 20)
    assert guard.unresponsive_commands == 1


def test_instability_requires_fresh_sustained_recovery_and_times_out():
    guard = ResponseGuard()
    assert "不稳定" in observe(guard, 10, std=4).reason
    assert observe(guard, 11).reason  # mixed window
    assert observe(guard, 20).reason
    assert observe(guard, 30).reason
    decision = observe(guard, 40)
    assert not decision.reason and decision.recovering
    guard.record_command(50, 1)
    assert not guard.recovery_pending
    observe(guard, 50, std=4)
    assert observe(guard, 170).stop


def test_drift_and_invalid_measurements_do_not_prove_air():
    guard = ResponseGuard()
    observe(guard, 10, 50)
    observe(guard, 20, 53)
    assert "漂移" in observe(guard, 30, 56).reason
    assert "测量无效" in observe(guard, 40, std=float("nan")).reason
    assert guard.recovery_pending


def advance(rec, clock, at, diameter=50):
    rec.capture_monotonic = clock[0] = at
    rec.frame_id += 1
    rec.control_period_id += 1
    rec.frame_avg_diameter = rec.avg_diameter = diameter
    rec.measurement_window_start, rec.measurement_window_end = at-5, at
    rec.measurement_sample_start, rec.measurement_sample_end = at-5, at-1


def test_unstable_control_never_writes_or_integrates_and_recovers(monkeypatch):
    service, clock, writes, rec = make_service(monkeypatch)
    rec.frame_diameter_std = 4
    original = service._pid_controller
    original.integral = 3
    service.run_control_step()
    assert not writes and original.integral == 3
    assert "不稳定" in service._control.reason
    rec.frame_diameter_std = 0.2
    for at in (110, 120):
        advance(rec, clock, at)
        service.run_control_step()
        assert not writes and original.integral == 3
    advance(rec, clock, 130)
    service.run_control_step()
    assert len(writes) == 1
    assert service._pid_controller.integral == 0  # no catch-up integration on recovery


def test_timeout_uses_stop_path_even_when_vision_remains_invalid(monkeypatch):
    service, clock, writes, rec = make_service(monkeypatch)
    stopped = []
    service._safety_stop_pump = lambda: stopped.append(True) or True
    rec.valid_for_control = False
    service.run_control_step()
    clock[0] += service.runtime.response_guard_timeout_s
    service.run_control_step()
    assert stopped and not writes
    assert service._state == SystemState.ERROR
    assert service._control.suggested_stop


def test_deadband_avoids_redundant_pump_transaction(monkeypatch):
    service, _, writes, rec = make_service(monkeypatch)
    rec.frame_avg_diameter = 55
    service.run_control_step()
    assert not writes and service._pid_controller.integral == 0
    assert "容差" in service._control.reason


def test_cancelled_retry_does_not_consume_flow_budget(monkeypatch):
    service, _, writes, _ = make_service(monkeypatch)
    guard = service._response_guard
    guard.record_command(50, 1)
    service._update_flow_with_lifecycle_guard = lambda *_: None
    service.run_control_step()
    assert not writes
    assert guard.cumulative_change == 0
    assert guard.unresponsive_commands == 1
    assert service._pid_controller.integral == 0


def test_control_stops_chasing_after_three_unresponsive_commands(monkeypatch):
    service, clock, writes, rec = make_service(monkeypatch)
    for at in (100, 130, 160):
        advance(rec, clock, at)
        service.run_control_step()
    assert len(writes) == 3
    for q1, q2 in writes[1:]:
        assert abs(q1 - 50) <= 1 and abs(q2 - 20) <= 1
    assert service._response_guard.cumulative_change <= 4
    integral = service._pid_controller.integral
    advance(rec, clock, 190)
    service.run_control_step()
    assert len(writes) == 3
    assert service._pid_controller.integral == integral
    assert "持续未检测到响应" in service._control.reason
    assert service._state == SystemState.RUNNING


def test_retry_budget_rejects_before_hardware_and_pid_commit(monkeypatch):
    service, _, writes, _ = make_service(monkeypatch)
    service._response_guard.record_command(50, 1)
    service._response_guard.cumulative_change = 4
    original = service._pid_controller
    service.run_control_step()
    assert not writes and service._pid_controller is original
    assert "预算" in service._control.reason


def test_pause_generation_requires_new_recovery_without_resetting_attempts(monkeypatch):
    service, clock, writes, rec = make_service(monkeypatch)
    service.run_control_step()
    service._response_guard.unresponsive_commands = 1
    service._lifecycle_generation += 1
    advance(rec, clock, 140)
    service.run_control_step()
    assert len(writes) == 1
    assert service._response_guard.unresponsive_commands == 1
    assert "全新" in service._control.reason


def test_cancelled_recovery_preserves_original_integral(monkeypatch):
    service, clock, writes, rec = make_service(monkeypatch)
    service._pid_controller.integral = 3
    original = service._pid_controller
    service._response_guard.invalidate(90)
    service._update_flow_with_lifecycle_guard = lambda *_: None
    for at in (100, 110, 120):
        advance(rec, clock, at)
        service.run_control_step()
    assert not writes and service._pid_controller is original
    assert original.integral == 3
    assert service._response_guard.recovery_pending


def test_unstable_calibration_deadline_preserves_diagnostic(monkeypatch, tmp_path):
    service, clock, _, _ = make_service(monkeypatch)
    monkeypatch.setattr("backend.orchestrator.service.ensure_user_subdir", lambda _: tmp_path)
    with pytest.raises(RuntimeError, match="工况持续不稳定"):
        service._check_calibration_deadline(deadline=clock[0], phase="validation", observations=[])
    payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete" and "record" not in payload


@pytest.mark.parametrize("values", [
    {"response_guard_timeout_s": float("inf")}, {"response_guard_max_attempts": 0},
    {"response_guard_max_attempts": True}, {"response_guard_retry_step": -1},
    {"response_guard_timeout_s": 10},
])
def test_invalid_guard_policy_rejected(values):
    with pytest.raises(ValueError):
        replace(OrchestratorConfig(), **values)
