from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.orchestrator.feedback_hold import FeedbackHold
from backend.orchestrator.models import RecognitionSnapshot, SystemConfig
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.pid_control.config import PIDConfig
from backend.pid_control.diameter_pid import DiameterPIDController
from backend.pid_control.models import PIDInput
from backend.pump_hardware.models import FlowUpdateResult


def test_only_complete_post_wait_windows_are_eligible() -> None:
    hold = FeedbackHold()
    hold.record(104.0, 10.0, "test")
    assert hold.rejection_reason(110.0, 100.0, 110.0)
    assert hold.rejection_reason(120.0, 110.0, 120.0)
    assert hold.rejection_reason(125.0, None, None)
    assert hold.rejection_reason(125.0, 120.0, 130.0)
    assert not hold.rejection_reason(130.0, 120.0, 130.0)
    assert hold.command_id == 1


def make_service(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("backend.orchestrator.service.time.monotonic", lambda: clock[0])
    writes = []

    def update(q1, q2):
        writes.append((q1, q2))
        clock[0] += 4.0  # verified two-channel transaction takes four seconds
        return FlowUpdateResult(ok=True, q1_ok=True, q2_ok=True, still_running=True)

    pump = SimpleNamespace(
        read_run_state=lambda: SimpleNamespace(ok=True, parsed_reply=object()),
        are_required_channels_running=lambda *_: (True, ""),
        get_current_q_state=lambda: (50.0, 20.0),
        update_flow_while_running=update,
    )
    service = OrchestratorService(vision_service=SimpleNamespace(), pump_service=pump,
                                  pid_config=PIDConfig(control_mode="CLASSIC_PID"))
    service._safety = None
    service._cfg = SystemConfig(55.0, 1.0, "camera", "", 50.0, 20.0, 10_000)
    service._state = SystemState.RUNNING
    service._pump_control_enabled = True
    service._pump_state.comm_established = True
    service._pump_state.q1, service._pump_state.q2 = 50.0, 20.0
    service.disturbance_service = SimpleNamespace(
        build_and_submit_sample=lambda **_: SimpleNamespace(pump_response_delay_ms=0),
        predict=lambda _: None,
    )
    service._refresh_pump_channels = lambda **_: None
    service._update_control_snapshot = lambda value: setattr(service, "_control", value)
    rec = RecognitionSnapshot(
        timestamp=1000, capture_monotonic=100, frame_id=1, control_period_id=1,
        frame_avg_diameter=50, avg_diameter=50, valid_for_control=True,
        frame_droplet_count=3, scale_source="calibration_file",
        total_droplet_count=3, new_crossing_count=3, single_cell_rate=0, reason="",
        droplet_count=3, active_droplet_count=3, has_droplet=True, control_reason="",
        measurement_window_start=90, measurement_window_end=100,
        measurement_sample_start=90,
    )
    service._read_recognition = lambda: rec
    return service, clock, writes, rec


def test_wait_origin_is_verified_completion_and_holds_do_not_integrate(monkeypatch) -> None:
    service, clock, writes, rec = make_service(monkeypatch)
    original = service._pid_controller
    service.run_control_step()
    assert len(writes) == 1
    assert original.integral == 0  # speculative controller did not mutate it
    assert service._pid_controller.integral > 0
    assert service._pid_controller.config is service.pid_config
    assert service._feedback_hold.completed_at == 104
    assert service._feedback_hold.ready_after == 114
    integral = service._pid_controller.integral
    rec.frame_id = rec.control_period_id = 2
    rec.capture_monotonic = clock[0] = 110
    rec.measurement_window_start, rec.measurement_window_end = 100, 110
    service.run_control_step()
    assert len(writes) == 1
    assert service._pid_controller.integral == integral
    rec.capture_monotonic = clock[0] = 120
    rec.frame_id = rec.control_period_id = 3
    rec.measurement_window_start, rec.measurement_window_end = 110, 120
    service.run_control_step()
    assert len(writes) == 1  # mixed window must not be used after timer expires
    rec.capture_monotonic = clock[0] = 130
    rec.frame_id = rec.control_period_id = 4
    rec.measurement_window_start, rec.measurement_window_end = 120, 130
    rec.measurement_sample_start = 120
    service.run_control_step()
    assert len(writes) == 2
    assert service._control.basis_command_id == 1
    assert service._control.command_id == 2


def test_cancelled_command_does_not_commit_pid_state(monkeypatch) -> None:
    service, _, writes, _ = make_service(monkeypatch)
    original = service._pid_controller
    service._update_flow_with_lifecycle_guard = lambda *_: None
    service.run_control_step()
    assert not writes
    assert service._pid_controller is original
    assert original.integral == 0
    assert service._feedback_hold.command_id == 0


@pytest.mark.parametrize("source_start", [None, 110.0])
def test_new_window_with_old_or_unknown_track_measurements_is_held(monkeypatch, source_start) -> None:
    service, clock, writes, rec = make_service(monkeypatch)
    service._feedback_hold.record(104, 10, "test")
    clock[0] = rec.capture_monotonic = 130
    rec.measurement_window_start, rec.measurement_window_end = 120, 130
    rec.measurement_sample_start = source_start
    service.run_control_step()
    assert not writes
    assert service._pid_controller.integral == 0
    assert "来源时间" in service._control.reason


def test_failed_transaction_does_not_commit_pid_and_keeps_stop_path(monkeypatch) -> None:
    service, _, _, _ = make_service(monkeypatch)
    original = service._pid_controller
    service.pump_service.update_flow_while_running = lambda *_: FlowUpdateResult(
        ok=False, q1_ok=True, q2_ok=False, still_running=False,
        safe_stop_verified=True, reason="CH2 verification failed",
    )
    service.run_control_step()
    assert service._pid_controller is original
    assert original.integral == 0
    assert service._feedback_hold.command_id == 0
    assert service._state == SystemState.ERROR
    assert service._stop_event.is_set()


def test_stop_during_transaction_does_not_commit_pid_or_arm_wait(monkeypatch) -> None:
    service, _, _, _ = make_service(monkeypatch)
    original = service._pid_controller

    def stop_during_write(*_):
        service._stop_event.set()
        return FlowUpdateResult(ok=True, q1_ok=True, q2_ok=True, still_running=False)

    service.pump_service.update_flow_while_running = stop_during_write
    service.run_control_step()
    assert service._pid_controller is original
    assert original.integral == 0
    assert service._feedback_hold.command_id == 0


def test_rejected_command_does_not_commit_pid_state(monkeypatch) -> None:
    service, _, writes, _ = make_service(monkeypatch)
    original = service._pid_controller

    def reject(*_):
        raise ValueError("test safety rejection")

    service._require_valid_phase_flows = reject
    service.run_control_step()
    assert not writes
    assert service._pid_controller is original
    assert original.integral == 0


def test_concurrent_control_step_does_not_enter(monkeypatch) -> None:
    service, _, writes, _ = make_service(monkeypatch)
    with service._control_step_lock:
        service.run_control_step()
    assert not writes


def test_wait_uses_existing_validated_dynamics_without_changing_period(monkeypatch) -> None:
    service, _, _, _ = make_service(monkeypatch)
    service._plant_calibration = SimpleNamespace(
        authorized_for_pi=True, conservative_response_delay_ms=2000,
        response_time_constant_ms=3000, response_time_constant_uncertainty_ms=500,
    )
    assert service._post_command_wait()[0] == 12.5
    assert service._cfg.control_interval_ms == 10_000


def test_integration_horizon_does_not_change_derivative_time_base() -> None:
    controller = DiameterPIDController(PIDConfig(control_mode="CLASSIC_PID", base_kd=.01))
    command = controller.update_input(PIDInput(
        target_diameter_um=55, current_diameter_um=50, current_q1=50, current_q2=20,
        dt=100, integration_dt=10, frame_id=1, vision_valid=True,
        pump_communication_ok=True, droplet_count=3,
    ))
    assert not command.freeze_feedback
    assert controller.integral == pytest.approx(50)
    second = controller.update_input(PIDInput(
        target_diameter_um=56, current_diameter_um=50, current_q1=command.q1, current_q2=command.q2,
        dt=100, integration_dt=0, frame_id=2, vision_valid=True,
        pump_communication_ok=True, droplet_count=3,
    ))
    assert second.d_term == pytest.approx(.01 * 1 / 100)
