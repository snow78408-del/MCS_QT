from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.orchestrator.models import ControlSnapshot, SystemConfig
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.pid_control.diameter_pid import DiameterPIDController
from backend.pid_control.models import PIDCommand
from backend.pid_control.mpc import DiameterMPCController
from backend.pump_hardware.models import FlowUpdateResult
from test_feedback_hold import make_service
from test_mpc_controller import record


def test_configure_selects_mpc_and_can_return_to_pi():
    service = OrchestratorService(vision_service=SimpleNamespace(), pump_service=SimpleNamespace())
    cfg = SystemConfig(62, 1, "file", "unused.mp4", 50, 20, 7500,
                       feedback_controller="MPC", plant_calibration=record().to_dict())
    service.configure(cfg)
    assert isinstance(service._pid_controller, DiameterMPCController)
    assert service.pid_config.control_mode == "MPC"
    assert service.get_snapshot().config.feedback_controller == "MPC"
    service.configure(replace(cfg, feedback_controller="PID"))
    assert type(service._pid_controller) is DiameterPIDController
    assert service.pid_config.control_mode == "CLASSIC_PID"


@pytest.mark.parametrize("calibration", [{}, record(validated_for_mpc=False).to_dict()])
def test_unqualified_mpc_is_rejected_before_configuration(calibration):
    service = OrchestratorService(vision_service=SimpleNamespace(), pump_service=SimpleNamespace())
    with pytest.raises(ValueError, match="MPC"):
        service.configure(SystemConfig(62, 1, "file", "unused.mp4", 50, 20, 7500,
                          feedback_controller="MPC", plant_calibration=calibration))
    assert service._state == SystemState.IDLE


def test_waiting_snapshot_preserves_mpc_identity():
    service = OrchestratorService(vision_service=SimpleNamespace(), pump_service=SimpleNamespace())
    service.configure(SystemConfig(62, 1, "file", "unused.mp4", 50, 20, 7500,
                      feedback_controller="MPC", plant_calibration=record().to_dict()))
    service._update_control_snapshot(ControlSnapshot(0, 0, 50, 20, True, False, "waiting", 0))
    assert service._control.control_owner == service._control.control_mode == "MPC"


def mpc_service(monkeypatch):
    service, clock, writes, rec = make_service(monkeypatch)
    service._plant_calibration = record()
    service.pid_config.control_mode = "MPC"
    service._pid_controller = DiameterMPCController(service.pid_config, record(), 10)
    rec.avg_diameter = rec.frame_avg_diameter = 61
    service._cfg.target_diameter = 64
    return service, clock, writes, rec


def test_success_commits_mpc_state_and_preserves_feedback_wait(monkeypatch):
    service, clock, writes, rec = mpc_service(monkeypatch)
    original = service._pid_controller
    service.run_control_step()
    assert len(writes) == 1
    assert original.offset_um == 0 and original.last_plan == ()
    assert service._pid_controller.offset_um > 0
    assert service._control.control_owner == "MPC"
    assert len(service._control.mpc_prediction_um) == 10
    assert service._feedback_hold.completed_at == 104
    assert service._feedback_hold.ready_after == pytest.approx(108.7)
    committed = service._pid_controller
    rec.frame_id = rec.control_period_id = 2
    clock[0] = rec.capture_monotonic = 106
    service.run_control_step()
    assert len(writes) == 1 and service._pid_controller is committed


@pytest.mark.parametrize("outcome", ["cancel", "fail", "stop", "pause"])
def test_transaction_failure_or_lifecycle_change_discards_mpc_state(monkeypatch, outcome):
    service, _, _, _ = mpc_service(monkeypatch)
    original = service._pid_controller
    if outcome == "cancel":
        service._update_flow_with_lifecycle_guard = lambda *_: None
    else:
        def update(*_):
            if outcome == "stop":
                service._stop_event.set()
            elif outcome == "pause":
                service._pause_event.set()
            return FlowUpdateResult(ok=outcome != "fail", q1_ok=True, q2_ok=outcome != "fail",
                                    still_running=False, safe_stop_verified=True, reason="mock transaction")
        service.pump_service.update_flow_while_running = update
    service.run_control_step()
    assert service._pid_controller is original
    assert original.offset_um == 0 and original.last_plan == ()
    assert service._feedback_hold.command_id == 0
    if outcome == "fail":
        assert service._state == SystemState.ERROR


def test_stop_request_takes_precedence_over_freeze(monkeypatch):
    service, _, writes, _ = mpc_service(monkeypatch)
    monkeypatch.setattr(DiameterMPCController, "update_input", lambda *_: PIDCommand(
        50, 20, 0, 0, True, True, "critical flow requires stop", control_mode="MPC"))
    stopped = []
    service._safety_stop_pump = lambda: stopped.append(True) or True
    service.run_control_step()
    assert not writes and stopped
    assert service._state == SystemState.ERROR


def test_parameter_page_saves_mpc_and_passes_it_to_system_config():
    from test_qt_ui_layout import _PageApp, _application
    from frontend.qt_app import ParameterPage, FrontendApp

    app = _application()
    state = _PageApp({"feedback_controller": "MPC"})
    page = ParameterPage(state)
    assert page.feedback_controller.currentData() == "MPC"
    assert not page.target_feedforward.isEnabled()
    page.submit()
    assert state.saved["feedback_controller"] == "MPC"
    state.frontend_config.update(video_source_type="file", video_source="unused.mp4", initial_q1=50, initial_q2=20)
    assert FrontendApp.build_system_config(state).feedback_controller == "MPC"
    page.feedback_controller.setCurrentIndex(0)
    assert page.target_feedforward.isEnabled()
    page.close()
