from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from backend.orchestrator.models import SystemConfig
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.pid_control.models import PIDCommand
from backend.pump_hardware.models import FlowUpdateResult, PumpConnectionState, PumpOperationResult, RunState


class _PumpStub:
    def __init__(self, *, stop_ok: bool = True, start_ok: bool = True) -> None:
        self.serial_config = SimpleNamespace(port="", address=1, baudrate=1200, parity="N")
        self.calls: list[str] = []
        self.stop_ok = stop_ok
        self.start_ok = start_ok

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def connect_and_probe(self) -> PumpConnectionState:
        self.calls.append("connect")
        return PumpConnectionState(serial_connected=True, comm_established=True, fully_ready=True)

    def start_infusion_and_verify(self, channels) -> PumpOperationResult:
        self.calls.append(f"start:{channels}")
        return PumpOperationResult(ok=self.start_ok, verified=self.start_ok, reason="start failed" if not self.start_ok else "")

    def stop_system_and_verify(self) -> PumpOperationResult:
        self.calls.append("stop")
        return PumpOperationResult(ok=self.stop_ok, verified=self.stop_ok, reason="stop failed" if not self.stop_ok else "")


class PumpInteractionFlowTests(unittest.TestCase):
    def test_interaction_test_rejects_q1_not_above_q2_before_connect(self) -> None:
        service = OrchestratorService.__new__(OrchestratorService)
        service.pump_service = _PumpStub()
        service._state = SystemState.IDLE
        service._log = lambda _message: None

        with self.assertRaises(ValueError):
            service.run_pump_interaction_test(
                port="COM3", address=1, baudrate=1200, parity="N", q1=20.0, q2=20.0
            )

        self.assertEqual(service.pump_service.calls, [])

    def test_interaction_start_verification_failure_still_protectively_stops(self) -> None:
        pump = _PumpStub(start_ok=False)
        service = OrchestratorService.__new__(OrchestratorService)
        service.pump_service = pump
        service._state = SystemState.IDLE
        service._log = lambda _message: None
        service._apply_init_flow_rates = lambda _q1, _q2: PumpOperationResult(ok=True, verified=True)

        result = service.run_pump_interaction_test(
            port="COM3", address=1, baudrate=1200, parity="N", q1=50.0, q2=20.0
        )

        self.assertFalse(result["ok"])
        self.assertEqual(pump.calls, ["disconnect", "connect", "start:[1, 2]", "stop"])
        self.assertEqual(result["steps"][-1]["name"], "异常保护停止")

    def test_flow_readback_failure_invalidates_communication_state(self) -> None:
        class ReadbackFailurePump(_PumpStub):
            def get_current_q_state(self):
                raise RuntimeError("readback timeout")

        pump = ReadbackFailurePump()
        service = OrchestratorService(vision_service=SimpleNamespace(), pump_service=pump)
        service._pump_control_enabled = True
        service._pump_state.comm_established = True
        service._pump_state.q1 = 50.0
        service._pump_state.q2 = 20.0

        self.assertFalse(service._sync_pump_flow_readback("test"))
        self.assertFalse(service._pump_state.comm_established)
        self.assertFalse(service._pump_control_enabled)
        self.assertIsNone(service._pump_state.q1_actual)
        self.assertIsNone(service._pump_state.q2_actual)

    def test_stop_failure_keeps_error_state_and_does_not_claim_safe(self) -> None:
        pump = _PumpStub(stop_ok=False)
        adapter = SimpleNamespace(stop=lambda: None)
        service = OrchestratorService(vision_service=SimpleNamespace(), vision_adapter=adapter, pump_service=pump)
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.RUNNING
        service._pump_state.running = True
        service._pump_control_enabled = True

        service.stop()

        self.assertEqual(service._state, SystemState.ERROR)
        self.assertTrue(service._pump_state.running)
        self.assertTrue(service._pump_state.last_error)
        self.assertIn("stop", pump.calls)

    def test_orchestrators_have_isolated_pid_controllers(self) -> None:
        first = OrchestratorService(vision_service=SimpleNamespace())
        second = OrchestratorService(vision_service=SimpleNamespace())
        self.assertIsNot(first._pid_controller, second._pid_controller)
        first._pid_controller.integral = 123.0
        self.assertNotEqual(second._pid_controller.integral, 123.0)

    def test_error_and_pause_recovery_never_restart_infusion(self) -> None:
        pump = _PumpStub()
        service = OrchestratorService(vision_service=SimpleNamespace(), vision_adapter=SimpleNamespace(), pump_service=pump)
        service._pump_control_enabled = True
        service._state = SystemState.ERROR
        ok, _ = service._try_resume_infusion("error")
        self.assertFalse(ok)
        service._state = SystemState.PAUSED
        service._pause_event.set()
        ok, _ = service._try_resume_infusion("pause")
        self.assertFalse(ok)
        self.assertEqual(pump.calls, [])

    def test_pause_publishes_paused_before_stopping_pump(self) -> None:
        pump = _PumpStub()
        service = OrchestratorService(vision_service=SimpleNamespace(), vision_adapter=SimpleNamespace(), pump_service=pump)
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.RUNNING
        service._pump_control_enabled = True
        original = pump.stop_system_and_verify
        def stop_check():
            self.assertEqual(service._state, SystemState.PAUSED)
            return original()
        pump.stop_system_and_verify = stop_check
        service.pause()
        self.assertEqual(service._state, SystemState.PAUSED)
        self.assertEqual(pump.calls, ["stop"])

    def test_adapter_start_failure_rolls_back_adapter_and_recorder(self) -> None:
        pump = _PumpStub()
        adapter = SimpleNamespace(
            start=lambda: (_ for _ in ()).throw(RuntimeError("adapter failed")),
            stop=lambda: setattr(adapter, "stopped", True),
            stopped=False,
        )
        service = OrchestratorService(vision_service=SimpleNamespace(), vision_adapter=adapter, pump_service=pump)
        service._cfg = SystemConfig(
            target_diameter=50.0,
            pixel_to_micron=1.0,
            video_source_type="camera",
            video_source="",
            initial_q1=50.0,
            initial_q2=20.0,
            control_interval_ms=500,
        )
        service._state = SystemState.INITIALIZED
        service._pump_control_enabled = True

        with self.assertRaises(RuntimeError):
            service.start()

        self.assertTrue(adapter.stopped)
        self.assertEqual(pump.calls, [])
        self.assertFalse(service._pump_control_enabled)
        self.assertFalse(service._pid_data_recorder.status()["active"])
        self.assertFalse(service._pid_data_recorder.has_unsaved_data())

    def test_resume_failure_protectively_stops_and_enters_error(self) -> None:
        pump = _PumpStub(start_ok=False)
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=SimpleNamespace(stop=lambda: None),
            pump_service=pump,
        )
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.PAUSED
        service._pause_event.set()
        service._pump_control_enabled = True
        service._pump_state.connected = True
        service._pump_state.comm_established = True

        with self.assertRaises(RuntimeError):
            service.resume()

        self.assertEqual(service._state, SystemState.ERROR)
        self.assertFalse(service._pump_control_enabled)
        self.assertIn("start:[1, 2]", pump.calls)
        self.assertIn("stop", pump.calls)

    def test_resume_readback_failure_rolls_back_before_publishing_running(self) -> None:
        pump = _PumpStub()
        pump.get_current_q_state = lambda: (_ for _ in ()).throw(RuntimeError("readback timeout"))
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=SimpleNamespace(stop=lambda: None),
            pump_service=pump,
        )
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.PAUSED
        service._pause_event.set()
        service._pump_control_enabled = True
        service._pump_state.comm_established = True

        with self.assertRaises(RuntimeError):
            service.resume()

        self.assertEqual(service._state, SystemState.ERROR)
        self.assertFalse(service._pump_control_enabled)
        self.assertFalse(service._pump_state.running)
        self.assertGreaterEqual(pump.calls.count("stop"), 1)

    def test_start_invalidated_by_stop_rolls_back_without_resurrecting_running(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingAdapter:
            def start(self):
                entered.set()
                release.wait(timeout=2.0)

            def stop(self):
                pass

        pump = _PumpStub()
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=BlockingAdapter(),
            pump_service=pump,
        )
        service._cfg = SystemConfig(50.0, 1.0, "camera", "", 50.0, 20.0, 500)
        service._state = SystemState.INITIALIZED
        service._pump_control_enabled = True
        service._pump_state.connected = True
        service._pump_state.comm_established = True

        start_errors: list[BaseException] = []

        def start_task():
            try:
                service.start()
            except BaseException as exc:
                start_errors.append(exc)
        start_thread = threading.Thread(target=start_task)
        start_thread.start()
        self.assertTrue(entered.wait(timeout=1.0))

        service.stop()
        release.set()
        start_thread.join(timeout=2.0)

        self.assertFalse(start_thread.is_alive())
        self.assertTrue(start_errors)
        self.assertEqual(service._state, SystemState.STOPPED)
        self.assertNotEqual(service._state, SystemState.RUNNING)
        self.assertFalse(service._pump_state.running)
        self.assertIn("stop", pump.calls)

    def test_flow_retry_is_invalidated_by_pause_without_holding_orchestration_lock(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        updates: list[int] = []

        class BlockingRetryPump(_PumpStub):
            def update_flow_while_running(self, _q1, _q2):
                updates.append(1)
                entered.set()
                release.wait(timeout=2.0)
                return FlowUpdateResult(
                    ok=False,
                    q1_ok=False,
                    q2_ok=False,
                    still_running=True,
                    reason="transient update failure",
                )

        pump = BlockingRetryPump()
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=SimpleNamespace(stop=lambda: None),
            pump_service=pump,
        )
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.RUNNING
        service._pump_control_enabled = True
        generation = service._lifecycle_generation

        result: list[object] = []
        update_thread = threading.Thread(
            target=lambda: result.append(service._update_flow_with_lifecycle_guard(50.0, 20.0, generation))
        )
        update_thread.start()
        self.assertTrue(entered.wait(timeout=1.0))
        service.pause()
        release.set()
        update_thread.join(timeout=2.0)

        self.assertFalse(update_thread.is_alive())
        self.assertIsNone(result[0])
        self.assertEqual(len(updates), 1)
        self.assertEqual(service._state, SystemState.PAUSED)

    def test_pause_stop_exception_disables_control_and_enters_error(self) -> None:
        class RaisingStopPump(_PumpStub):
            def stop_system_and_verify(self):
                self.calls.append("stop")
                raise RuntimeError("serial stop timeout")

        pump = RaisingStopPump()
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=SimpleNamespace(stop=lambda: None),
            pump_service=pump,
        )
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.RUNNING
        service._pump_control_enabled = True
        service._pump_state.running = True

        with self.assertRaises(RuntimeError):
            service.pause()

        self.assertEqual(service._state, SystemState.ERROR)
        self.assertFalse(service._pump_control_enabled)
        self.assertTrue(service._stop_event.is_set())
        self.assertTrue(service._pump_state.running)
        self.assertIn("serial stop timeout", service._pump_state.last_error)

    def test_stop_invalidates_inflight_resume_before_it_can_publish_running(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingPump(_PumpStub):
            def start_infusion_and_verify(self, channels):
                self.calls.append(f"start:{channels}")
                entered.set()
                release.wait(timeout=2.0)
                return PumpOperationResult(ok=True, verified=True)

        pump = BlockingPump()
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=SimpleNamespace(stop=lambda: None),
            pump_service=pump,
        )
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.PAUSED
        service._pause_event.set()
        service._pump_control_enabled = True
        service._pump_state.connected = True
        service._pump_state.comm_established = True

        resume_thread = threading.Thread(target=service.resume)
        resume_thread.start()
        self.assertTrue(entered.wait(timeout=1.0))
        service.stop()
        release.set()
        resume_thread.join(timeout=2.0)

        self.assertFalse(resume_thread.is_alive())
        self.assertEqual(service._state, SystemState.STOPPED)
        self.assertFalse(service._pump_state.running)
        self.assertNotEqual(service._state, SystemState.RUNNING)
        self.assertIn("stop", pump.calls)

    def test_partial_flow_update_stops_and_enters_error(self) -> None:
        pump = _PumpStub()
        pump.read_run_state = lambda: PumpOperationResult(
            ok=True,
            parsed_reply=RunState(0x03, 0x03, True, [True, True]),
        )
        pump.are_required_channels_running = lambda _channels, _state: (True, "ok")
        pump.get_current_q_state = lambda: (50.0, 20.0)
        pump.update_flow_while_running = lambda _q1, _q2: FlowUpdateResult(
            ok=False,
            q1_ok=True,
            q2_ok=False,
            still_running=True,
            reason="CH2 readback failed",
        )
        service = OrchestratorService(vision_service=SimpleNamespace(), pump_service=pump)
        service._cfg = SystemConfig(50.0, 1.0, "camera", "", 50.0, 20.0, 500)
        service._state = SystemState.RUNNING
        service._pump_control_enabled = True
        service._pump_state.comm_established = True
        service._pump_state.q1 = 50.0
        service._pump_state.q2 = 20.0
        service._read_recognition = lambda: SimpleNamespace(
            control_period_id=1,
            frame_id=1,
            timestamp=__import__("time").time(),
            valid_for_control=True,
            reason="",
            control_reason="",
            frame_avg_diameter=50.0,
            avg_diameter=50.0,
            frame_droplet_count=1,
            frame_diameter_cv=0.0,
        )
        service.disturbance_service = SimpleNamespace(
            build_and_submit_sample=lambda **_kwargs: SimpleNamespace(pump_response_delay_ms=0.0),
            predict=lambda _sample: None,
        )
        service._pid_controller.update_input = lambda _input: PIDCommand(
            q1=50.0,
            q2=20.0,
            diameter_error=1.0,
            adjustment=0.1,
            freeze_feedback=False,
            suggested_stop=False,
            reason="",
        )

        service.run_control_step()

        self.assertEqual(service._state, SystemState.ERROR)
        self.assertFalse(service._pump_control_enabled)
        self.assertFalse(service._pump_state.running)
        self.assertIn("stop", pump.calls)

    def test_pause_invalidates_inflight_automatic_recovery(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingPump(_PumpStub):
            def start_infusion_and_verify(self, channels):
                self.calls.append(f"start:{channels}")
                entered.set()
                release.wait(timeout=2.0)
                return PumpOperationResult(ok=True, verified=True)

        pump = BlockingPump()
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=SimpleNamespace(stop=lambda: None),
            pump_service=pump,
        )
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.RUNNING
        service._pump_control_enabled = True
        service._pump_state.comm_established = True

        recovery_thread = threading.Thread(target=lambda: service._try_resume_infusion("test"))
        recovery_thread.start()
        self.assertTrue(entered.wait(timeout=1.0))
        service.pause()
        release.set()
        recovery_thread.join(timeout=2.0)

        self.assertFalse(recovery_thread.is_alive())
        self.assertEqual(service._state, SystemState.PAUSED)
        self.assertFalse(service._pump_state.running)

    def test_stop_invalidates_inflight_automatic_recovery(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingPump(_PumpStub):
            def start_infusion_and_verify(self, channels):
                self.calls.append(f"start:{channels}")
                entered.set()
                release.wait(timeout=2.0)
                return PumpOperationResult(ok=True, verified=True)

        pump = BlockingPump()
        service = OrchestratorService(
            vision_service=SimpleNamespace(),
            vision_adapter=SimpleNamespace(stop=lambda: None),
            pump_service=pump,
        )
        service._cfg = SimpleNamespace(video_source_type="camera")
        service._state = SystemState.RUNNING
        service._pump_control_enabled = True
        service._pump_state.comm_established = True

        recovery_result: list[tuple[bool, str]] = []
        recovery_thread = threading.Thread(
            target=lambda: recovery_result.append(service._try_resume_infusion("test"))
        )
        recovery_thread.start()
        self.assertTrue(entered.wait(timeout=1.0))
        service.stop()
        release.set()
        recovery_thread.join(timeout=2.0)

        self.assertFalse(recovery_thread.is_alive())
        self.assertEqual(service._state, SystemState.STOPPED)
        self.assertEqual(recovery_result[0][0], False)
        self.assertFalse(service._pump_state.running)

    def test_start_failure_rolls_back_adapter_recorder_and_pump(self) -> None:
        pump = _PumpStub(start_ok=False)
        adapter = SimpleNamespace(start=lambda: None, stop=lambda: setattr(adapter, "stopped", True), stopped=False)
        service = OrchestratorService(vision_service=SimpleNamespace(), vision_adapter=adapter, pump_service=pump)
        service._cfg = SystemConfig(
            target_diameter=50.0,
            pixel_to_micron=1.0,
            video_source_type="camera",
            video_source="",
            initial_q1=50.0,
            initial_q2=20.0,
            control_interval_ms=500,
        )
        service._state = SystemState.INITIALIZED
        service._pump_control_enabled = True
        service._pump_state.connected = True
        service._pump_state.comm_established = True

        with self.assertRaises(RuntimeError):
            service.start()

        self.assertTrue(adapter.stopped)
        self.assertIn("stop", pump.calls)
        self.assertFalse(service._pump_control_enabled)
        self.assertFalse(service._pid_data_recorder.status()["active"])
        self.assertFalse(service._pid_data_recorder.has_unsaved_data())

    def test_complete_test_writes_starts_and_stops_in_order(self) -> None:
        service = OrchestratorService.__new__(OrchestratorService)
        service.pump_service = _PumpStub()
        service._state = SystemState.IDLE
        service._log = lambda _message: None

        def apply_params(q1: float, q2: float) -> PumpOperationResult:
            service.pump_service.calls.append(f"write:{q1}:{q2}")
            return PumpOperationResult(ok=True, verified=True)

        service._apply_init_flow_rates = apply_params
        result = service.run_pump_interaction_test(
            port="COM3", address=1, baudrate=1200, parity="N", q1=50.0, q2=20.0
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            service.pump_service.calls,
            ["disconnect", "connect", "write:50.0:20.0", "start:[1, 2]", "stop"],
        )
        self.assertTrue(all(step["ok"] for step in result["steps"]))


if __name__ == "__main__":
    unittest.main()
