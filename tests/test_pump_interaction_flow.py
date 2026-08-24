from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.pump_hardware.models import PumpConnectionState, PumpOperationResult


class _PumpStub:
    def __init__(self) -> None:
        self.serial_config = SimpleNamespace(port="", address=1, baudrate=1200, parity="N")
        self.calls: list[str] = []

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def connect_and_probe(self) -> PumpConnectionState:
        self.calls.append("connect")
        return PumpConnectionState(serial_connected=True, comm_established=True, fully_ready=True)

    def start_infusion_and_verify(self, channels) -> PumpOperationResult:
        self.calls.append(f"start:{channels}")
        return PumpOperationResult(ok=True, verified=True)

    def stop_system_and_verify(self) -> PumpOperationResult:
        self.calls.append("stop")
        return PumpOperationResult(ok=True, verified=True)


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
