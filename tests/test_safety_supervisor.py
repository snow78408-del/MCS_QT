from __future__ import annotations

import threading

from backend.orchestrator.safety import SafetyState, SafetySupervisor


def test_pause_invalidates_token_and_independent_thread_verifies_stop() -> None:
    stopped = threading.Event()

    def stop_pump() -> bool:
        stopped.set()
        return True

    supervisor = SafetySupervisor(stop_pump, retry_interval_s=0.01)
    try:
        token = supervisor.begin_session()
        supervisor.arm(token, heartbeat_timeout_s=10.0)
        assert supervisor.permits(token)

        supervisor.pause()

        assert not supervisor.permits(token)
        assert stopped.wait(1.0)
        snapshot = supervisor.snapshot()
        assert snapshot.state == SafetyState.PAUSED_LOCKED
        assert snapshot.stop_verified
    finally:
        supervisor.shutdown()


def test_latched_fault_requires_verified_stop_before_reset() -> None:
    supervisor = SafetySupervisor(lambda: False, retry_interval_s=1.0)
    try:
        token = supervisor.begin_session()
        supervisor.arm(token, heartbeat_timeout_s=10.0)
        supervisor.trip("test fault")
        assert supervisor.snapshot().state == SafetyState.FAULT_LATCHED
        try:
            supervisor.reset_latch()
        except RuntimeError as exc:
            assert "verified" in str(exc)
        else:  # pragma: no cover - regression assertion
            raise AssertionError("unverified fault latch was reset")
    finally:
        supervisor.shutdown()
