from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.orchestrator.service import OrchestratorService


class ControlPeriodEventTests(unittest.TestCase):
    def test_periodic_deadline_skips_missed_slots_without_catch_up(self) -> None:
        deadline, skipped = OrchestratorService._advance_control_deadline(
            deadline=10.0,
            interval_s=1.5,
            completed_at=15.2,
        )

        self.assertEqual(deadline, 16.0)
        self.assertEqual(skipped, 3)

    def test_rejected_period_is_still_marked_as_seen(self) -> None:
        service = OrchestratorService.__new__(OrchestratorService)
        service._read_recognition = lambda: SimpleNamespace(control_period_id=12)
        service._last_seen_vision_period_id = None
        service._last_control_period_id = None
        service._last_control_ts = None
        service._cfg = SimpleNamespace(
            control_interval_ms=5000,
            video_source_type="video",
        )
        service.runtime = SimpleNamespace(default_control_interval_ms=5000)
        service._pump_state = SimpleNamespace(q1=50.0, q2=20.0)
        service._log = lambda _message: None
        service._update_control_snapshot = lambda _snapshot: None

        # Local-video mode rejects PID output immediately, which exercises the
        # early-return path that previously left the waiter spinning.
        service.run_control_step()

        self.assertEqual(service._last_seen_vision_period_id, 12)
        self.assertIsNone(service._last_control_period_id)


if __name__ == "__main__":
    unittest.main()
