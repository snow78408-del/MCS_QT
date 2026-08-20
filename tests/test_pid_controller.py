from __future__ import annotations

import unittest

from backend.pid_control.config import PIDConfig, PIDControlMode
from backend.pid_control.diameter_pid import DiameterPIDController
from backend.pid_control.models import PIDInput


def _input(*, frame_id: int, q1: float, q2: float, current: float = 40.0) -> PIDInput:
    return PIDInput(
        target_diameter_um=50.0,
        current_diameter_um=current,
        current_q1=q1,
        current_q2=q2,
        dt=0.5,
        frame_id=frame_id,
        vision_valid=True,
        pump_communication_ok=True,
        droplet_count=2,
    )


class DiameterPidTests(unittest.TestCase):
    def test_positional_output_does_not_accumulate_on_every_frame(self) -> None:
        config = PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
        )
        controller = DiameterPIDController(config)

        first = controller.update_input(_input(frame_id=1, q1=50.0, q2=20.0))
        second = controller.update_input(_input(frame_id=2, q1=first.q1, q2=first.q2))

        self.assertAlmostEqual(first.q1, 49.0)
        self.assertAlmostEqual(first.q2, 21.0)
        self.assertAlmostEqual(second.q1, first.q1)
        self.assertAlmostEqual(second.q2, first.q2)

    def test_large_droplet_increases_q1_and_decreases_q2(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=50.0, q2=20.0, current=60.0))
        self.assertGreater(command.q1, 50.0)
        self.assertLess(command.q2, 20.0)

    def test_small_droplet_decreases_q1_and_increases_q2(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=50.0, q2=20.0, current=40.0))
        self.assertLess(command.q1, 50.0)
        self.assertGreater(command.q2, 20.0)

    def test_integral_does_not_wind_up_while_output_is_saturated(self) -> None:
        config = PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=10.0,
            base_ki=1.0,
            output_min=-1.0,
            output_max=1.0,
            output_rate_limit=10.0,
        )
        controller = DiameterPIDController(config)
        controller.update_input(_input(frame_id=1, q1=50.0, q2=20.0, current=1.0))
        self.assertEqual(controller.integral, 0.0)

    def test_per_cycle_flow_change_is_limited(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=100.0,
            base_ki=0.0,
            output_rate_limit=10000.0,
            max_flow_change_per_cycle=5.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=50.0, q2=50.0, current=40.0))
        self.assertGreaterEqual(command.q1, 45.0)
        self.assertLessEqual(command.q2, 55.0)

    def test_total_flow_is_limited(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            total_flow_max=60.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=50.0, q2=50.0, current=40.0))
        self.assertLessEqual(command.q1 + command.q2, 60.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
