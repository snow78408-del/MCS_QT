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
    def test_default_flow_limits_match_commissioning_envelope(self) -> None:
        config = PIDConfig()

        self.assertEqual((config.q1_min, config.q1_max), (20.0, 200.0))
        self.assertEqual((config.q2_min, config.q2_max), (5.0, 25.0))
        self.assertEqual(config.total_flow_max, 225.0)

    def test_invalid_current_flow_requests_stop_without_control_output(self) -> None:
        controller = DiameterPIDController(PIDConfig())
        for q1, q2 in ((0.0, 20.0), (-1.0, 20.0), (float("nan"), 20.0), (50.0, float("inf"))):
            command = controller.update_input(_input(frame_id=1, q1=q1, q2=q2))
            self.assertTrue(command.freeze_feedback)
            self.assertTrue(command.suggested_stop)
            self.assertIn("flow invalid", command.reason)

    def test_regressed_frame_id_is_rejected(self) -> None:
        controller = DiameterPIDController(PIDConfig(control_mode=PIDControlMode.CLASSIC_PID.value))
        controller.update_input(_input(frame_id=2, q1=50.0, q2=20.0))
        command = controller.update_input(_input(frame_id=1, q1=50.0, q2=20.0))
        self.assertTrue(command.freeze_feedback)
        self.assertFalse(command.suggested_stop)
        self.assertIn("stale or regressed", command.reason)

    def test_default_adaptive_mode_is_wired_and_tunes_after_three_valid_periods(self) -> None:
        config = PIDConfig(feedforward_enabled=False)
        controller = DiameterPIDController(config)

        commands = []
        q1, q2 = 50.0, 20.0
        for frame_id in range(1, 4):
            command = controller.update_input(
                _input(frame_id=frame_id, q1=q1, q2=q2, current=40.0)
            )
            commands.append(command)
            q1, q2 = command.q1, command.q2

        self.assertTrue(all(command.adaptive_enabled for command in commands))
        self.assertFalse(commands[0].adaptive_active)
        self.assertTrue(commands[-1].adaptive_active)
        self.assertGreater(commands[-1].kp, config.base_kp)
        self.assertIn("feedback history", commands[-1].adaptive_reason)

    def test_adaptive_pid_updates_without_disturbance_model(self) -> None:
        config = PIDConfig(
            control_mode=PIDControlMode.ADAPTIVE_PID.value,
            adaptive_min_sample_count=4,
            adaptive_update_interval=1,
            base_kp=0.08,
            base_ki=0.01,
            base_kd=0.0,
        )
        controller = DiameterPIDController(config)

        command = None
        q1, q2 = 50.0, 20.0
        for frame_id in range(1, 5):
            command = controller.update_input(
                _input(frame_id=frame_id, q1=q1, q2=q2, current=40.0)
            )
            q1, q2 = command.q1, command.q2

        assert command is not None
        self.assertTrue(command.adaptive_active)
        self.assertIn("feedback history", command.adaptive_reason)
        self.assertGreater(command.kp, config.base_kp)
        self.assertGreater(command.ki, config.base_ki)
    def test_positional_output_does_not_accumulate_on_every_frame(self) -> None:
        config = PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
        )
        controller = DiameterPIDController(config)

        first = controller.update_input(_input(frame_id=1, q1=100.0, q2=10.0))
        second = controller.update_input(_input(frame_id=2, q1=first.q1, q2=first.q2))

        self.assertAlmostEqual(first.q1, 98.0)
        self.assertAlmostEqual(first.q2, 11.0)
        self.assertAlmostEqual(second.q1, first.q1)
        self.assertAlmostEqual(second.q2, first.q2)

    def test_large_droplet_increases_q1_and_decreases_q2(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=100.0, q2=10.0, current=60.0))
        self.assertGreater(command.q1, 100.0)
        self.assertLess(command.q2, 10.0)

    def test_small_droplet_decreases_q1_and_increases_q2(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=100.0, q2=10.0, current=40.0))
        self.assertLess(command.q1, 100.0)
        self.assertGreater(command.q2, 10.0)

    def test_q1_uses_larger_adjustment_gain_than_q2(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
            q1_output_gain=2.0,
            q2_output_gain=1.0,
        ))
        command = controller.update_input(
            _input(frame_id=1, q1=100.0, q2=10.0, current=40.0)
        )

        self.assertAlmostEqual(100.0 - command.q1, 2.0)
        self.assertAlmostEqual(command.q2 - 10.0, 1.0)
        self.assertAlmostEqual(command.q1_output_gain, 2.0)
        self.assertAlmostEqual(command.q2_output_gain, 1.0)

    def test_identified_control_signs_can_reverse_pump_allocation(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
            q1_control_sign=1.0,
            q2_control_sign=-1.0,
        ))
        command = controller.update_input(
            _input(frame_id=1, q1=100.0, q2=10.0, current=40.0)
        )
        self.assertGreater(command.q1, 100.0)
        self.assertLess(command.q2, 10.0)

    def test_zero_control_sign_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PIDConfig(q1_control_sign=0.0)

    def test_sub_resolution_channel_can_be_inactive_without_division_by_zero(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            base_kd=0.0,
            q2_control_sign=0.0,
            q2_output_gain=0.0,
        ))

        command = controller.update_input(
            _input(frame_id=1, q1=100.0, q2=10.0, current=40.0)
        )

        self.assertLess(command.q1,100.0)
        self.assertEqual(command.q2,10.0)
        self.assertEqual(command.q2_output_gain,0.0)

    def test_nonpositive_phase_gap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PIDConfig(min_q1_q2_gap=0.0)
        with self.assertRaises(ValueError):
            PIDConfig(min_q1_q2_gap=0.1)

    def test_nonpositive_output_gain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PIDConfig(q1_output_gain=0.0)

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
        controller.update_input(_input(frame_id=1, q1=100.0, q2=10.0, current=1.0))
        self.assertEqual(controller.integral, 0.0)

    def test_per_cycle_flow_change_is_limited(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=100.0,
            base_ki=0.0,
            output_rate_limit=10000.0,
            max_flow_change_per_cycle=5.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=100.0, q2=10.0, current=40.0))
        self.assertGreaterEqual(command.q1, 95.0)
        self.assertLessEqual(command.q2, 15.0)

    def test_total_flow_is_limited(self) -> None:
        controller = DiameterPIDController(PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=0.1,
            base_ki=0.0,
            total_flow_max=60.0,
        ))
        command = controller.update_input(_input(frame_id=1, q1=70.0, q2=10.0, current=40.0))
        self.assertLessEqual(command.q1 + command.q2, 60.0 + 1e-9)

    def test_small_droplet_saturates_before_q1_reaches_q2(self) -> None:
        config = PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=10.0,
            base_ki=0.1,
            base_kd=0.0,
            output_rate_limit=1000.0,
            q1_min=0.2,
            q1_max=5000.0,
            q2_min=0.2,
            q2_max=5000.0,
            total_flow_max=8000.0,
            min_q1_q2_gap=0.2,
        )
        controller = DiameterPIDController(config)
        command = controller.update_input(
            _input(frame_id=1, q1=50.0, q2=20.0, current=1.0)
        )

        self.assertFalse(command.freeze_feedback)
        self.assertFalse(command.suggested_stop)
        self.assertGreater(command.q1, command.q2)
        self.assertAlmostEqual(command.q1 - command.q2, 0.2)
        self.assertEqual(controller.integral, 0.0)
        self.assertIn("saturated", command.reason)

    def test_pump_limit_saturates_without_requesting_system_stop(self) -> None:
        config = PIDConfig(
            control_mode=PIDControlMode.CLASSIC_PID.value,
            base_kp=1.0,
            base_ki=0.1,
            base_kd=0.0,
            q1_min=0.2,
            q2_min=0.2,
            output_rate_limit=1000.0,
        )
        controller = DiameterPIDController(config)
        command = controller.update_input(
            _input(frame_id=1, q1=50.0, q2=20.0, current=100.0)
        )

        self.assertFalse(command.freeze_feedback)
        self.assertFalse(command.suggested_stop)
        self.assertAlmostEqual(command.q1, 89.6)
        self.assertAlmostEqual(command.q2, 0.2)
        self.assertAlmostEqual(command.q1 + command.q2, 89.8)
        self.assertEqual(controller.integral, 0.0)
        self.assertIn("saturated", command.reason)


if __name__ == "__main__":
    unittest.main()
