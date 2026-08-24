from __future__ import annotations

import pytest

from backend.pid_control.identification import identify_pump_control_directions


def test_identifies_default_differential_pump_directions() -> None:
    result = identify_pump_control_directions(
        baseline_diameter_um=50.0,
        q1_perturbed_diameter_um=48.0,
        q1_flow_step=5.0,
        q2_perturbed_diameter_um=53.0,
        q2_flow_step=5.0,
    )
    assert result.q1_control_sign == -1.0
    assert result.q2_control_sign == 1.0
    assert result.q1_sensitivity_um_per_flow == pytest.approx(-0.4)
    assert result.q2_sensitivity_um_per_flow == pytest.approx(0.6)


def test_rejects_response_below_measurement_resolution() -> None:
    with pytest.raises(ValueError, match="too small"):
        identify_pump_control_directions(
            baseline_diameter_um=50.0,
            q1_perturbed_diameter_um=50.1,
            q1_flow_step=5.0,
            q2_perturbed_diameter_um=53.0,
            q2_flow_step=5.0,
        )
