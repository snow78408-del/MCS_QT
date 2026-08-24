from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class PumpDirectionIdentification:
    q1_control_sign: float
    q2_control_sign: float
    q1_sensitivity_um_per_flow: float
    q2_sensitivity_um_per_flow: float


def _sensitivity(
    *,
    baseline_diameter_um: float,
    perturbed_diameter_um: float,
    flow_step: float,
    minimum_diameter_response_um: float,
) -> float:
    values = (baseline_diameter_um, perturbed_diameter_um, flow_step)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("identification inputs must be finite")
    if abs(float(flow_step)) <= 1e-12:
        raise ValueError("identification flow step must be non-zero")
    diameter_change = float(perturbed_diameter_um) - float(baseline_diameter_um)
    if abs(diameter_change) < max(0.0, float(minimum_diameter_response_um)):
        raise ValueError("diameter response is too small to identify pump direction")
    return diameter_change / float(flow_step)


def identify_pump_control_directions(
    *,
    baseline_diameter_um: float,
    q1_perturbed_diameter_um: float,
    q1_flow_step: float,
    q2_perturbed_diameter_um: float,
    q2_flow_step: float,
    minimum_diameter_response_um: float = 0.5,
) -> PumpDirectionIdentification:
    """Infer safe PID-to-pump signs from two isolated step experiments.

    A positive PID output means that measured diameter must increase. Therefore
    each control sign follows the sign of the identified diameter sensitivity.
    The caller remains responsible for applying bounded perturbations and
    restoring the baseline flow between experiments.
    """
    q1_sensitivity = _sensitivity(
        baseline_diameter_um=baseline_diameter_um,
        perturbed_diameter_um=q1_perturbed_diameter_um,
        flow_step=q1_flow_step,
        minimum_diameter_response_um=minimum_diameter_response_um,
    )
    q2_sensitivity = _sensitivity(
        baseline_diameter_um=baseline_diameter_um,
        perturbed_diameter_um=q2_perturbed_diameter_um,
        flow_step=q2_flow_step,
        minimum_diameter_response_um=minimum_diameter_response_um,
    )
    return PumpDirectionIdentification(
        q1_control_sign=1.0 if q1_sensitivity > 0.0 else -1.0,
        q2_control_sign=1.0 if q2_sensitivity > 0.0 else -1.0,
        q1_sensitivity_um_per_flow=q1_sensitivity,
        q2_sensitivity_um_per_flow=q2_sensitivity,
    )
