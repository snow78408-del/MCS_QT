from __future__ import annotations

from .base import BaseDiameterController
from .config import PIDConfig
from .diameter_pid import DiameterPIDController
from .models import PIDCommand, PIDInput, PumpState, TargetParams, VisionMetrics

_controller: BaseDiameterController | None = None


def build_controller(config: PIDConfig | None = None) -> BaseDiameterController:
    global _controller
    _controller = DiameterPIDController(config=config)
    return _controller


def reset_controller() -> None:
    global _controller
    if _controller is None:
        _controller = DiameterPIDController()
    _controller.reset()


def run_feedback_step(
    vision_metrics: VisionMetrics | PIDInput,
    target_params: TargetParams | None = None,
    pump_state: PumpState | None = None,
    dt: float | None = None,
) -> PIDCommand:
    global _controller
    if _controller is None:
        _controller = DiameterPIDController()
    if isinstance(vision_metrics, PIDInput):
        if hasattr(_controller, "update_input"):
            return _controller.update_input(vision_metrics)  # type: ignore[attr-defined]
        raise TypeError("controller does not support PIDInput")
    if target_params is None or pump_state is None or dt is None:
        raise TypeError("legacy PID call requires target_params, pump_state and dt")
    return _controller.update(
        vision_metrics=vision_metrics,
        target_params=target_params,
        pump_state=pump_state,
        dt=float(dt),
    )
