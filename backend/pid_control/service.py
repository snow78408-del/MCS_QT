from __future__ import annotations

from contextvars import ContextVar

from .base import BaseDiameterController
from .config import PIDConfig
from .diameter_pid import DiameterPIDController
from .models import PIDCommand, PIDInput, PumpState, TargetParams, VisionMetrics

_controller_var: ContextVar[BaseDiameterController | None] = ContextVar(
    "pid_controller", default=None
)


def build_controller(config: PIDConfig | None = None) -> BaseDiameterController:
    controller = DiameterPIDController(config=config)
    _controller_var.set(controller)
    return controller


def reset_controller() -> None:
    controller = _controller_var.get()
    if controller is None:
        controller = DiameterPIDController()
        _controller_var.set(controller)
    controller.reset()


def run_feedback_step(
    vision_metrics: VisionMetrics | PIDInput,
    target_params: TargetParams | None = None,
    pump_state: PumpState | None = None,
    dt: float | None = None,
) -> PIDCommand:
    controller = _controller_var.get()
    if controller is None:
        controller = DiameterPIDController()
        _controller_var.set(controller)
    if isinstance(vision_metrics, PIDInput):
        if hasattr(controller, "update_input"):
            return controller.update_input(vision_metrics)  # type: ignore[attr-defined]
        raise TypeError("controller does not support PIDInput")
    if target_params is None or pump_state is None or dt is None:
        raise TypeError("legacy PID call requires target_params, pump_state and dt")
    return controller.update(
        vision_metrics=vision_metrics,
        target_params=target_params,
        pump_state=pump_state,
        dt=float(dt),
    )
