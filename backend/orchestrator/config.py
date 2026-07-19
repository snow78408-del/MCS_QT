from __future__ import annotations

from dataclasses import dataclass

default_control_interval_ms = 500
min_control_interval_ms = 500
max_control_interval_ms = 5000
pump_command_retry = 2
init_timeout_s = 8.0
stop_timeout_s = 5.0


@dataclass(slots=True)
class OrchestratorConfig:
    default_control_interval_ms: int = default_control_interval_ms
    min_control_interval_ms: int = min_control_interval_ms
    max_control_interval_ms: int = max_control_interval_ms
    pump_command_retry: int = pump_command_retry
    init_timeout_s: float = init_timeout_s
    stop_timeout_s: float = stop_timeout_s
