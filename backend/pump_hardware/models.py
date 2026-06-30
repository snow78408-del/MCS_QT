from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PumpConnectionState:
    serial_connected: bool = False
    comm_established: bool = False
    fully_ready: bool = False
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PumpChannelState:
    logical_name: str
    physical_channel: str
    enabled: bool
    running: bool
    communication_ok: bool
    target_flow_rate: float | None
    actual_flow_rate: float | None
    flow_rate_unit: str = "uL/min"
    last_readback_time: float | None = None
    error: str = ""


@dataclass(slots=True)
class SystemSetup:
    enable_mask: int
    copy_mask: int
    delay_values: list[int]
    delay_units: list[int]


@dataclass(slots=True)
class RunState:
    sys_runstate: int
    q_runstate: int
    system_running: bool
    channel_running: list[bool]


@dataclass(slots=True)
class ChannelParams:
    channel: int
    mode: int
    syringe_code: int
    dispense_value: int
    dispense_unit: int
    infuse_time_value: int
    infuse_time_unit: int
    withdraw_time_value: int
    withdraw_time_unit: int
    repeat_count: int
    interval_value: int


@dataclass(slots=True)
class PumpOperationResult:
    ok: bool
    error: str | None = None
    raw_reply: bytes | None = None
    parsed_reply: Any = None
    verified: bool = False
    reason: str | None = None


@dataclass(slots=True)
class FlowUpdateResult:
    ok: bool
    q1_ok: bool
    q2_ok: bool
    still_running: bool
    q1_error: str | None = None
    q2_error: str | None = None
    run_state_error: str | None = None
    verified_q1: ChannelParams | None = None
    verified_q2: ChannelParams | None = None
    reason: str | None = None
