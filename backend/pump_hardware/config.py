from __future__ import annotations

from dataclasses import dataclass
import math


DEFAULT_BAUDRATE = 1200
DEFAULT_PARITY = "N"
DEFAULT_BYTESIZE = 8
DEFAULT_STOPBITS = 1


@dataclass(slots=True)
class SerialConfig:
    port: str = ""
    baudrate: int = DEFAULT_BAUDRATE
    parity: str = DEFAULT_PARITY
    timeout: float = 0.25
    write_timeout: float = 0.8
    address: int = 1
    allow_parity_fallback_n: bool = True

    def __post_init__(self) -> None:
        if not 1 <= int(self.address) <= 0x1F:
            raise ValueError("pump address must be in [1, 31]")
        if int(self.baudrate) <= 0:
            raise ValueError("baudrate must be positive")
        self.parity = str(self.parity or "").upper()
        if self.parity not in {"N", "E"}:
            raise ValueError("parity must be 'N' or 'E'")
        for name in ("timeout", "write_timeout"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(slots=True)
class PumpHardwareConfig:
    reply_timeout: float = 0.8
    idle_timeout: float = 0.22
    retry_count: int = 2
    retry_interval: float = 0.08
    post_write_delay: float = 0.08
    probe_step_delay: float = 0.06
    wsp_verify_read_retry: int = 3
    wsp_verify_retry_interval: float = 0.12
    # The controller communicates at only 1200 baud. Give CH1 enough time to
    # finish its commit before issuing CH2, then recover CH2 independently if
    # its write/readback is lost. A successful CH1 is never written again.
    inter_channel_update_delay: float = 0.18
    q2_update_max_attempts: int = 2
    q2_update_retry_interval: float = 0.25
    wss_swap_fallback: bool = True
    # CH1 is oil and CH2 is water in this system.
    min_q1_q2_gap: float = 0.2

    def __post_init__(self) -> None:
        positive = (
            "reply_timeout",
            "idle_timeout",
            "retry_interval",
            "wsp_verify_retry_interval",
            "min_q1_q2_gap",
        )
        non_negative = (
            "post_write_delay",
            "probe_step_delay",
            "inter_channel_update_delay",
            "q2_update_retry_interval",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in non_negative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("retry_count", "wsp_verify_read_retry", "q2_update_max_attempts"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
