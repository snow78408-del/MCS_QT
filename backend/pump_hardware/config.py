from __future__ import annotations

from dataclasses import dataclass


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
    wss_swap_fallback: bool = True
