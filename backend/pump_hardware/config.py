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
    # The controller communicates at only 1200 baud. Give CH1 enough time to
    # finish its commit before issuing CH2, then recover CH2 independently if
    # its write/readback is lost. A successful CH1 is never written again.
    inter_channel_update_delay: float = 0.18
    q2_update_max_attempts: int = 2
    q2_update_retry_interval: float = 0.25
    wss_swap_fallback: bool = True
    # CH1 is oil and CH2 is water in this system.
    min_q1_q2_gap: float = 0.2
