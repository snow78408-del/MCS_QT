from __future__ import annotations

import pytest

from backend.pump_hardware import protocol
from backend.pump_hardware.client import CommandMismatchError, PumpClient
from backend.pump_hardware.config import PumpHardwareConfig, SerialConfig


class _SerialReply:
    is_open = True

    def __init__(self, reply: bytes) -> None:
        self._reply = bytearray(reply)

    def reset_input_buffer(self) -> None:
        pass

    def write(self, _frame: bytes) -> None:
        pass

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        if not self._reply:
            return b""
        value = bytes(self._reply[:size])
        del self._reply[:size]
        return value


def test_reply_from_another_pump_address_is_rejected() -> None:
    config = PumpHardwareConfig(retry_count=1, post_write_delay=0.0)
    client = PumpClient(SerialConfig(address=1), config)
    client._ser = _SerialReply(protocol.build_frame(2, protocol.pdu_rss()))

    with pytest.raises(CommandMismatchError, match="地址不匹配"):
        client.send_pdu(protocol.pdu_rss(), expect_cmd="RSS")
