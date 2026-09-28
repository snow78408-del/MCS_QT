from __future__ import annotations

import pytest

from backend.pump_hardware import protocol
from backend.pump_hardware.config import SerialConfig


def test_ts_serial_defaults_are_1200_8e1() -> None:
    config = SerialConfig()

    assert config.baudrate == 1200
    assert config.parity == "E"
    assert not config.allow_parity_fallback_n


def test_wss_encodes_each_channel_delay_value_next_to_its_unit() -> None:
    pdu = protocol.pdu_wss(
        enable_mask=0x05,
        copy_mask=0x02,
        delay_values=[0x0123, 0x0456, 0x0789, 10],
        delay_units=[0, 1, 2, 0],
    )

    assert pdu == bytes.fromhex(
        "57 53 53 05 02 01 23 00 04 56 01 07 89 02 00 0A 00"
    )


def test_rss_decodes_interleaved_delay_value_and_unit_fields() -> None:
    pdu = bytes.fromhex(
        "52 53 53 05 02 01 23 00 04 56 01 07 89 02 00 0A 00"
    )

    setup = protocol.parse_rss_pdu(pdu)

    assert setup.enable_mask == 0x05
    assert setup.copy_mask == 0x02
    assert setup.delay_values == [0x0123, 0x0456, 0x0789, 10]
    assert setup.delay_units == [0, 1, 2, 0]


def test_wss_rejects_copying_from_multiple_channels() -> None:
    with pytest.raises(ValueError, match="copy_mask"):
        protocol.pdu_wss(
            enable_mask=0x03,
            copy_mask=0x03,
            delay_values=[0, 0, 0, 0],
            delay_units=[0, 0, 0, 0],
        )


def test_captured_ch1_rsp_frame_from_connected_ts_pump_decodes_exactly() -> None:
    raw = bytes.fromhex(
        "E9 01 13 52 53 50 01 01 18 03 E8 00 04 00 C8 02 "
        "02 26 02 00 01 00 01 58"
    )

    frame = protocol.parse_frame(raw)
    params = protocol.parse_rsp_pdu(frame.pdu)

    assert frame.addr == 1
    assert params.channel == 1
    assert params.mode == 1
    assert params.syringe_code == 0x18
    assert params.dispense_value == 1000
    assert params.dispense_unit == 4
    assert params.infuse_time_value == 200
    assert params.infuse_time_unit == 2
    assert params.withdraw_time_value == 550
    assert params.withdraw_time_unit == 2
    assert params.repeat_count == 1
    assert params.interval_value == 1


def test_captured_stopped_rse_frame_from_connected_ts_pump_decodes_exactly() -> None:
    frame = protocol.parse_frame(bytes.fromhex("E9 01 05 52 53 45 00 00 40"))
    state = protocol.parse_rse_pdu(frame.pdu)

    assert not state.system_running
    assert state.channel_running == [False, False, False, False]
    assert state.q_runstate == 0


def test_wsp_rejects_values_outside_documented_ts_range() -> None:
    with pytest.raises(ValueError, match="dispense_value"):
        protocol.pdu_wsp(
            channel=1,
            mode=1,
            syringe_code=0x18,
            dispense_value=10_000,
            dispense_unit=4,
            infuse_time_value=200,
            infuse_time_unit=2,
            withdraw_time_value=550,
            withdraw_time_unit=2,
            repeat_count=1,
            interval_value=1,
        )
