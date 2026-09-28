from __future__ import annotations

from dataclasses import dataclass

from .models import ChannelParams, RunState, SystemSetup

FLAG = 0xE9
ESC = 0xE8

CMD_WSS = b"WSS"
CMD_RSS = b"RSS"
CMD_WSP = b"WSP"
CMD_RSP = b"RSP"
CMD_WSE = b"WSE"
CMD_RSE = b"RSE"

KNOWN_COMMANDS = {
    "WSS": CMD_WSS,
    "RSS": CMD_RSS,
    "WSP": CMD_WSP,
    "RSP": CMD_RSP,
    "WSE": CMD_WSE,
    "RSE": CMD_RSE,
}


@dataclass(slots=True)
class ParsedFrame:
    addr: int
    length: int
    pdu: bytes
    fcs: int
    raw: bytes


def xor_fcs(addr: int, length: int, pdu: bytes) -> int:
    x = (addr & 0xFF) ^ (length & 0xFF)
    for b in pdu:
        x ^= (b & 0xFF)
    return x & 0xFF


def escape(data: bytes) -> bytes:
    out = bytearray()
    for b in data:
        if b == ESC:
            out.extend((ESC, 0x00))
        elif b == FLAG:
            out.extend((ESC, 0x01))
        else:
            out.append(b)
    return bytes(out)


def unescape(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        b = data[i]
        if b != ESC:
            out.append(b)
            i += 1
            continue
        if i + 1 >= len(data):
            raise ValueError("反转义失败: 遇到不完整 ESC 序列")
        nxt = data[i + 1]
        if nxt == 0x00:
            out.append(ESC)
        elif nxt == 0x01:
            out.append(FLAG)
        else:
            raise ValueError(f"反转义失败: 无效序列 E8 {nxt:02X}")
        i += 2
    return bytes(out)


def build_frame(addr: int, pdu: bytes) -> bytes:
    if not (1 <= int(addr) <= 0x1F):
        raise ValueError(f"地址超范围: {addr}")
    if len(pdu) > 0xFF:
        raise ValueError("PDU 长度不能超过 255")
    length = len(pdu)
    fcs = xor_fcs(addr, length, pdu)
    body = bytes([addr & 0xFF, length & 0xFF]) + pdu + bytes([fcs])
    return bytes([FLAG]) + escape(body)


def parse_frame(raw: bytes) -> ParsedFrame:
    if not raw:
        raise ValueError("空帧")
    if raw[0] != FLAG:
        raise ValueError(f"帧头错误: 0x{raw[0]:02X}")

    body = unescape(raw[1:])
    if len(body) < 3:
        raise ValueError("帧过短")

    addr = body[0]
    length = body[1]
    expected = 2 + length + 1
    if len(body) != expected:
        raise ValueError(f"长度不匹配: LEN={length}, 实际PDU长度={len(body)-3}")

    pdu = body[2 : 2 + length]
    fcs = body[-1]
    calc = xor_fcs(addr, length, pdu)
    if calc != fcs:
        raise ValueError(f"FCS 校验失败: recv=0x{fcs:02X}, calc=0x{calc:02X}")

    return ParsedFrame(addr=addr, length=length, pdu=pdu, fcs=fcs, raw=raw)


def identify_command(pdu: bytes) -> str:
    if len(pdu) >= 3:
        head = pdu[:3]
        for name, cmd in KNOWN_COMMANDS.items():
            if head == cmd:
                return name
    return "UNKNOWN"


def pdu_rss() -> bytes:
    return CMD_RSS


def pdu_rse() -> bytes:
    return CMD_RSE


def pdu_rsp(channel: int) -> bytes:
    if not (1 <= int(channel) <= 4):
        raise ValueError(f"通道号非法: {channel}")
    return CMD_RSP + bytes([channel & 0xFF])


def _require_range(name: str, value: int, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} 超范围: {parsed}, 期望 [{minimum}, {maximum}]")
    return parsed


def pdu_wss(
    *,
    enable_mask: int,
    copy_mask: int,
    delay_values: list[int],
    delay_units: list[int],
) -> bytes:
    if len(delay_values) != 4 or len(delay_units) != 4:
        raise ValueError("delay_values / delay_units 必须为长度 4")
    enable_mask = _require_range("enable_mask", enable_mask, 0, 0x0F)
    copy_mask = _require_range("copy_mask", copy_mask, 0, 0x0F)
    if copy_mask and copy_mask & (copy_mask - 1):
        raise ValueError("copy_mask 最多只能选中一个通道")

    # TS 规约 4.4：enable -> copy -> 逐通道 [delay_value(2), delay_unit(1)]。
    # 不能先写完 4 个 value 再写 4 个 unit，否则 CH2 起字段全部错位。
    payload = bytearray(CMD_WSS)
    payload.append(enable_mask)
    payload.append(copy_mask)
    for index, (value, unit) in enumerate(zip(delay_values, delay_units), start=1):
        vv = _require_range(f"delay_values[{index}]", value, 0, 9999)
        uu = _require_range(f"delay_units[{index}]", unit, 0, 2)
        payload.extend(((vv >> 8) & 0xFF, vv & 0xFF))
        payload.append(uu)
    return bytes(payload)


def pdu_wsp(
    channel: int,
    mode: int,
    syringe_code: int,
    dispense_value: int,
    dispense_unit: int,
    infuse_time_value: int,
    infuse_time_unit: int,
    withdraw_time_value: int,
    withdraw_time_unit: int,
    repeat_count: int,
    interval_value: int,
) -> bytes:
    channel = _require_range("channel", channel, 1, 4)
    mode = _require_range("mode", mode, 1, 5)
    syringe_code = _require_range("syringe_code", syringe_code, 0x11, 0x88)
    if not (1 <= (syringe_code >> 4) <= 8 and 1 <= (syringe_code & 0x0F) <= 8):
        raise ValueError(f"syringe_code 高低半字节必须都在 [1, 8]: 0x{syringe_code:02X}")
    dispense_value = _require_range("dispense_value", dispense_value, 1, 9999)
    dispense_unit = _require_range("dispense_unit", dispense_unit, 1, 5)
    infuse_time_value = _require_range("infuse_time_value", infuse_time_value, 1, 9999)
    infuse_time_unit = _require_range("infuse_time_unit", infuse_time_unit, 1, 3)
    withdraw_time_value = _require_range("withdraw_time_value", withdraw_time_value, 1, 9999)
    withdraw_time_unit = _require_range("withdraw_time_unit", withdraw_time_unit, 1, 3)
    repeat_count = _require_range("repeat_count", repeat_count, 1, 999)
    interval_value = _require_range("interval_value", interval_value, 1, 9999)
    payload = bytearray(CMD_WSP)
    payload.extend(
        [
            channel & 0xFF,
            mode & 0xFF,
            syringe_code & 0xFF,
            (dispense_value >> 8) & 0xFF,
            dispense_value & 0xFF,
            dispense_unit & 0xFF,
            (infuse_time_value >> 8) & 0xFF,
            infuse_time_value & 0xFF,
            infuse_time_unit & 0xFF,
            (withdraw_time_value >> 8) & 0xFF,
            withdraw_time_value & 0xFF,
            withdraw_time_unit & 0xFF,
            (repeat_count >> 8) & 0xFF,
            repeat_count & 0xFF,
            (interval_value >> 8) & 0xFF,
            interval_value & 0xFF,
        ]
    )
    return bytes(payload)


def pdu_wse(sys_runstate: int, q_runstate: int) -> bytes:
    sys_runstate = _require_range("sys_runstate", sys_runstate, 0, 0x1F)
    q_runstate = _require_range("q_runstate", q_runstate, 0, 0xFF)
    return CMD_WSE + bytes([sys_runstate, q_runstate])


def parse_rss_pdu(pdu: bytes) -> SystemSetup:
    if len(pdu) != 17 or pdu[:3] != CMD_RSS:
        raise ValueError("RSS PDU 非法")
    # TS 规约 4.5：enable -> copy -> 逐通道 [delay_value(2), delay_unit(1)]。
    enable_mask = pdu[3]
    copy_mask = pdu[4]
    delay_values: list[int] = []
    delay_units: list[int] = []
    for channel_index in range(4):
        offset = 5 + channel_index * 3
        delay_values.append((pdu[offset] << 8) | pdu[offset + 1])
        delay_units.append(pdu[offset + 2])
    return SystemSetup(
        enable_mask=enable_mask,
        copy_mask=copy_mask,
        delay_values=delay_values,
        delay_units=delay_units,
    )


def parse_rse_pdu(pdu: bytes) -> RunState:
    if len(pdu) != 5 or pdu[:3] != CMD_RSE:
        raise ValueError("RSE PDU 非法")
    sys_state = pdu[3]
    q_state = pdu[4]
    channel_running = [
        bool(sys_state & 0x02),
        bool(sys_state & 0x04),
        bool(sys_state & 0x08),
        bool(sys_state & 0x10),
    ]
    return RunState(
        sys_runstate=sys_state,
        q_runstate=q_state,
        system_running=bool(sys_state & 0x01),
        channel_running=channel_running,
    )


def parse_rsp_pdu(pdu: bytes) -> ChannelParams:
    if len(pdu) != 19 or pdu[:3] != CMD_RSP:
        raise ValueError("RSP PDU 非法")
    return ChannelParams(
        channel=pdu[3],
        mode=pdu[4],
        syringe_code=pdu[5],
        dispense_value=(pdu[6] << 8) | pdu[7],
        dispense_unit=pdu[8],
        infuse_time_value=(pdu[9] << 8) | pdu[10],
        infuse_time_unit=pdu[11],
        withdraw_time_value=(pdu[12] << 8) | pdu[13],
        withdraw_time_unit=pdu[14],
        repeat_count=(pdu[15] << 8) | pdu[16],
        interval_value=(pdu[17] << 8) | pdu[18],
    )
