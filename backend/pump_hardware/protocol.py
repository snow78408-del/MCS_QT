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


def pdu_wss(copy_mask: int, enable_mask: int, delay_values: list[int], delay_units: list[int]) -> bytes:
    if len(delay_values) != 4 or len(delay_units) != 4:
        raise ValueError("delay_values / delay_units 必须为长度 4")
    # TS protocol order: enable_mask -> copy_mask -> 4x delay_value -> 4x delay_unit.
    payload = bytearray(CMD_WSS)
    payload.append(enable_mask & 0xFF)
    payload.append(copy_mask & 0xFF)
    for v in delay_values:
        vv = int(v) & 0xFFFF
        payload.extend(((vv >> 8) & 0xFF, vv & 0xFF))
    for u in delay_units:
        payload.append(int(u) & 0xFF)
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
    if not (1 <= int(channel) <= 4):
        raise ValueError(f"通道号非法: {channel}")
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
    return CMD_WSE + bytes([sys_runstate & 0xFF, q_runstate & 0xFF])


def parse_rss_pdu(pdu: bytes) -> SystemSetup:
    if len(pdu) != 17 or pdu[:3] != CMD_RSS:
        raise ValueError("RSS PDU 非法")
    # TS protocol order: enable_mask -> copy_mask -> 4x delay_value -> 4x delay_unit.
    enable_mask = pdu[3]
    copy_mask = pdu[4]
    delay_values = [
        (pdu[5] << 8) | pdu[6],
        (pdu[7] << 8) | pdu[8],
        (pdu[9] << 8) | pdu[10],
        (pdu[11] << 8) | pdu[12],
    ]
    delay_units = [pdu[13], pdu[14], pdu[15], pdu[16]]
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
