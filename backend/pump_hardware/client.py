from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

try:
    import serial
except Exception:  # pragma: no cover - runtime dependency
    serial = None

from .config import PumpHardwareConfig, SerialConfig
from . import protocol


class PumpClientError(RuntimeError):
    pass


class NoReplyError(PumpClientError):
    pass


class FrameParseError(PumpClientError):
    pass


class CommandMismatchError(PumpClientError):
    pass


@dataclass(slots=True)
class PDUReply:
    raw_frame: bytes
    cmd: str
    pdu: bytes
    addr: int


class PumpClient:
    def __init__(
        self,
        serial_config: SerialConfig,
        runtime_config: PumpHardwareConfig | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.serial_config = serial_config
        self.runtime_config = runtime_config or PumpHardwareConfig()
        self._logger = logger or (lambda _msg: None)
        self._ser = None
        self._lock = threading.RLock()
        self.connected_parity = serial_config.parity

    def log(self, msg: str) -> None:
        self._logger(msg)

    def is_connected(self) -> bool:
        return bool(self._ser is not None and self._ser.is_open)

    def connect(self, port: str | None = None) -> None:
        if serial is None:
            raise PumpClientError("缺少 pyserial，请先安装: pip install pyserial")
        if self.is_connected():
            return
        if port:
            self.serial_config.port = port
        if not self.serial_config.port:
            raise PumpClientError("串口号为空")

        preferred = str(self.serial_config.parity or "N").upper()
        parities = [preferred]
        if self.serial_config.allow_parity_fallback_n:
            for fallback in ("N", "E"):
                if fallback not in parities:
                    parities.append(fallback)

        parity_map = {
            "E": serial.PARITY_EVEN,
            "N": serial.PARITY_NONE,
        }

        last_err = None
        for p in parities:
            try:
                self._ser = serial.Serial(
                    port=self.serial_config.port,
                    baudrate=int(self.serial_config.baudrate),
                    bytesize=serial.EIGHTBITS,
                    parity=parity_map[p],
                    stopbits=serial.STOPBITS_ONE,
                    timeout=float(self.serial_config.timeout),
                    write_timeout=float(self.serial_config.write_timeout),
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False,
                )
                self.connected_parity = p
                self._reset_input_only()
                self.log(
                    f"[CONNECT][OK] 串口已打开: {self.serial_config.port} @ {self.serial_config.baudrate}bps ({p})"
                )
                return
            except Exception as e:
                last_err = e
                self._ser = None
                self.log(f"[CONNECT][WARN] parity={p} 打开失败: {e}")
        raise PumpClientError(f"串口打开失败: {last_err}")

    def disconnect(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None

    def _require_open(self) -> None:
        if not self.is_connected():
            raise PumpClientError("串口未连接")

    def _reset_input_only(self) -> None:
        if self._ser is None:
            return
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def _read_one_frame(self, timeout: float, idle_timeout: float) -> bytes:
        self._require_open()
        start = time.monotonic()
        while True:
            if time.monotonic() - start > timeout:
                raise NoReplyError("等待帧头超时")
            b = self._ser.read(1)
            if not b:
                continue
            if b[0] == protocol.FLAG:
                break

        raw_body = bytearray()
        decoded = bytearray()
        esc_pending = False
        expected_len = None
        last_rx = time.monotonic()

        while True:
            now = time.monotonic()
            if now - start > timeout:
                if raw_body:
                    return bytes([protocol.FLAG]) + bytes(raw_body)
                raise NoReplyError("读取帧内容超时")

            b = self._ser.read(1)
            if b:
                v = b[0]
                raw_body.append(v)
                last_rx = now
                if esc_pending:
                    if v == 0x00:
                        decoded.append(protocol.ESC)
                    elif v == 0x01:
                        decoded.append(protocol.FLAG)
                    else:
                        raise FrameParseError(f"无效转义序列: E8 {v:02X}")
                    esc_pending = False
                else:
                    if v == protocol.ESC:
                        esc_pending = True
                    else:
                        decoded.append(v)

                if expected_len is None and len(decoded) >= 2:
                    expected_len = int(decoded[1]) + 3
                if expected_len is not None and len(decoded) >= expected_len:
                    return bytes([protocol.FLAG]) + bytes(raw_body)
            else:
                if expected_len is not None and (now - last_rx) >= idle_timeout and len(decoded) >= expected_len:
                    return bytes([protocol.FLAG]) + bytes(raw_body)

    def send_pdu(
        self,
        pdu: bytes,
        expect_cmd: str | None = None,
        allow_no_reply: bool = False,
        retries: int | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
        post_write_delay: float | None = None,
        addr: int | None = None,
    ) -> PDUReply | None:
        self._require_open()
        retries = int(retries if retries is not None else self.runtime_config.retry_count)
        timeout = float(timeout if timeout is not None else self.runtime_config.reply_timeout)
        idle_timeout = float(
            idle_timeout if idle_timeout is not None else self.runtime_config.idle_timeout
        )
        post_write_delay = float(
            post_write_delay if post_write_delay is not None else self.runtime_config.post_write_delay
        )
        addr = int(addr if addr is not None else self.serial_config.address)

        frame = protocol.build_frame(addr=addr, pdu=pdu)
        last_err: Exception | None = None

        for attempt in range(1, retries + 1):
            with self._lock:
                try:
                    self._reset_input_only()
                    self._ser.write(frame)
                    self._ser.flush()
                except Exception as e:
                    last_err = PumpClientError(f"串口发送失败: {e}")
                    if attempt < retries:
                        time.sleep(self.runtime_config.retry_interval)
                        continue
                    raise last_err

                if post_write_delay > 0:
                    time.sleep(post_write_delay)

                if allow_no_reply and expect_cmd is None:
                    return None

                try:
                    raw = self._read_one_frame(timeout=timeout, idle_timeout=idle_timeout)
                    parsed = protocol.parse_frame(raw)
                    if parsed.addr != addr:
                        raise CommandMismatchError(
                            f"应答设备地址不匹配: expect={addr}, got={parsed.addr}"
                        )
                    cmd = protocol.identify_command(parsed.pdu)
                    if expect_cmd and cmd != expect_cmd:
                        raise CommandMismatchError(
                            f"应答命令不匹配: expect={expect_cmd}, got={cmd}, pdu={parsed.pdu.hex(' ').upper()}"
                        )
                    return PDUReply(
                        raw_frame=raw,
                        cmd=cmd,
                        pdu=parsed.pdu,
                        addr=parsed.addr,
                    )
                except NoReplyError as e:
                    last_err = e
                    if allow_no_reply:
                        return None
                except FrameParseError as e:
                    last_err = e
                except CommandMismatchError as e:
                    last_err = e
                except Exception as e:
                    last_err = FrameParseError(f"回包解析失败: {e}")

            if attempt < retries:
                time.sleep(self.runtime_config.retry_interval)

        if allow_no_reply and isinstance(last_err, NoReplyError):
            return None
        if last_err is None:
            raise PumpClientError("未知发送失败")
        raise last_err
