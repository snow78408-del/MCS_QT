from __future__ import annotations

import time
from typing import Callable

from . import protocol
from .client import CommandMismatchError, FrameParseError, NoReplyError, PumpClient
from .config import PumpHardwareConfig, SerialConfig
from .models import (
    ChannelParams,
    FlowUpdateResult,
    PumpConnectionState,
    PumpOperationResult,
    RunState,
    SystemSetup,
)


class PumpHardwareService:
    """TS 注射泵硬件服务层，供 orchestrator / PID 调用。"""

    # The UI and controller use uL/min. Generate WSP parameters in the same
    # user-facing flow-rate domain first: uL volume over 0.1 min time units.
    # The write is considered successful only after RSP readback converts back
    # to the requested uL/min target.
    _VOLUME_UNIT_TO_UL = {
        1: 0.001,
        2: 0.01,
        3: 0.1,
        4: 1.0,
        5: 10.0,
    }
    _TIME_UNIT_TO_MIN = {
        1: 1.0 / 600.0,  # 0.1 s
        2: 0.1,
        3: 6.0,
    }
    _DEFAULT_SYRINGE_CODE = 0x18
    _DEFAULT_DISPENSE_VALUE = 1000
    _DEFAULT_DISPENSE_UNIT = 4
    _DEFAULT_INFUSE_TIME_UNIT = 2
    _DEFAULT_WITHDRAW_TIME_VALUE = 100
    # Withdraw time is unrelated to the requested infusion flow display. Keep
    # it on the known-good TS time unit; using volume-unit code here makes some
    # firmware reject the whole WSP command.
    _DEFAULT_WITHDRAW_TIME_UNIT = 1
    _DEFAULT_REPEAT_COUNT = 10
    _DEFAULT_INTERVAL_VALUE = 50
    _MIN_INFUSE_TIME_VALUE = 10
    _MAX_PARAM_VALUE = 65535

    def __init__(
        self,
        serial_config: SerialConfig | None = None,
        runtime_config: PumpHardwareConfig | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.serial_config = serial_config or SerialConfig()
        self.runtime_config = runtime_config or PumpHardwareConfig()
        self._logger = logger or (lambda _msg: None)
        self.client = PumpClient(
            serial_config=self.serial_config,
            runtime_config=self.runtime_config,
            logger=self._logger,
        )
        self.connection_state = PumpConnectionState()
        self.last_system_setup: SystemSetup | None = None
        self.last_run_state: RunState | None = None
        self.last_channel_params: dict[int, ChannelParams] = {}

    def log(self, msg: str) -> None:
        self._logger(msg)

    @classmethod
    def _volume_unit_to_ul(cls, unit: int) -> float:
        return float(cls._VOLUME_UNIT_TO_UL.get(int(unit), 1.0))

    @classmethod
    def _time_unit_to_min(cls, unit: int) -> float:
        return float(cls._TIME_UNIT_TO_MIN.get(int(unit), 1.0))

    @classmethod
    def flow_from_channel_params(cls, params: ChannelParams | None) -> float | None:
        if params is None:
            return None
        try:
            volume_ul = float(params.dispense_value) * cls._volume_unit_to_ul(int(params.dispense_unit))
            time_min = float(params.infuse_time_value) * cls._time_unit_to_min(int(params.infuse_time_unit))
            if time_min <= 0.0:
                return None
            return volume_ul / time_min
        except Exception:
            return None

    @staticmethod
    def ul_min_to_nl_sec(flow_ul_min: float) -> float:
        return float(flow_ul_min) * 1000.0 / 60.0

    @classmethod
    def flow_nl_sec_from_channel_params(cls, params: ChannelParams | None) -> float | None:
        flow_ul_min = cls.flow_from_channel_params(params)
        if flow_ul_min is None:
            return None
        return cls.ul_min_to_nl_sec(flow_ul_min)

    @classmethod
    def _infuse_time_value_for_flow(
        cls,
        *,
        dispense_value: int,
        dispense_unit: int,
        infuse_time_unit: int,
        flow_ul_min: float,
    ) -> int:
        safe_q = max(float(flow_ul_min), 1e-6)
        volume_ul = max(1, int(dispense_value)) * cls._volume_unit_to_ul(int(dispense_unit))
        time_unit_min = cls._time_unit_to_min(int(infuse_time_unit))
        raw_time_value = volume_ul / safe_q / max(time_unit_min, 1e-9)
        return max(1, min(cls._MAX_PARAM_VALUE, int(round(raw_time_value))))

    @classmethod
    def _raw_infuse_time_value_for_flow(
        cls,
        *,
        dispense_value: int,
        dispense_unit: int,
        infuse_time_unit: int,
        flow_ul_min: float,
    ) -> float:
        safe_q = max(float(flow_ul_min), 1e-6)
        volume_ul = max(1, int(dispense_value)) * cls._volume_unit_to_ul(int(dispense_unit))
        time_unit_min = cls._time_unit_to_min(int(infuse_time_unit))
        return volume_ul / safe_q / max(time_unit_min, 1e-9)

    @classmethod
    def _dispense_and_infuse_for_flow(
        cls,
        *,
        dispense_value: int,
        dispense_unit: int,
        infuse_time_unit: int,
        flow_ul_min: float,
    ) -> tuple[int, int]:
        safe_q = max(float(flow_ul_min), 1e-6)
        dispense_unit_ul = cls._volume_unit_to_ul(int(dispense_unit))
        time_unit_min = cls._time_unit_to_min(int(infuse_time_unit))
        dispense_value = max(1, min(cls._MAX_PARAM_VALUE, int(dispense_value)))

        raw_time_value = cls._raw_infuse_time_value_for_flow(
            dispense_value=dispense_value,
            dispense_unit=dispense_unit,
            infuse_time_unit=infuse_time_unit,
            flow_ul_min=safe_q,
        )
        if raw_time_value > cls._MAX_PARAM_VALUE:
            target_dispense = int(
                safe_q * cls._MAX_PARAM_VALUE * max(time_unit_min, 1e-9) / max(dispense_unit_ul, 1e-9)
            )
            dispense_value = max(1, min(cls._MAX_PARAM_VALUE, target_dispense))

        infuse_time_value = cls._infuse_time_value_for_flow(
            dispense_value=dispense_value,
            dispense_unit=dispense_unit,
            infuse_time_unit=infuse_time_unit,
            flow_ul_min=safe_q,
        )
        if infuse_time_value >= cls._MIN_INFUSE_TIME_VALUE:
            return dispense_value, infuse_time_value

        # Some TS pump firmware ignores very small infuse_time_value fields.
        # Preserve the requested uL/min by raising dispense volume instead.
        min_time_value = cls._MIN_INFUSE_TIME_VALUE
        target_dispense = int(round(safe_q * min_time_value * max(time_unit_min, 1e-9) / max(dispense_unit_ul, 1e-9)))
        dispense_value = max(1, min(cls._MAX_PARAM_VALUE, target_dispense))
        infuse_time_value = cls._infuse_time_value_for_flow(
            dispense_value=dispense_value,
            dispense_unit=dispense_unit,
            infuse_time_unit=infuse_time_unit,
            flow_ul_min=safe_q,
        )
        return dispense_value, max(cls._MIN_INFUSE_TIME_VALUE, infuse_time_value)

    @staticmethod
    def _ok(parsed=None, raw: bytes | None = None, verified: bool = False, reason: str | None = None):
        return PumpOperationResult(
            ok=True,
            parsed_reply=parsed,
            raw_reply=raw,
            verified=verified,
            reason=reason,
        )

    @staticmethod
    def _fail(error: Exception | str, parsed=None, raw: bytes | None = None, reason: str | None = None):
        return PumpOperationResult(
            ok=False,
            error=str(error),
            parsed_reply=parsed,
            raw_reply=raw,
            verified=False,
            reason=reason or str(error),
        )

    def _send(self, pdu: bytes, expect_cmd: str, allow_no_reply: bool = False):
        return self.client.send_pdu(
            pdu=pdu,
            expect_cmd=expect_cmd,
            allow_no_reply=allow_no_reply,
            retries=self.runtime_config.retry_count,
            timeout=self.runtime_config.reply_timeout,
            idle_timeout=self.runtime_config.idle_timeout,
            post_write_delay=self.runtime_config.post_write_delay,
            addr=self.serial_config.address,
        )

    def connect_and_probe(self) -> PumpConnectionState:
        preferred = str(self.serial_config.parity or "N").strip().upper()
        candidates: list[str] = []
        for parity in (preferred, "E", "N"):
            if parity in {"E", "N"} and parity not in candidates:
                candidates.append(parity)

        best_state = PumpConnectionState(serial_connected=False, comm_established=False, fully_ready=False)
        for parity in candidates:
            self.client.disconnect()
            self.serial_config.parity = parity
            state = PumpConnectionState(serial_connected=False, comm_established=False, fully_ready=False)
            try:
                self.client.connect()
                state.serial_connected = self.client.is_connected()
            except Exception as e:
                state.failed["serial"] = str(e)
                self.log(f"[CONNECT][FAIL] parity={parity} {e}")
                if not best_state.serial_connected:
                    best_state = state
                continue

            self.log(f"[CONNECT][PROBE] parity={parity} start")
            self._probe_current_connection(state)
            if state.fully_ready:
                self.connection_state = state
                self.log(f"[CONNECT][OK] parity={parity} 串口连接、通信建立、设备完全就绪")
                return state
            if state.comm_established:
                self.connection_state = state
                self.log(f"[CONNECT][OK] parity={parity} 通信已建立但设备未完全就绪: failed={state.failed}")
                return state
            self.log(f"[CONNECT][WARN] parity={parity} 串口已开但通信未建立: failed={state.failed}")
            if not best_state.comm_established:
                best_state = state

        self.connection_state = best_state
        if best_state.serial_connected:
            self.log(f"[CONNECT][FAIL] 串口已开但通信未建立: failed={best_state.failed}")
        else:
            self.log(f"[CONNECT][FAIL] 串口打开失败: failed={best_state.failed}")
        return best_state

    def _probe_current_connection(self, state: PumpConnectionState) -> None:
        probe_results: list[tuple[str, PumpOperationResult]] = []
        probe_results.append(("RSS", self.read_rss()))
        time.sleep(self.runtime_config.probe_step_delay)
        probe_results.append(("RSE", self.read_rse()))
        time.sleep(self.runtime_config.probe_step_delay)
        for ch in (1, 2, 3, 4):
            probe_results.append((f"RSP{ch}", self.read_rsp(ch)))
            time.sleep(self.runtime_config.probe_step_delay)

        for key, res in probe_results:
            if res.ok:
                state.succeeded.append(key)
            else:
                state.failed[key] = res.error or "unknown"

        state.comm_established = any(k in state.succeeded for k in ("RSS", "RSE", "RSP1", "RSP2", "RSP3", "RSP4"))
        state.fully_ready = (
            "RSS" in state.succeeded
            and "RSE" in state.succeeded
            and all(f"RSP{c}" in state.succeeded for c in (1, 2, 3, 4))
        )

    def disconnect(self) -> None:
        self.client.disconnect()
        self.connection_state = PumpConnectionState()
        self.log("[CONNECT][OK] 串口已断开")

    def read_rss(self) -> PumpOperationResult:
        try:
            rep = self._send(protocol.pdu_rss(), expect_cmd="RSS")
            setup = protocol.parse_rss_pdu(rep.pdu)
            self.last_system_setup = setup
            self.log(
                f"[RSS][OK] copy=0x{setup.copy_mask:02X}, enable=0x{setup.enable_mask:02X}, "
                f"delay={setup.delay_values}, unit={setup.delay_units}"
            )
            return self._ok(parsed=setup, raw=rep.raw_frame, verified=True)
        except Exception as e:
            self.log(f"[RSS][FAIL] {e}")
            return self._fail(e)

    def read_rss_with_retry(self, attempts: int | None = None) -> PumpOperationResult:
        """Read pump setup with bounded retries for the slow serial link."""
        total = max(1, int(attempts if attempts is not None else self.runtime_config.retry_count + 1))
        last = self._fail("RSS 未执行")
        for attempt in range(1, total + 1):
            last = self.read_rss()
            if last.ok:
                if attempt > 1:
                    self.log(f"[RSS][RECOVERED] attempt={attempt}/{total}")
                return last
            if attempt < total:
                self.log(
                    f"[RSS][RETRY] attempt={attempt}/{total} "
                    f"reason={last.error or last.reason or 'unknown'}"
                )
                time.sleep(max(0.12, float(self.runtime_config.retry_interval)))
        return last

    def read_rse(self) -> PumpOperationResult:
        try:
            rep = self._send(protocol.pdu_rse(), expect_cmd="RSE")
            run_state = protocol.parse_rse_pdu(rep.pdu)
            self.last_run_state = run_state
            self.log(
                f"[RSE][OK] sys=0x{run_state.sys_runstate:02X}, q=0x{run_state.q_runstate:02X}, "
                f"running={run_state.channel_running}"
            )
            return self._ok(parsed=run_state, raw=rep.raw_frame, verified=True)
        except Exception as e:
            self.log(f"[RSE][FAIL] {e}")
            return self._fail(e)

    def read_rsp(self, channel: int) -> PumpOperationResult:
        try:
            rep = self._send(protocol.pdu_rsp(channel), expect_cmd="RSP")
            params = protocol.parse_rsp_pdu(rep.pdu)
            if params.channel != channel:
                raise ValueError(f"RSP 通道号不一致: expect={channel}, got={params.channel}")
            self.last_channel_params[channel] = params
            flow = self.flow_from_channel_params(params)
            self.log(
                f"[RSP][OK][CH{channel}] mode={params.mode}, sid={params.syringe_code}, "
                f"dispense={params.dispense_value}/{params.dispense_unit}, "
                f"infuse={params.infuse_time_value}/{params.infuse_time_unit}, "
                f"flow={(flow if flow is not None else 0.0):.6f}uL/min"
            )
            return self._ok(parsed=params, raw=rep.raw_frame, verified=True)
        except Exception as e:
            self.log(f"[RSP][FAIL][CH{channel}] {e}")
            return self._fail(e)

    def write_wss(self, setup: SystemSetup) -> PumpOperationResult:
        pdu = protocol.pdu_wss(
            copy_mask=setup.copy_mask,
            enable_mask=setup.enable_mask,
            delay_values=setup.delay_values,
            delay_units=setup.delay_units,
        )
        try:
            rep = self._send(pdu, expect_cmd="WSS", allow_no_reply=True)
            raw = rep.raw_frame if rep is not None else None
            self.log("[WSS][OK] 命令发送完成")
            return self._ok(parsed=setup, raw=raw, verified=False)
        except (NoReplyError, FrameParseError, CommandMismatchError) as e:
            self.log(f"[WSS][FAIL] {e}")
            return self._fail(e)
        except Exception as e:
            self.log(f"[WSS][FAIL] {e}")
            return self._fail(e)

    def write_wss_and_verify(self, setup: SystemSetup) -> PumpOperationResult:
        wr = self.write_wss(setup)
        if not wr.ok:
            return wr

        rd = self.read_rss()
        if not rd.ok:
            return self._fail(f"WSS 写入后 RSS 回读失败: {rd.error}")

        got: SystemSetup = rd.parsed_reply
        def _effective_enable(s: SystemSetup) -> int:
            return (int(s.enable_mask) | int(s.copy_mask)) & 0x0F

        mismatch = []
        if bin(int(setup.enable_mask) & 0x0F).count("1") >= 2:
            # 多通道设备上，RSS 有时把位分散在 enable/copy 中返回。
            exp = int(setup.enable_mask) & 0x0F
            got_eff = _effective_enable(got)
            if (got_eff & exp) != exp:
                mismatch.append(
                    f"enable_mask expect=0x{exp:02X}, got_enable=0x{int(got.enable_mask) & 0x0F:02X}, "
                    f"got_copy=0x{int(got.copy_mask) & 0x0F:02X}, effective=0x{got_eff:02X}"
                )
        else:
            if got.enable_mask != setup.enable_mask:
                mismatch.append(f"enable_mask expect=0x{setup.enable_mask:02X}, got=0x{got.enable_mask:02X}")
            if got.copy_mask != setup.copy_mask:
                mismatch.append(f"copy_mask expect=0x{setup.copy_mask:02X}, got=0x{got.copy_mask:02X}")
        if got.delay_values != setup.delay_values:
            mismatch.append(f"delay_values expect={setup.delay_values}, got={got.delay_values}")
        if got.delay_units != setup.delay_units:
            mismatch.append(f"delay_units expect={setup.delay_units}, got={got.delay_units}")

        if not mismatch:
            self.log("[WSS][OK] 写后读回校验通过")
            return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSS 校验通过")

        if self.runtime_config.wss_swap_fallback:
            self.log("[WSS][WARN] 主顺序校验失败，尝试 fallback(enable->copy)")
            try:
                payload = bytearray(protocol.CMD_WSS)
                payload.append(setup.copy_mask & 0xFF)
                payload.append(setup.enable_mask & 0xFF)
                for v in setup.delay_values:
                    vv = int(v) & 0xFFFF
                    payload.extend(((vv >> 8) & 0xFF, vv & 0xFF))
                for u in setup.delay_units:
                    payload.append(int(u) & 0xFF)

                rep = self._send(bytes(payload), expect_cmd="WSS", allow_no_reply=True)
                _ = rep.raw_frame if rep is not None else None
                rd2 = self.read_rss()
                if rd2.ok and rd2.parsed_reply is not None:
                    got2: SystemSetup = rd2.parsed_reply
                    if (
                        got2.enable_mask == setup.enable_mask
                        and got2.copy_mask == setup.copy_mask
                        and got2.delay_values == setup.delay_values
                        and got2.delay_units == setup.delay_units
                    ):
                        self.log("[WSS][OK] fallback 顺序校验通过")
                        return self._ok(parsed=got2, raw=rd2.raw_reply, verified=True, reason="WSS fallback 校验通过")
            except Exception as e:
                self.log(f"[WSS][WARN] fallback 执行异常: {e}")

        reason = "; ".join(mismatch)
        self.log(f"[WSS][FAIL] 回读不一致: {reason}")
        return self._fail("WSS 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)

    def write_wsp(self, params: ChannelParams) -> PumpOperationResult:
        pdu = protocol.pdu_wsp(
            channel=params.channel,
            mode=params.mode,
            syringe_code=params.syringe_code,
            dispense_value=params.dispense_value,
            dispense_unit=params.dispense_unit,
            infuse_time_value=params.infuse_time_value,
            infuse_time_unit=params.infuse_time_unit,
            withdraw_time_value=params.withdraw_time_value,
            withdraw_time_unit=params.withdraw_time_unit,
            repeat_count=params.repeat_count,
            interval_value=params.interval_value,
        )
        try:
            self.log(f"[WSP][TX][CH{params.channel}] pdu={pdu.hex(' ').upper()}")
            rep = self._send(pdu, expect_cmd="WSP", allow_no_reply=True)
            raw = rep.raw_frame if rep is not None else None
            if rep is None:
                self.log(f"[WSP][NO_REPLY][CH{params.channel}] no ack; will verify by RSP readback")
            else:
                raw_text = raw.hex(" ").upper() if raw else "None"
                reply_text = rep.pdu.hex(" ").upper()
                self.log(f"[WSP][OK][CH{params.channel}] raw={raw_text} pdu={reply_text}")
            return self._ok(parsed=params, raw=raw, verified=False)
        except Exception as e:
            self.log(f"[WSP][FAIL][CH{params.channel}] {e}")
            return self._fail(e)

    def write_wsp_and_verify(self, channel: int, params: ChannelParams) -> PumpOperationResult:
        wr = self.write_wsp(params)
        if not wr.ok:
            return wr

        commit = self.commit_channel_params(channel)
        if not commit.ok:
            self.log(f"[WSP][WARN][CH{channel}] 参数提交触发失败: {commit.reason or commit.error}")

        retries = max(1, int(self.runtime_config.wsp_verify_read_retry))
        for idx in range(retries):
            rd = self.read_rsp(channel)
            if not rd.ok:
                if idx < retries - 1:
                    time.sleep(float(self.runtime_config.wsp_verify_retry_interval))
                    continue
                return self._fail(f"WSP 写入后 RSP 回读失败: {rd.error}")

            got: ChannelParams = rd.parsed_reply
            mismatch = []
            strict_fields = [
                "channel",
                "mode",
                "syringe_code",
                "dispense_value",
                "dispense_unit",
                "infuse_time_value",
                "infuse_time_unit",
                "withdraw_time_value",
                "withdraw_time_unit",
                "repeat_count",
                "interval_value",
            ]

            for name in strict_fields:
                if int(getattr(got, name)) != int(getattr(params, name)):
                    mismatch.append(f"{name} expect={getattr(params, name)}, got={getattr(got, name)}")

            if not mismatch:
                expected_flow = self.flow_from_channel_params(params)
                actual_flow = self.flow_from_channel_params(got)
                if expected_flow is None or actual_flow is None:
                    reason = "无法从 WSP/RSP 参数换算 uL/min 流速"
                    self.log(f"[VERIFY][FAIL][CH{channel}] {reason}")
                    return self._fail("WSP 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)

                delta = abs(float(actual_flow) - float(expected_flow))
                allowed = max(0.01, abs(float(expected_flow)) * 0.002)
                if delta > allowed:
                    reason = (
                        f"flow expect={expected_flow:.6f}uL/min, "
                        f"got={actual_flow:.6f}uL/min, delta={delta:.6f}"
                    )
                    self.log(f"[VERIFY][FAIL][CH{channel}] {reason}")
                    return self._fail("WSP 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)

                self.log(
                    f"[VERIFY][OK][CH{channel}] WSP 写后读回一致, "
                    f"flow={actual_flow:.6f}uL/min, "
                    f"panel_equiv={self.ul_min_to_nl_sec(actual_flow):.6f}nL/sec"
                )
                return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSP 校验通过")

            if idx < retries - 1:
                time.sleep(float(self.runtime_config.wsp_verify_retry_interval))
                continue

            reason = "; ".join(mismatch)
            self.log(f"[VERIFY][FAIL][CH{channel}] {reason}")
            return self._fail("WSP 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)

        return self._fail("WSP 校验失败：未知错误")

    def commit_channel_params(self, channel: int) -> PumpOperationResult:
        if not (1 <= int(channel) <= 4):
            return self._fail(f"非法通道: {channel}")
        rss = self.read_rss()
        if not rss.ok or rss.parsed_reply is None:
            return self._fail(f"参数提交前读取 RSS 失败: {rss.error or rss.reason}")

        setup: SystemSetup = rss.parsed_reply
        bit = 1 << (int(channel) - 1)
        req = SystemSetup(
            enable_mask=int(setup.enable_mask) & 0x0F,
            copy_mask=(int(setup.copy_mask) | bit) & 0x0F,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        res = self.write_wss(req)
        if res.ok:
            self.log(
                f"[WSP][COMMIT][CH{channel}] copy=0x{req.copy_mask:02X}, "
                f"enable=0x{req.enable_mask:02X}"
            )
            time.sleep(float(self.runtime_config.wsp_verify_retry_interval))
            return res
        return res

    def write_wse(self, sys_runstate: int, q_runstate: int = 0x00) -> PumpOperationResult:
        pdu = protocol.pdu_wse(sys_runstate=sys_runstate, q_runstate=q_runstate)
        try:
            rep = self._send(pdu, expect_cmd="WSE", allow_no_reply=True)
            raw = rep.raw_frame if rep is not None else None
            self.log(f"[WSE][OK] sys=0x{sys_runstate:02X}, q=0x{q_runstate:02X}")
            return self._ok(raw=raw, parsed={"sys_runstate": sys_runstate, "q_runstate": q_runstate})
        except Exception as e:
            self.log(f"[WSE][FAIL] {e}")
            return self._fail(e)

    def write_wse_and_verify(self, sys_runstate: int, q_runstate: int = 0x00) -> PumpOperationResult:
        wr = self.write_wse(sys_runstate=sys_runstate, q_runstate=q_runstate)
        if not wr.ok:
            return wr
        rd = self.read_rse()
        if not rd.ok:
            return self._fail(f"WSE 写入后 RSE 回读失败: {rd.error}")
        got: RunState = rd.parsed_reply
        if int(got.sys_runstate) != (int(sys_runstate) & 0xFF):
            reason = f"sys_runstate expect=0x{sys_runstate:02X}, got=0x{got.sys_runstate:02X}"
            self.log(f"[WSE][FAIL] {reason}")
            return self._fail("WSE 回读校验失败", parsed=got, raw=rd.raw_reply, reason=reason)
        self.log("[WSE][OK] 写后读回校验通过")
        return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSE 校验通过")

    def enable_channels(self, mask: int) -> PumpOperationResult:
        rss = self.read_rss()
        if not rss.ok:
            return self._fail(f"enable_channels 前读取 RSS 失败: {rss.error}")
        setup: SystemSetup = rss.parsed_reply
        enable = int(mask) & 0x0F
        copy_mask = setup.copy_mask & 0x0F
        if enable == 0:
            copy_mask = 0
        elif (copy_mask & enable) != enable:
            # 多通道使能时，copy_mask 必须覆盖全部目标通道，否则设备可能只保留最低位通道。
            copy_mask = enable
        req = SystemSetup(
            enable_mask=enable,
            copy_mask=copy_mask & 0x0F,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        return self.write_wss(req)

    def enable_channels_and_verify(self, mask: int) -> PumpOperationResult:
        def _effective(enable_mask: int, copy_mask: int) -> int:
            return (int(enable_mask) | int(copy_mask)) & 0x0F

        rss = self.read_rss_with_retry()
        if not rss.ok:
            return self._fail(f"enable_channels_and_verify 前读取 RSS 失败: {rss.error}")
        setup: SystemSetup = rss.parsed_reply
        enable = int(mask) & 0x0F
        copy_mask = setup.copy_mask & 0x0F
        if enable == 0:
            copy_mask = 0
        elif (copy_mask & enable) != enable:
            # 多通道使能时，copy_mask 必须覆盖全部目标通道，否则设备可能只保留最低位通道。
            copy_mask = enable
        req = SystemSetup(
            enable_mask=enable,
            copy_mask=copy_mask & 0x0F,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        first = self.write_wss_and_verify(req)
        if first.ok:
            return first

        # 某些设备在多通道使能时需要“逐位收敛”，例如先 0x01 再到 0x03。
        if bin(enable).count("1") < 2:
            return first

        rd = self.read_rss()
        if not rd.ok or rd.parsed_reply is None:
            return first
        cur_setup: SystemSetup = rd.parsed_reply
        cur_enable = _effective(cur_setup.enable_mask, cur_setup.copy_mask)
        if (cur_enable & enable) == enable:
            self.log("[WSS][OK] 多通道收敛前已满足目标使能")
            return self._ok(parsed=cur_setup, raw=rd.raw_reply, verified=True, reason="WSS 多通道收敛校验通过")

        missing = enable & (~cur_enable & 0x0F)
        for bit_idx in range(4):
            bit = 1 << bit_idx
            if (missing & bit) == 0:
                continue
            target_enable = (cur_enable | bit) & 0x0F
            step = SystemSetup(
                enable_mask=target_enable,
                # 使用目标掩码覆盖 copy，避免 0x01/0x02 互相覆盖。
                copy_mask=target_enable & 0x0F,
                delay_values=list(cur_setup.delay_values),
                delay_units=list(cur_setup.delay_units),
            )
            step_res = self.write_wss_and_verify(step)
            if not step_res.ok:
                self.log(f"[WSS][WARN] 多通道收敛步骤失败: bit=0x{bit:02X}, reason={step_res.reason or step_res.error}")

            rd2 = self.read_rss()
            if rd2.ok and rd2.parsed_reply is not None:
                cur_setup = rd2.parsed_reply
                cur_enable = _effective(cur_setup.enable_mask, cur_setup.copy_mask)
                if (cur_enable & enable) == enable:
                    self.log("[WSS][OK] 多通道收敛校验通过")
                    return self._ok(
                        parsed=cur_setup,
                        raw=rd2.raw_reply,
                        verified=True,
                        reason="WSS 多通道收敛校验通过",
                    )

        final_rd = self.read_rss()
        if final_rd.ok and final_rd.parsed_reply is not None:
            final_setup: SystemSetup = final_rd.parsed_reply
            final_enable = _effective(final_setup.enable_mask, final_setup.copy_mask)
            if (final_enable & enable) == enable:
                self.log("[WSS][OK] 多通道收敛最终校验通过")
                return self._ok(
                    parsed=final_setup,
                    raw=final_rd.raw_reply,
                    verified=True,
                    reason="WSS 多通道收敛最终校验通过",
                )
            self.log(
                f"[WSS][FAIL] 多通道收敛后仍未满足: expect=0x{enable:02X}, "
                f"got_enable=0x{int(final_setup.enable_mask) & 0x0F:02X}, "
                f"got_copy=0x{int(final_setup.copy_mask) & 0x0F:02X}, "
                f"effective=0x{final_enable:02X}"
            )
            return self._fail(
                "WSS 多通道收敛失败",
                parsed=final_setup,
                raw=final_rd.raw_reply,
                reason=(
                    f"enable_mask expect=0x{enable:02X}, "
                    f"got_enable=0x{int(final_setup.enable_mask) & 0x0F:02X}, "
                    f"got_copy=0x{int(final_setup.copy_mask) & 0x0F:02X}, "
                    f"effective=0x{final_enable:02X}"
                ),
            )

        return first

    def stop_system(self) -> PumpOperationResult:
        return self.write_wse(sys_runstate=0x00, q_runstate=0x00)

    def stop_system_and_verify(self) -> PumpOperationResult:
        return self.write_wse_and_verify(sys_runstate=0x00, q_runstate=0x00)

    def prepare_parameter_write(self, mask: int) -> PumpOperationResult:
        target = int(mask) & 0x0F
        stop = self.stop_system_and_verify()
        if not stop.ok:
            return self._fail(f"参数写入前停泵失败: {stop.reason or stop.error}")

        rss = self.read_rss()
        if not rss.ok or rss.parsed_reply is None:
            return self._fail(f"参数写入前读取 RSS 失败: {rss.error or rss.reason}")

        setup: SystemSetup = rss.parsed_reply
        req = SystemSetup(
            enable_mask=(int(setup.enable_mask) | target) & 0x0F,
            copy_mask=(int(setup.copy_mask) | target) & 0x0F,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        res = self.write_wss_and_verify(req)
        if res.ok:
            # write_wss_and_verify() has already completed an RSS readback and
            # validated the requested masks.  A second immediate RSS query is
            # redundant and, on the slow 1200-bps pump link, can intermittently
            # time out after an otherwise successful write.  Trust the verified
            # result so initialization is not rejected because of that extra
            # diagnostic read.
            got = res.parsed_reply if isinstance(res.parsed_reply, SystemSetup) else req
            self.log(
                f"[PUMP][PARAM_WRITE][READY] mask=0x{target:02X} "
                f"enable=0x{int(got.enable_mask) & 0x0F:02X} copy=0x{int(got.copy_mask) & 0x0F:02X}"
            )
            return self._ok(
                parsed=got,
                raw=res.raw_reply,
                verified=True,
                reason="参数写入前通道已进入可写状态",
            )

        # If the combined write/readback verification failed, one recovery
        # read may still show that the pump accepted the command.  This keeps
        # the existing tolerance for devices whose first response is corrupt.
        rd = self.read_rss()
        if rd.ok and rd.parsed_reply is not None:
            got: SystemSetup = rd.parsed_reply
            effective = (int(got.enable_mask) | int(got.copy_mask)) & 0x0F
            if (effective & target) == target:
                self.log(
                    f"[PUMP][PARAM_WRITE][READY] mask=0x{target:02X} "
                    f"enable=0x{int(got.enable_mask) & 0x0F:02X} copy=0x{int(got.copy_mask) & 0x0F:02X}"
                )
                return self._ok(
                    parsed=got,
                    raw=rd.raw_reply,
                    verified=bool(res.ok),
                    reason="参数写入前通道已进入可写状态",
                )

        reason = res.error or res.reason or (rd.error if rd else None) or (rd.reason if rd else None) or "unknown"
        self.log(f"[PUMP][PARAM_WRITE][FAIL] enable/copy prepare failed: {reason}")
        return self._fail("参数写入前通道准备失败", parsed=setup, raw=rss.raw_reply, reason=reason)

    def start_system(self) -> PumpOperationResult:
        rs = self.read_rss_with_retry()
        if not rs.ok:
            return self._fail(f"系统启动前 RSS 读取失败: {rs.error}")
        setup: SystemSetup = rs.parsed_reply
        effective_enable = (int(setup.enable_mask) | int(setup.copy_mask)) & 0x0F
        run_mask = (effective_enable & 0x0F) << 1
        target_sys = (0x01 | run_mask) if run_mask else 0x00
        return self.write_wse(sys_runstate=target_sys, q_runstate=0x00)

    def start_system_and_verify(self) -> PumpOperationResult:
        rs = self.read_rss_with_retry()
        if not rs.ok:
            return self._fail(f"系统启动前 RSS 读取失败: {rs.error}")
        setup: SystemSetup = rs.parsed_reply
        effective_enable = (int(setup.enable_mask) | int(setup.copy_mask)) & 0x0F
        run_mask = (effective_enable & 0x0F) << 1
        target_sys = (0x01 | run_mask) if run_mask else 0x00
        wr = self.write_wse(sys_runstate=target_sys, q_runstate=0x00)
        if not wr.ok:
            return wr

        # 启动后状态有时滞后，短轮询避免瞬时 0x00 误判。
        last_state: RunState | None = None
        for _ in range(8):
            rd = self.read_rse()
            if rd.ok and rd.parsed_reply is not None:
                got: RunState = rd.parsed_reply
                last_state = got
                got_mask = int(got.sys_runstate) & 0x1E
                if target_sys == 0x00:
                    if int(got.sys_runstate) == 0x00:
                        self.log("[WSE][OK] 停机态回读校验通过")
                        return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSE 校验通过")
                else:
                    if bool(got.system_running) and ((got_mask & run_mask) == run_mask):
                        self.log("[WSE][OK] 启动态回读校验通过")
                        return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSE 校验通过")
            time.sleep(0.12)

        if last_state is not None:
            reason = (
                f"sys_runstate expect=0x{target_sys:02X}, got=0x{int(last_state.sys_runstate) & 0xFF:02X}, "
                f"target_run_mask=0x{run_mask:02X}"
            )
            self.log(f"[WSE][FAIL] {reason}")
            return self._fail("WSE 回读校验失败", parsed=last_state, reason=reason)
        return self._fail("WSE 回读校验失败: RSE 无有效回包")

    def read_run_state(self) -> PumpOperationResult:
        return self.read_rse()

    @staticmethod
    def is_channel_running(channel: int, run_state: RunState | None) -> bool:
        if run_state is None:
            return False
        idx = int(channel) - 1
        if idx < 0 or idx >= len(run_state.channel_running):
            return False
        return bool(run_state.channel_running[idx])

    def are_required_channels_running(self, q_channels: list[int], run_state: RunState | None = None) -> tuple[bool, str]:
        state = run_state
        if state is None:
            rs = self.read_run_state()
            if not rs.ok or rs.parsed_reply is None:
                return False, f"读取运行状态失败: {rs.error or rs.reason}"
            state = rs.parsed_reply

        missing = [ch for ch in q_channels if not self.is_channel_running(ch, state)]
        if not state.system_running:
            return False, "系统运行位未置位"
        if missing:
            return False, f"通道未运行: {missing}"
        return True, "ok"

    def start_infusion_and_verify(self, q_channels: list[int]) -> PumpOperationResult:
        if not q_channels:
            return self._fail("未提供需要灌注的通道")

        mask = 0
        for ch in q_channels:
            if not (1 <= int(ch) <= 4):
                return self._fail(f"非法通道: {ch}")
            mask |= (1 << (int(ch) - 1))

        self.log("[PUMP][START] 开始灌注命令已发送")
        en = self.enable_channels_and_verify(mask)
        if not en.ok:
            reason = en.reason or en.error or "使能失败"
            self.log(f"[PUMP][START][FAIL] 使能失败: {reason}")
            return self._fail("启动灌注失败", reason=f"使能失败: {reason}")

        st = self.start_system_and_verify()
        if not st.ok:
            reason = st.reason or st.error or "启动失败"
            self.log(f"[PUMP][START][FAIL] {reason}")
            return self._fail("启动灌注失败", reason=reason)

        rs = self.read_run_state()
        if not rs.ok or rs.parsed_reply is None:
            reason = rs.error or rs.reason or "RSE回读失败"
            self.log(f"[PUMP][START][FAIL] {reason}")
            return self._fail("启动灌注失败", reason=reason)

        running_ok, running_reason = self.are_required_channels_running(q_channels, run_state=rs.parsed_reply)
        if not running_ok:
            self.log(f"[PUMP][START][FAIL] {running_reason}")
            return self._fail("启动灌注失败", parsed=rs.parsed_reply, raw=rs.raw_reply, reason=running_reason)

        self.log("[PUMP][START][OK] 回读确认灌注中")
        return self._ok(parsed=rs.parsed_reply, raw=rs.raw_reply, verified=True, reason="启动并确认灌注成功")

    def _default_channel_params_for_q(self, channel: int, q: float) -> ChannelParams:
        dispense_value = self._DEFAULT_DISPENSE_VALUE
        dispense_unit = self._DEFAULT_DISPENSE_UNIT
        infuse_time_unit = self._DEFAULT_INFUSE_TIME_UNIT
        dispense_value, infuse_time_value = self._dispense_and_infuse_for_flow(
            dispense_value=dispense_value,
            dispense_unit=dispense_unit,
            infuse_time_unit=infuse_time_unit,
            flow_ul_min=q,
        )
        return ChannelParams(
            channel=channel,
            mode=1,
            syringe_code=self._DEFAULT_SYRINGE_CODE,
            dispense_value=dispense_value,
            dispense_unit=dispense_unit,
            infuse_time_value=infuse_time_value,
            infuse_time_unit=infuse_time_unit,
            withdraw_time_value=self._DEFAULT_WITHDRAW_TIME_VALUE,
            withdraw_time_unit=self._DEFAULT_WITHDRAW_TIME_UNIT,
            repeat_count=self._DEFAULT_REPEAT_COUNT,
            interval_value=self._DEFAULT_INTERVAL_VALUE,
        )

    def _channel_params_preserving_profile(self, current: ChannelParams, q: float) -> ChannelParams:
        dispense_value = max(1, int(current.dispense_value))
        infuse_time_value = self._infuse_time_value_for_flow(
            dispense_value=dispense_value,
            dispense_unit=int(current.dispense_unit),
            infuse_time_unit=int(current.infuse_time_unit),
            flow_ul_min=q,
        )
        return ChannelParams(
            channel=int(current.channel),
            mode=int(current.mode),
            syringe_code=int(current.syringe_code),
            dispense_value=dispense_value,
            dispense_unit=int(current.dispense_unit),
            infuse_time_value=infuse_time_value,
            infuse_time_unit=int(current.infuse_time_unit),
            withdraw_time_value=int(current.withdraw_time_value),
            withdraw_time_unit=int(current.withdraw_time_unit),
            repeat_count=int(current.repeat_count),
            interval_value=int(current.interval_value),
        )

    def _channel_params_with_flow(self, channel: int, q: float) -> ChannelParams:
        rsp = self.read_rsp(channel)
        if rsp.ok and rsp.parsed_reply is not None:
            current: ChannelParams = rsp.parsed_reply
            p = self._channel_params_preserving_profile(current, q)
            self.log(
                f"[PUMP][PARAM_PROFILE][CH{channel}] preserving pump profile; only flow is updated "
                f"syringe=0x{p.syringe_code:02X}, volume_unit={p.dispense_unit}, "
                f"time_unit={p.infuse_time_unit}, repeat={p.repeat_count}, interval={p.interval_value}, "
                f"user_unit=uL/min"
            )
            return p
        return self._default_channel_params_for_q(channel, q)

    def channel_params_for_flow(self, channel: int, q: float) -> ChannelParams:
        return self._channel_params_with_flow(channel, q)

    def update_flow_while_running(self, q1: float, q2: float) -> FlowUpdateResult:
        self.log(f"[PUMP][UPDATE] 运行中参数更新: q1={q1:.6f}, q2={q2:.6f}")
        min_gap = max(1e-9, float(self.runtime_config.min_q1_q2_gap))
        if float(q1) < float(q2) + min_gap:
            reason = (
                f"拒绝泵流量更新：油相 Q1 必须至少比水相 Q2 大 {min_gap:.6f} uL/min；"
                f"当前 q1={float(q1):.6f}, q2={float(q2):.6f}"
            )
            self.log(f"[PUMP][UPDATE][REJECT] {reason}")
            return FlowUpdateResult(
                ok=False,
                q1_ok=False,
                q2_ok=False,
                still_running=False,
                reason=reason,
            )
        rs_before = self.read_run_state()
        if not rs_before.ok or rs_before.parsed_reply is None:
            reason = rs_before.error or rs_before.reason or "读取运行状态失败"
            self.log(f"[PUMP][RUNSTATE][FAIL] 更新前读取失败: {reason}")
            return FlowUpdateResult(
                ok=False,
                q1_ok=False,
                q2_ok=False,
                still_running=False,
                run_state_error=reason,
                reason=reason,
            )

        running_ok, running_reason = self.are_required_channels_running([1, 2], run_state=rs_before.parsed_reply)
        if not running_ok:
            self.log(f"[PUMP][RUNSTATE][FAIL] 更新前未运行: {running_reason}")
            return FlowUpdateResult(
                ok=False,
                q1_ok=False,
                q2_ok=False,
                still_running=False,
                run_state_error=running_reason,
                reason=running_reason,
            )

        p1 = self._channel_params_with_flow(1, q1)
        p2 = self._channel_params_with_flow(2, q2)
        encoded_q1 = self.flow_from_channel_params(p1)
        encoded_q2 = self.flow_from_channel_params(p2)
        if (
            encoded_q1 is None
            or encoded_q2 is None
            or encoded_q1 < encoded_q2 + min_gap
        ):
            reason = (
                "拒绝泵流量更新：编码后的泵参数不能保持油相 Q1 严格大于水相 Q2；"
                f"encoded_q1={encoded_q1}, encoded_q2={encoded_q2}, min_gap={min_gap:.6f}"
            )
            self.log(f"[PUMP][UPDATE][REJECT] {reason}")
            return FlowUpdateResult(
                ok=False,
                q1_ok=False,
                q2_ok=False,
                still_running=True,
                reason=reason,
            )
        self.log(
            "[PUMP][UPDATE][PARAMS] "
            f"CH1 target={q1:.6f}uL/min dispense={p1.dispense_value}/unit{p1.dispense_unit} "
            f"infuse={p1.infuse_time_value}/unit{p1.infuse_time_unit} "
            f"calc={self.flow_from_channel_params(p1) or 0.0:.6f}uL/min; "
            f"CH2 target={q2:.6f}uL/min dispense={p2.dispense_value}/unit{p2.dispense_unit} "
            f"infuse={p2.infuse_time_value}/unit{p2.infuse_time_unit} "
            f"calc={self.flow_from_channel_params(p2) or 0.0:.6f}uL/min"
        )

        wr1 = self.write_wsp_and_verify(1, p1)
        q1_ok = bool(wr1.ok)
        q1_error = None if wr1.ok else (wr1.reason or wr1.error or "Q1下发失败")
        if q1_ok:
            self.log("[PUMP][VERIFY][OK] CH1 参数回读校验成功")
        else:
            self.log(f"[PUMP][VERIFY][FAIL] CH1 参数回读校验失败: {q1_error}")

        inter_channel_delay = max(
            0.0,
            float(getattr(self.runtime_config, "inter_channel_update_delay", 0.0)),
        )
        if inter_channel_delay > 0.0:
            time.sleep(inter_channel_delay)

        q2_attempts = max(
            1,
            int(getattr(self.runtime_config, "q2_update_max_attempts", 1)),
        )
        wr2 = PumpOperationResult(ok=False, error="Q2 尚未写入")
        for attempt in range(1, q2_attempts + 1):
            wr2 = self.write_wsp_and_verify(2, p2)
            if wr2.ok:
                if attempt > 1:
                    self.log(f"[PUMP][Q2][RECOVERED] 第 {attempt} 次独立写入/回读成功")
                break
            q2_attempt_error = wr2.reason or wr2.error or "Q2下发失败"
            self.log(
                f"[PUMP][Q2][RETRY] 第 {attempt}/{q2_attempts} 次写入/回读失败: "
                f"{q2_attempt_error}"
            )
            if attempt < q2_attempts:
                time.sleep(
                    max(
                        0.0,
                        float(getattr(self.runtime_config, "q2_update_retry_interval", 0.0)),
                    )
                )
        q2_ok = bool(wr2.ok)
        q2_error = None if wr2.ok else (wr2.reason or wr2.error or "Q2下发失败")
        if q2_ok:
            self.log("[PUMP][VERIFY][OK] CH2 参数回读校验成功")
        else:
            self.log(f"[PUMP][VERIFY][FAIL] CH2 参数回读校验失败: {q2_error}")

        rs_after = self.read_run_state()
        still_running = False
        run_state_error = None
        if not rs_after.ok or rs_after.parsed_reply is None:
            run_state_error = rs_after.error or rs_after.reason or "更新后读取运行状态失败"
            self.log(f"[PUMP][RUNSTATE][FAIL] {run_state_error}")
        else:
            still_running, running_reason = self.are_required_channels_running([1, 2], run_state=rs_after.parsed_reply)
            if still_running:
                self.log("[PUMP][RUNSTATE][OK] 参数更新后仍在灌注")
            else:
                run_state_error = running_reason
                self.log(f"[PUMP][RUNSTATE][FAIL] 参数更新后灌注状态异常: {run_state_error}")

        ok = q1_ok and q2_ok and still_running
        reason_parts: list[str] = []
        if not q1_ok:
            reason_parts.append(f"q1失败:{q1_error}")
        if not q2_ok:
            reason_parts.append(f"q2失败:{q2_error}")
        if not still_running:
            reason_parts.append(f"运行状态异常:{run_state_error}")
        reason = "；".join(reason_parts) if reason_parts else "ok"

        return FlowUpdateResult(
            ok=ok,
            q1_ok=q1_ok,
            q2_ok=q2_ok,
            still_running=still_running,
            q1_error=q1_error,
            q2_error=q2_error,
            run_state_error=run_state_error,
            verified_q1=wr1.parsed_reply if wr1.ok else None,
            verified_q2=wr2.parsed_reply if wr2.ok else None,
            reason=reason,
        )

    def get_current_q_state(self) -> tuple[float, float]:
        def _q_from_rsp(channel: int) -> float:
            rsp = self.read_rsp(channel)
            if not rsp.ok or rsp.parsed_reply is None:
                raise RuntimeError(f"读取 CH{channel} 参数失败: {rsp.error or rsp.reason}")
            p: ChannelParams = rsp.parsed_reply
            q = self.flow_from_channel_params(p)
            if q is None:
                raise RuntimeError(f"CH{channel} infuse_time_value 非法: {p.infuse_time_value}")
            return float(q)

        q1 = _q_from_rsp(1)
        q2 = _q_from_rsp(2)
        return q1, q2

    def start_single_channel_safely(self, channel: int) -> PumpOperationResult:
        if not (1 <= int(channel) <= 4):
            return self._fail(f"无效通道: {channel}")

        rse_now = self.read_rse()
        if not rse_now.ok:
            return self._fail(f"单通道启动前 RSE 读取失败: {rse_now.error}")
        rss_now = self.read_rss()
        if not rss_now.ok:
            return self._fail(f"单通道启动前 RSS 读取失败: {rss_now.error}")

        run_state: RunState = rse_now.parsed_reply
        setup: SystemSetup = rss_now.parsed_reply
        current_run_mask = run_state.sys_runstate & 0x1E
        target_bit = (1 << int(channel)) & 0x1E
        expected_after_mask = (current_run_mask | target_bit) & 0x1E
        desired_enable_mask = (expected_after_mask >> 1) & 0x0F
        current_enable_mask = setup.enable_mask & 0x0F

        if desired_enable_mask != current_enable_mask:
            adjust_setup = SystemSetup(
                enable_mask=desired_enable_mask,
                copy_mask=desired_enable_mask if desired_enable_mask else 0,
                delay_values=list(setup.delay_values),
                delay_units=list(setup.delay_units),
            )
            en = self.write_wss_and_verify(adjust_setup)
            if not en.ok:
                return self._fail(f"[START][FAIL][CH{channel}] 启动前使能收敛失败: {en.reason or en.error}")

        target_sys = 0x01 | expected_after_mask
        wr = self.write_wse_and_verify(sys_runstate=target_sys, q_runstate=run_state.q_runstate)
        if not wr.ok:
            return wr

        final_rse = self.read_rse()
        if not final_rse.ok:
            return self._fail(f"[START][FAIL][CH{channel}] 启动后二次 RSE 确认失败: {final_rse.error}")
        final_state: RunState = final_rse.parsed_reply
        final_run_mask = final_state.sys_runstate & 0x1E

        if final_run_mask != expected_after_mask:
            extra = final_run_mask & (~expected_after_mask & 0x1E)
            extra_channels = [i + 1 for i in range(4) if extra & (1 << (i + 1))]
            reason = (
                f"检测到额外通道被启动: expected_mask=0x{expected_after_mask:02X}, "
                f"actual_mask=0x{final_run_mask:02X}, extra_channels={extra_channels}"
            )
            return self._fail("[START] 检测到额外通道启动", parsed=final_state, raw=final_rse.raw_reply, reason=reason)

        return self._ok(parsed=final_state, raw=final_rse.raw_reply, verified=True, reason="单通道启动校验通过")

    def stop_single_channel_safely(self, channel: int) -> PumpOperationResult:
        if not (1 <= int(channel) <= 4):
            return self._fail(f"无效通道: {channel}")

        rse_now = self.read_rse()
        if not rse_now.ok:
            return self._fail(f"单通道停止前 RSE 读取失败: {rse_now.error}")
        run_state: RunState = rse_now.parsed_reply

        current_run_mask = run_state.sys_runstate & 0x1E
        target_bit = (1 << int(channel)) & 0x1E
        expected_after_mask = current_run_mask & (~target_bit & 0x1E)
        target_sys = (0x01 | expected_after_mask) if expected_after_mask else 0x00

        wr = self.write_wse_and_verify(sys_runstate=target_sys, q_runstate=run_state.q_runstate)
        if not wr.ok:
            return wr

        final_rse = self.read_rse()
        if not final_rse.ok:
            return self._fail(f"[STOP][FAIL][CH{channel}] 停止后二次 RSE 确认失败: {final_rse.error}")
        final_state: RunState = final_rse.parsed_reply
        final_mask = final_state.sys_runstate & 0x1E

        if (final_mask & target_bit) != 0:
            reason = f"目标通道仍在运行: ch={channel}, final_mask=0x{final_mask:02X}"
            return self._fail("[STOP] 目标通道未停止", parsed=final_state, raw=final_rse.raw_reply, reason=reason)

        return self._ok(parsed=final_state, raw=final_rse.raw_reply, verified=True, reason="单通道停止校验通过")
