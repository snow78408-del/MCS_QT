from __future__ import annotations

import math
import time
from typing import Callable

from . import protocol
from .client import CommandMismatchError, FrameParseError, NoReplyError, PumpClient
from .config import PumpHardwareConfig, SerialConfig
from ..device_lock import DeviceLockError
from .invariants import effective_q1_q2_gap, q1_is_strictly_more_than_twice_q2
from .models import (
    ChannelParams,
    FlowUpdateResult,
    PumpConnectionState,
    PumpOperationResult,
    RunState,
    SystemSetup,
)


class ChannelFlowParseError(ValueError):
    """泵通道回读参数存在但无法解析（协议错位、字段损坏或单位码未知）。

    与 ``None`` 严格区分：``None`` 表示**没有回读数据**，本异常表示
    **回读数据存在但不可用**。回读校验路径必须用
    ``flow_from_channel_params_strict``；若用宽松版本，调用方会把损坏回读
    当成「无读数」而回退到指令值，使 ``_flow_matches`` 退化成自比恒真。
    """


class FlowReadbackUnavailableError(RuntimeError):
    """没有回读数据，因此无法认定写入已通过验证。

    指令值只能作为**指令**单独记录展示，不能拿来充当实测值——否则
    ``_flow_matches(name, commanded, actual)`` 是拿指令值和它自己比，必然通过。
    """


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
    # TS 规约中除分配次数外的两字节参数范围是 1..9999。
    _MAX_PARAM_VALUE = 9999

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

    @staticmethod
    def _unit_code(value: object, name: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ChannelFlowParseError(f"{name} 不是整数单位码：{value!r}") from exc

    @staticmethod
    def _finite_number(value: object, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ChannelFlowParseError(f"{name} 不是数值：{value!r}") from exc
        if not math.isfinite(number):
            raise ChannelFlowParseError(f"{name} 不是有限值：{value!r}")
        return number

    @classmethod
    def _parse_channel_flow(cls, params: ChannelParams | None) -> float | None:
        """把回读参数换算成 uL/min。

        ``None`` 只表示**没有回读数据**；参数存在但不可用时抛 ``ChannelFlowParseError``。

        单位码必须命中已知单位表：不允许走 ``.get(code, 1.0)`` 的默认倍率，否则未知单位
        会被静默当成 1.0，流量可能差几个数量级。
        """
        if params is None:
            return None
        dispense_unit = cls._unit_code(params.dispense_unit, "dispense_unit")
        infuse_unit = cls._unit_code(params.infuse_time_unit, "infuse_time_unit")
        if dispense_unit not in cls._VOLUME_UNIT_TO_UL:
            raise ChannelFlowParseError(f"未知容积单位码：{dispense_unit!r}")
        if infuse_unit not in cls._TIME_UNIT_TO_MIN:
            raise ChannelFlowParseError(f"未知时间单位码：{infuse_unit!r}")
        dispense_value = cls._finite_number(params.dispense_value, "dispense_value")
        infuse_value = cls._finite_number(params.infuse_time_value, "infuse_time_value")
        if dispense_value < 0.0:
            raise ChannelFlowParseError(f"回读容积为负：{dispense_value!r}")
        if infuse_value <= 0.0:
            raise ChannelFlowParseError(f"回读灌注时间为非正：{infuse_value!r}")
        volume_ul = dispense_value * cls._VOLUME_UNIT_TO_UL[dispense_unit]
        time_min = infuse_value * cls._TIME_UNIT_TO_MIN[infuse_unit]
        if not math.isfinite(volume_ul) or not math.isfinite(time_min):
            raise ChannelFlowParseError("回读换算中间量非有限")
        if time_min <= 0.0:
            raise ChannelFlowParseError(f"回读换算后的时间为非正：{time_min!r}")
        flow = volume_ul / time_min
        if not math.isfinite(flow) or flow < 0.0:
            raise ChannelFlowParseError(f"回读换算结果非法：{flow!r}")
        return flow

    @classmethod
    def flow_from_channel_params(cls, params: ChannelParams | None) -> float | None:
        """宽松解析，仅供日志与展示：无回读和损坏回读都返回 None。

        校验回读必须用 ``flow_from_channel_params_strict``。
        """
        try:
            return cls._parse_channel_flow(params)
        except ChannelFlowParseError:
            return None

    @classmethod
    def flow_from_channel_params_strict(cls, params: ChannelParams | None) -> float | None:
        """严格解析，供回读校验：``None`` 只表示没有回读数据。

        参数存在但损坏时抛 ``ChannelFlowParseError``，避免调用方回退到指令值。
        """
        return cls._parse_channel_flow(params)

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
        preferred = str(self.serial_config.parity or "E").strip().upper()
        candidates = [preferred]
        if self.serial_config.allow_parity_fallback_n and preferred != "N":
            candidates.append("N")

        best_state = PumpConnectionState(serial_connected=False, comm_established=False, fully_ready=False)
        for parity in candidates:
            self.client.disconnect(preserve_borrowed_lock=True)
            self.serial_config.parity = parity
            state = PumpConnectionState(serial_connected=False, comm_established=False, fully_ready=False)
            try:
                self.client.connect()
                state.serial_connected = self.client.is_connected()
            except DeviceLockError as e:
                # 设备被其他进程占用：换奇偶校验重试没有意义，立刻失败并保留真正的原因——
                # 否则会被下面的分支记成「串口打开失败」，把设备互斥冲突掩盖掉。
                state.failed["device_lock"] = str(e)
                self.log(f"[CONNECT][BLOCKED] {e}")
                self.connection_state = state
                return state
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

    def borrow_device_lock(self, lock) -> None:
        """Bind a pump lock owned by an outer multi-device session."""
        self.client.borrow_device_lock(lock)

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
        mismatch = []
        if got.enable_mask != setup.enable_mask:
            mismatch.append(
                f"enable_mask expect=0x{setup.enable_mask:02X}, got=0x{got.enable_mask:02X}"
            )
        if got.copy_mask != setup.copy_mask:
            mismatch.append(
                f"copy_mask expect=0x{setup.copy_mask:02X}, got=0x{got.copy_mask:02X}"
            )
        if got.delay_values != setup.delay_values:
            mismatch.append(f"delay_values expect={setup.delay_values}, got={got.delay_values}")
        if got.delay_units != setup.delay_units:
            mismatch.append(f"delay_units expect={setup.delay_units}, got={got.delay_units}")

        if not mismatch:
            self.log("[WSS][OK] 写后读回校验通过")
            return self._ok(parsed=got, raw=rd.raw_reply, verified=True, reason="WSS 校验通过")

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
        req = SystemSetup(
            enable_mask=enable,
            # copy_mask 是“运行时其它通道拷贝哪一通道的参数”，
            # 不是通道使能位。Q1/Q2 独立控制时必须关闭拷贝。
            copy_mask=0,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        return self.write_wss(req)

    def enable_channels_and_verify(self, mask: int) -> PumpOperationResult:
        rss = self.read_rss_with_retry()
        if not rss.ok:
            return self._fail(f"enable_channels_and_verify 前读取 RSS 失败: {rss.error}")
        setup: SystemSetup = rss.parsed_reply
        enable = int(mask) & 0x0F
        req = SystemSetup(
            enable_mask=enable,
            copy_mask=0,
            delay_values=list(setup.delay_values),
            delay_units=list(setup.delay_units),
        )
        return self.write_wss_and_verify(req)

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
            enable_mask=target,
            copy_mask=0,
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
            if (int(got.enable_mask) & 0x0F) == target and int(got.copy_mask) == 0:
                self.log(
                    f"[PUMP][PARAM_WRITE][READY] mask=0x{target:02X} "
                    f"enable=0x{int(got.enable_mask) & 0x0F:02X} copy=0x{int(got.copy_mask) & 0x0F:02X}"
                )
                return self._ok(
                    parsed=got,
                    raw=rd.raw_reply,
                    verified=True,
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
        run_mask = (int(setup.enable_mask) & 0x0F) << 1
        target_sys = (0x01 | run_mask) if run_mask else 0x00
        return self.write_wse(sys_runstate=target_sys, q_runstate=0x00)

    def start_system_and_verify(self) -> PumpOperationResult:
        rs = self.read_rss_with_retry()
        if not rs.ok:
            return self._fail(f"系统启动前 RSS 读取失败: {rs.error}")
        setup: SystemSetup = rs.parsed_reply
        run_mask = (int(setup.enable_mask) & 0x0F) << 1
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
                    if bool(got.system_running) and got_mask == run_mask:
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

        def _abort_started_pump(reason: str) -> PumpOperationResult:
            stopped = self.stop_system_and_verify()
            suffix = "safe stop verified" if stopped.ok else f"safe stop failed: {stopped.reason or stopped.error}"
            self.log(f"[PUMP][START][ABORT] {reason}; {suffix}")
            return self._fail("启动灌注失败", reason=f"{reason}; {suffix}")

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
            return _abort_started_pump(reason)

        rs = self.read_run_state()
        if not rs.ok or rs.parsed_reply is None:
            reason = rs.error or rs.reason or "RSE回读失败"
            self.log(f"[PUMP][START][FAIL] {reason}")
            return _abort_started_pump(reason)

        running_ok, running_reason = self.are_required_channels_running(q_channels, run_state=rs.parsed_reply)
        if not running_ok:
            self.log(f"[PUMP][START][FAIL] {running_reason}")
            return _abort_started_pump(running_reason)

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
        transaction_started = time.monotonic()
        self.log(f"[PUMP][UPDATE] 停泵后更新并重启: q1={q1:.6f}, q2={q2:.6f}")
        if not math.isfinite(float(q1)) or not math.isfinite(float(q2)) or float(q1) <= 0.0 or float(q2) <= 0.0:
            reason = "拒绝泵流量更新：Q1 和 Q2 必须为有限正数"
            self.log(f"[PUMP][UPDATE][REJECT] {reason}")
            return FlowUpdateResult(
                ok=False,
                q1_ok=False,
                q2_ok=False,
                still_running=False,
                reason=reason,
            )
        min_gap = effective_q1_q2_gap(self.runtime_config.min_q1_q2_gap)
        if not q1_is_strictly_more_than_twice_q2(q1, q2):
            reason = (
                "拒绝泵流量更新：油相 Q1 必须严格大于水相 Q2 的 2 倍；"
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
        before_p1 = self.last_channel_params.get(1)
        p2 = self._channel_params_with_flow(2, q2)
        before_p2 = self.last_channel_params.get(2)
        encoded_q1 = self.flow_from_channel_params(p1)
        encoded_q2 = self.flow_from_channel_params(p2)
        if (
            encoded_q1 is None
            or encoded_q2 is None
            or not q1_is_strictly_more_than_twice_q2(encoded_q1, encoded_q2)
        ):
            reason = (
                "拒绝泵流量更新：编码后的泵参数不能保持油相 Q1 严格大于水相 Q2 的 2 倍；"
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

        # This pump accepts WSP readback while running, but the new infusion
        # rate is latched only by a subsequent start. Never treat a successful
        # parameter echo or a still-running RSE as proof of the new flow.
        try:
            stopped = self.stop_system_and_verify()
        except Exception as exc:
            stopped = PumpOperationResult(ok=False, error=f"停泵调用异常: {exc!r}")
        if not stopped.ok:
            reason = f"更新前停泵未核验，拒绝写参数: {stopped.reason or stopped.error}"
            self.log(f"[PUMP][UPDATE][FAIL] {reason}")
            return FlowUpdateResult(
                ok=False, q1_ok=False, q2_ok=False, still_running=False,
                reason=reason, safe_stop_verified=False,
                command_started_monotonic=transaction_started,
                readback_completed_monotonic=time.monotonic(),
            )
        self.log("[PUMP][UPDATE][STOPPED] 停泵回读通过，开始写参数")

        # A Q2-only step should not spend several serial round trips rewriting
        # unchanged CH1. Re-read it after the verified stop; skip WSP only when
        # the stopped pump confirms the exact requested parameter profile.
        q1_written = False
        unchanged_q1 = None
        if before_p1 is not None and before_p1 == p1:
            try:
                unchanged_q1 = self.read_rsp(1)
            except Exception:
                unchanged_q1 = None
        if (unchanged_q1 is not None and unchanged_q1.ok
                and unchanged_q1.parsed_reply == p1):
            wr1 = unchanged_q1
            self.log("[PUMP][UPDATE][CH1] 停泵后回读确认参数未变，省略重复写入")
        else:
            q1_written = True
            try:
                wr1 = self.write_wsp_and_verify(1, p1)
            except Exception as exc:
                wr1 = PumpOperationResult(ok=False, error=f"CH1 写入异常: {exc!r}")
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
        if q1_written and inter_channel_delay > 0.0:
            time.sleep(inter_channel_delay)

        q2_attempts = max(
            1,
            int(getattr(self.runtime_config, "q2_update_max_attempts", 1)),
        )
        wr2 = PumpOperationResult(ok=False, error="Q2 尚未写入")
        q2_written = False
        if q1_ok:
            unchanged_q2 = None
            if before_p2 is not None and before_p2 == p2:
                try:
                    unchanged_q2 = self.read_rsp(2)
                except Exception:
                    unchanged_q2 = None
            if (unchanged_q2 is not None and unchanged_q2.ok
                    and unchanged_q2.parsed_reply == p2):
                wr2 = unchanged_q2
                self.log("[PUMP][UPDATE][CH2] 停泵后回读确认参数未变，省略重复写入")
            else:
                q2_written = True
                for attempt in range(1, q2_attempts + 1):
                    try:
                        wr2 = self.write_wsp_and_verify(2, p2)
                    except Exception as exc:
                        wr2 = PumpOperationResult(ok=False, error=f"CH2 写入异常: {exc!r}")
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
        else:
            wr2 = PumpOperationResult(ok=False, error="Q1 未验证，禁止继续写入 Q2")
        q2_ok = bool(wr2.ok)
        q2_error = None if wr2.ok else (wr2.reason or wr2.error or "Q2下发失败")
        if q2_ok:
            self.log("[PUMP][VERIFY][OK] CH2 参数回读校验成功")
        else:
            self.log(f"[PUMP][VERIFY][FAIL] CH2 参数回读校验失败: {q2_error}")

        still_running = False
        run_state_error = None
        if q1_ok and q2_ok:
            try:
                restarted = self.start_infusion_and_verify([1, 2])
            except Exception as exc:
                restarted = PumpOperationResult(ok=False, error=f"重新启动异常: {exc!r}")
            still_running = bool(restarted.ok)
            if still_running:
                self.log("[PUMP][UPDATE][RESTARTED] 新参数写入后重新启动，运行态回读通过")
            else:
                run_state_error = restarted.reason or restarted.error or "重新启动未核验"
                self.log(f"[PUMP][UPDATE][RESTART_FAIL] {run_state_error}")
        else:
            run_state_error = "参数写入未全部核验，禁止重新启动"

        ok = q1_ok and q2_ok and still_running
        reason_parts: list[str] = []
        if not q1_ok:
            reason_parts.append(f"q1失败:{q1_error}")
        if not q2_ok:
            reason_parts.append(f"q2失败:{q2_error}")
        if not still_running:
            reason_parts.append(f"运行状态异常:{run_state_error}")
        reason = "；".join(reason_parts) if reason_parts else "ok"

        if not ok:
            # The TS protocol cannot atomically commit two channels.  Treat a
            # partial result as a failed transaction: stop first, restore every
            # channel that may have changed, and deliberately remain stopped.
            try:
                stop_result = self.stop_system_and_verify()
            except Exception as exc:
                stop_result = PumpOperationResult(ok=False, error=f"失败后停泵异常: {exc!r}")
            safe_stop_verified = bool(stop_result.ok)
            rollback_errors: list[str] = []
            rollback_attempted = False
            if safe_stop_verified:
                for channel, changed, previous in (
                    (1, q1_written and q1_ok, before_p1),
                    (2, q2_written and q2_ok, before_p2),
                ):
                    if not changed:
                        continue
                    rollback_attempted = True
                    if previous is None:
                        rollback_errors.append(f"CH{channel} 缺少事务前快照")
                        continue
                    try:
                        restored = self.write_wsp_and_verify(channel, previous)
                    except Exception as exc:
                        restored = PumpOperationResult(ok=False, error=f"CH{channel} 回滚异常: {exc!r}")
                    if not restored.ok:
                        rollback_errors.append(
                            f"CH{channel} 回滚失败:{restored.reason or restored.error}"
                        )
                try:
                    final_stop = self.stop_system_and_verify()
                except Exception as exc:
                    final_stop = PumpOperationResult(ok=False, error=f"回滚后停泵异常: {exc!r}")
                safe_stop_verified = bool(final_stop.ok)
                if not final_stop.ok:
                    rollback_errors.append(
                        f"回滚后停泵验证失败:{final_stop.reason or final_stop.error}"
                    )
            else:
                rollback_errors.append(
                    f"安全停泵失败:{stop_result.reason or stop_result.error}"
                )
            rolled_back = rollback_attempted and not rollback_errors
            reason = f"{reason}；双通道事务失败，系统保持停机"
            if rollback_errors:
                reason = f"{reason}；{'；'.join(rollback_errors)}"
            self.log(
                f"[PUMP][UPDATE][SAFE_DEGRADE] stopped={safe_stop_verified} "
                f"rolled_back={rolled_back} reason={reason}"
            )
            return FlowUpdateResult(
                ok=False,
                q1_ok=q1_ok,
                q2_ok=q2_ok,
                still_running=False,
                q1_error=q1_error,
                q2_error=q2_error,
                run_state_error=run_state_error,
                verified_q1=wr1.parsed_reply if wr1.ok else None,
                verified_q2=wr2.parsed_reply if wr2.ok else None,
                reason=reason,
                rolled_back=rolled_back,
                rollback_error="；".join(rollback_errors) or None,
                safe_stop_verified=safe_stop_verified,
                stop_verified_before_write=True,
                restart_verified=False,
                command_started_monotonic=transaction_started,
                readback_completed_monotonic=time.monotonic(),
            )

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
            safe_stop_verified=False,
            stop_verified_before_write=True,
            restart_verified=True,
            command_started_monotonic=transaction_started,
            readback_completed_monotonic=time.monotonic(),
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
                copy_mask=0,
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
