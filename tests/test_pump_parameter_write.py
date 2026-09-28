import pytest

from backend.pump_hardware.client import CommandMismatchError, PumpClient
from backend.pump_hardware.config import PumpHardwareConfig, SerialConfig
from backend.pump_hardware.models import ChannelParams, PumpOperationResult, RunState, SystemSetup
from backend.pump_hardware.service import PumpHardwareService
from backend.pump_hardware import protocol


def test_client_rejects_reply_from_wrong_device_address():
    class FakeSerial:
        is_open = True

        def __init__(self, frame):
            self.frame = frame

        def reset_input_buffer(self):
            pass

        def write(self, _frame):
            pass

        def flush(self):
            pass

        def read(self, _size):
            if self.frame:
                value, self.frame = self.frame[:1], self.frame[1:]
                return value
            return b""

    client = PumpClient(SerialConfig(address=1), PumpHardwareConfig(reply_timeout=0.01, retry_count=1))
    client._ser = FakeSerial(protocol.build_frame(addr=2, pdu=protocol.pdu_rss()))
    with pytest.raises(CommandMismatchError, match="地址不匹配"):
        client.send_pdu(protocol.pdu_rss(), expect_cmd="RSS", retries=1, timeout=0.01, addr=1)


def test_prepare_parameter_write_does_not_repeat_rss_after_verified_wss(monkeypatch):
    service = PumpHardwareService()
    setup = SystemSetup(
        enable_mask=0x03,
        copy_mask=0x00,
        delay_values=[0, 0, 0, 0],
        delay_units=[0, 0, 0, 0],
    )
    read_calls = 0

    monkeypatch.setattr(
        service,
        "stop_system_and_verify",
        lambda: PumpOperationResult(ok=True, verified=True),
    )

    def read_rss():
        nonlocal read_calls
        read_calls += 1
        return PumpOperationResult(ok=True, parsed_reply=setup, verified=True)

    monkeypatch.setattr(service, "read_rss", read_rss)
    monkeypatch.setattr(
        service,
        "write_wss_and_verify",
        lambda _requested: PumpOperationResult(
            ok=True,
            parsed_reply=setup,
            verified=True,
            reason="WSS 校验通过",
        ),
    )

    result = service.prepare_parameter_write(0x03)

    assert result.ok
    assert result.verified
    assert read_calls == 1


def test_prepare_parameter_write_can_recover_with_one_rss_read(monkeypatch):
    service = PumpHardwareService()
    before = SystemSetup(
        enable_mask=0x00,
        copy_mask=0x00,
        delay_values=[0, 0, 0, 0],
        delay_units=[0, 0, 0, 0],
    )
    accepted = SystemSetup(
        enable_mask=0x03,
        copy_mask=0x00,
        delay_values=[0, 0, 0, 0],
        delay_units=[0, 0, 0, 0],
    )
    replies = iter(
        [
            PumpOperationResult(ok=True, parsed_reply=before),
            PumpOperationResult(ok=True, parsed_reply=accepted),
        ]
    )

    monkeypatch.setattr(
        service,
        "stop_system_and_verify",
        lambda: PumpOperationResult(ok=True, verified=True),
    )
    monkeypatch.setattr(service, "read_rss", lambda: next(replies))
    monkeypatch.setattr(
        service,
        "write_wss_and_verify",
        lambda _requested: PumpOperationResult(ok=False, error="temporary readback timeout"),
    )

    result = service.prepare_parameter_write(0x03)

    assert result.ok
    assert result.parsed_reply is accepted


def test_enabling_q1_q2_keeps_protocol_copy_mask_disabled(monkeypatch):
    service = PumpHardwareService()
    setup = SystemSetup(
        enable_mask=0x01,
        copy_mask=0x01,
        delay_values=[0, 0, 0, 0],
        delay_units=[0, 0, 0, 0],
    )
    requested = []
    monkeypatch.setattr(
        service,
        "read_rss_with_retry",
        lambda: PumpOperationResult(ok=True, parsed_reply=setup, verified=True),
    )

    def write_and_verify(value):
        requested.append(value)
        return PumpOperationResult(ok=True, parsed_reply=value, verified=True)

    monkeypatch.setattr(service, "write_wss_and_verify", write_and_verify)

    result = service.enable_channels_and_verify(0x03)

    assert result.ok
    assert requested[0].enable_mask == 0x03
    assert requested[0].copy_mask == 0x00


def test_wsp_verification_does_not_use_wss_copy_mask_as_commit(monkeypatch):
    service = PumpHardwareService()
    params = service._default_channel_params_for_q(1, 50.0)
    monkeypatch.setattr(
        service,
        "write_wsp",
        lambda value: PumpOperationResult(ok=True, parsed_reply=value),
    )
    monkeypatch.setattr(
        service,
        "read_rsp",
        lambda channel: PumpOperationResult(
            ok=True,
            parsed_reply=params,
            verified=True,
        ),
    )
    monkeypatch.setattr(
        service,
        "read_rss",
        lambda: pytest.fail("WSP/RSP verification must not trigger a WSS commit"),
    )

    result = service.write_wsp_and_verify(1, params)

    assert result.ok
    assert result.verified


def test_rss_retry_recovers_after_transient_timeouts(monkeypatch):
    service = PumpHardwareService()
    setup = SystemSetup(
        enable_mask=0x03,
        copy_mask=0x02,
        delay_values=[0, 0, 0, 0],
        delay_units=[0, 0, 0, 0],
    )
    replies = iter(
        [
            PumpOperationResult(ok=False, error="等待帧头超时"),
            PumpOperationResult(ok=False, error="等待帧头超时"),
            PumpOperationResult(ok=True, parsed_reply=setup, verified=True),
        ]
    )
    monkeypatch.setattr(service, "read_rss", lambda: next(replies))
    monkeypatch.setattr("backend.pump_hardware.service.time.sleep", lambda _seconds: None)

    result = service.read_rss_with_retry(attempts=3)

    assert result.ok
    assert result.parsed_reply is setup


@pytest.mark.parametrize("q1, q2", [(0.0, 20.0), (-1.0, 20.0), (float("nan"), 20.0), (50.0, float("inf"))])
def test_running_update_rejects_nonfinite_or_nonpositive_flows_before_serial_io(monkeypatch, q1, q2):
    service = PumpHardwareService()
    monkeypatch.setattr(service, "read_run_state", lambda: pytest.fail("invalid flow reached serial I/O"))
    result = service.update_flow_while_running(q1, q2)
    assert not result.ok
    assert not result.still_running
    assert "有限正数" in result.reason


def test_running_update_rejects_q1_not_above_q2_before_serial_io(monkeypatch):
    service = PumpHardwareService()
    serial_io_attempted = False

    def read_run_state():
        nonlocal serial_io_attempted
        serial_io_attempted = True
        raise AssertionError("invalid phase flows must be rejected before serial I/O")

    monkeypatch.setattr(service, "read_run_state", read_run_state)
    result = service.update_flow_while_running(20.0, 20.0)

    assert not result.ok
    assert not serial_io_attempted
    assert "Q1" in result.reason


def test_running_update_retries_only_q2_after_transient_failure(monkeypatch):
    service = PumpHardwareService()
    service.runtime_config.inter_channel_update_delay = 0.0
    service.runtime_config.q2_update_retry_interval = 0.0
    service.runtime_config.q2_update_max_attempts = 2
    run_state = RunState(
        sys_runstate=1,
        q_runstate=0x03,
        system_running=True,
        channel_running=[True, True, False, False],
    )
    monkeypatch.setattr(
        service,
        "read_run_state",
        lambda: PumpOperationResult(ok=True, parsed_reply=run_state),
    )
    monkeypatch.setattr(
        service,
        "_channel_params_with_flow",
        lambda channel, q: service._default_channel_params_for_q(channel, q),
    )
    calls = []

    monkeypatch.setattr(service, "stop_system_and_verify",
                        lambda: (calls.append("stop") or PumpOperationResult(ok=True, verified=True)))
    monkeypatch.setattr(service, "start_infusion_and_verify",
                        lambda channels: (calls.append("start") or PumpOperationResult(ok=True, verified=True)))

    def write_and_verify(channel, params):
        calls.append(channel)
        if channel == 2 and calls.count(2) == 1:
            return PumpOperationResult(ok=False, error="temporary Q2 timeout")
        return PumpOperationResult(ok=True, parsed_reply=params, verified=True)

    monkeypatch.setattr(service, "write_wsp_and_verify", write_and_verify)

    result = service.update_flow_while_running(50.0, 20.0)

    assert result.ok
    assert result.q1_ok and result.q2_ok
    assert result.stop_verified_before_write and result.restart_verified
    assert calls == ["stop", 1, 2, 2, "start"]


def test_running_update_never_writes_when_stop_is_unverified(monkeypatch):
    service = PumpHardwareService()
    running = RunState(1, 0x03, True, [True, True, False, False])
    monkeypatch.setattr(service, "read_run_state",
                        lambda: PumpOperationResult(ok=True, parsed_reply=running))
    monkeypatch.setattr(service, "_channel_params_with_flow",
                        lambda ch, q: service._default_channel_params_for_q(ch, q))
    monkeypatch.setattr(service, "stop_system_and_verify",
                        lambda: PumpOperationResult(ok=False, error="stop timeout"))
    monkeypatch.setattr(service, "write_wsp_and_verify",
                        lambda *_: pytest.fail("must not write while stop is unverified"))
    monkeypatch.setattr(service, "start_infusion_and_verify",
                        lambda *_: pytest.fail("must not restart after failed stop"))

    result = service.update_flow_while_running(100.0, 20.0)

    assert not result.ok
    assert not result.safe_stop_verified
    assert not result.stop_verified_before_write
    assert "停泵未核验" in result.reason


def test_q2_only_step_skips_rewriting_verified_unchanged_q1(monkeypatch):
    service = PumpHardwareService()
    service.runtime_config.inter_channel_update_delay = 0.0
    p1 = service._default_channel_params_for_q(1, 100.0)
    p2 = service._default_channel_params_for_q(2, 30.0)
    service.last_channel_params[1] = p1
    service.last_channel_params[2] = service._default_channel_params_for_q(2, 10.0)
    running = RunState(1, 0x03, True, [True, True, False, False])
    monkeypatch.setattr(service, "read_run_state",
                        lambda: PumpOperationResult(ok=True, parsed_reply=running))
    monkeypatch.setattr(service, "_channel_params_with_flow",
                        lambda channel, _q: p1 if channel == 1 else p2)
    events = []
    monkeypatch.setattr(service, "stop_system_and_verify",
                        lambda: (events.append("stop") or PumpOperationResult(ok=True, verified=True)))
    monkeypatch.setattr(service, "read_rsp", lambda ch: (
        events.append("read_q1") or PumpOperationResult(ok=True, parsed_reply=p1,
                                                          verified=True)))
    monkeypatch.setattr(service, "write_wsp_and_verify", lambda ch, p: (
        events.append(f"write_q{ch}") or PumpOperationResult(ok=True, parsed_reply=p,
                                                                verified=True)))
    monkeypatch.setattr(service, "start_infusion_and_verify", lambda channels: (
        events.append("start") or PumpOperationResult(ok=True, verified=True)))

    result = service.update_flow_while_running(100.0, 30.0)

    assert result.ok and result.stop_verified_before_write and result.restart_verified
    assert events == ["stop", "read_q1", "write_q2", "start"]


def test_q1_only_step_skips_rewriting_verified_unchanged_q2(monkeypatch):
    service = PumpHardwareService()
    service.runtime_config.inter_channel_update_delay = 0.0
    p1 = service._default_channel_params_for_q(1, 70.0)
    p2 = service._default_channel_params_for_q(2, 20.0)
    service.last_channel_params[1] = service._default_channel_params_for_q(1, 60.0)
    service.last_channel_params[2] = p2
    running = RunState(1, 0x03, True, [True, True, False, False])
    monkeypatch.setattr(service, "read_run_state",
                        lambda: PumpOperationResult(ok=True, parsed_reply=running))
    monkeypatch.setattr(service, "_channel_params_with_flow",
                        lambda channel, _q: p1 if channel == 1 else p2)
    events = []
    monkeypatch.setattr(service, "stop_system_and_verify",
                        lambda: (events.append("stop") or PumpOperationResult(ok=True, verified=True)))
    monkeypatch.setattr(service, "read_rsp", lambda ch: (
        events.append("read_q2") or PumpOperationResult(ok=True, parsed_reply=p2,
                                                          verified=True)))
    monkeypatch.setattr(service, "write_wsp_and_verify", lambda ch, p: (
        events.append(f"write_q{ch}") or PumpOperationResult(ok=True, parsed_reply=p,
                                                                verified=True)))
    monkeypatch.setattr(service, "start_infusion_and_verify", lambda channels: (
        events.append("start") or PumpOperationResult(ok=True, verified=True)))

    result = service.update_flow_while_running(70.0, 20.0)

    assert result.ok and result.stop_verified_before_write and result.restart_verified
    assert events == ["stop", "write_q1", "read_q2", "start"]


def test_running_update_restart_failure_rolls_back_and_stays_stopped(monkeypatch):
    service = PumpHardwareService()
    service.runtime_config.inter_channel_update_delay = 0.0
    running = RunState(1, 0x03, True, [True, True, False, False])
    monkeypatch.setattr(service, "read_run_state",
                        lambda: PumpOperationResult(ok=True, parsed_reply=running))
    previous = {ch: service._default_channel_params_for_q(ch, q)
                for ch, q in ((1, 50.0), (2, 20.0))}

    def build(ch, q):
        service.last_channel_params[ch] = previous[ch]
        return service._default_channel_params_for_q(ch, q)

    monkeypatch.setattr(service, "_channel_params_with_flow", build)
    events = []
    monkeypatch.setattr(service, "stop_system_and_verify",
                        lambda: (events.append("stop") or PumpOperationResult(ok=True, verified=True)))
    monkeypatch.setattr(service, "write_wsp_and_verify",
                        lambda ch, p: (events.append((ch, p)) or PumpOperationResult(
                            ok=True, parsed_reply=p, verified=True)))
    monkeypatch.setattr(service, "start_infusion_and_verify",
                        lambda channels: (events.append("start") or PumpOperationResult(
                            ok=False, error="restart failed")))

    result = service.update_flow_while_running(100.0, 20.0)

    assert not result.ok and not result.still_running
    assert result.safe_stop_verified and result.rolled_back
    assert result.stop_verified_before_write and not result.restart_verified
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "stop", 1, 2, "start", "stop", 1, 2, "stop"]


def test_partial_two_channel_update_stops_rolls_back_and_stays_stopped(monkeypatch):
    service = PumpHardwareService()
    service.runtime_config.inter_channel_update_delay = 0.0
    service.runtime_config.q2_update_retry_interval = 0.0
    service.runtime_config.q2_update_max_attempts = 1
    run_state = RunState(1, 0x03, True, [True, True, False, False])
    monkeypatch.setattr(
        service,
        "read_run_state",
        lambda: PumpOperationResult(ok=True, parsed_reply=run_state),
    )
    old = {
        1: service._default_channel_params_for_q(1, 50.0),
        2: service._default_channel_params_for_q(2, 20.0),
    }

    def build(channel, flow):
        service.last_channel_params[channel] = old[channel]
        return service._default_channel_params_for_q(channel, flow)

    monkeypatch.setattr(service, "_channel_params_with_flow", build)
    writes: list[tuple[int, ChannelParams]] = []

    def write(channel, params):
        writes.append((channel, params))
        if channel == 2:
            return PumpOperationResult(ok=False, error="Q2 failed")
        return PumpOperationResult(ok=True, parsed_reply=params, verified=True)

    monkeypatch.setattr(service, "write_wsp_and_verify", write)
    stop_calls = 0

    def stop():
        nonlocal stop_calls
        stop_calls += 1
        return PumpOperationResult(ok=True, verified=True)

    monkeypatch.setattr(service, "stop_system_and_verify", stop)

    result = service.update_flow_while_running(55.0, 22.0)

    assert not result.ok
    assert not result.still_running
    assert result.safe_stop_verified
    assert result.rolled_back
    assert stop_calls == 3
    assert [channel for channel, _ in writes] == [1, 2, 1]
    assert writes[-1][1] == old[1]
