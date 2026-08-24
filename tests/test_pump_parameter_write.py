from backend.pump_hardware.models import PumpOperationResult, RunState, SystemSetup
from backend.pump_hardware.service import PumpHardwareService


def test_prepare_parameter_write_does_not_repeat_rss_after_verified_wss(monkeypatch):
    service = PumpHardwareService()
    setup = SystemSetup(
        enable_mask=0x03,
        copy_mask=0x03,
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
        enable_mask=0x00,
        copy_mask=0x03,
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

    def write_and_verify(channel, params):
        calls.append(channel)
        if channel == 2 and calls.count(2) == 1:
            return PumpOperationResult(ok=False, error="temporary Q2 timeout")
        return PumpOperationResult(ok=True, parsed_reply=params, verified=True)

    monkeypatch.setattr(service, "write_wsp_and_verify", write_and_verify)

    result = service.update_flow_while_running(50.0, 20.0)

    assert result.ok
    assert result.q1_ok and result.q2_ok
    assert calls == [1, 2, 2]
