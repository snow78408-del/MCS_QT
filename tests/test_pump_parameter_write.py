from backend.pump_hardware.models import PumpOperationResult, SystemSetup
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
