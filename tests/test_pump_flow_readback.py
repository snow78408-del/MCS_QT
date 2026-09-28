"""泵回读流量的解析与校验路径回归测试。

覆盖 P0-2：损坏的泵回读参数曾被 ``except Exception: return None`` 吞掉，再被调用方的
``or float(q)`` 兜底替换成指令值，于是 ``_flow_matches(name, commanded, actual)``
变成拿指令值和自己比较——校验恒真，损坏的回读被报告成「更新成功」。

这里同时钉住三件事，缺一不可：

1. 损坏回读必须抛 ``ChannelFlowParseError``，且**不能**回退成指令值；
2. 回读缺失（``None``）同样**不能**认定验证成功——指令值只能作为指令单独记录展示；
3. 校验本身仍能抓出真实不符——不能把校验修成恒假。

第 2 条曾按「缺失时沿用指令值」实现，现按复查要求改为拒绝：两者都会让
``_flow_matches(name, commanded, actual)`` 退化成拿指令值和自己比。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.orchestrator.service import OrchestratorService
from backend.pump_hardware.models import ChannelParams
from backend.pump_hardware.service import (
    ChannelFlowParseError,
    FlowReadbackUnavailableError,
    PumpHardwareService,
)

# _verified_flow_actual 只通过 self 取一个静态解析入口，用一个最小 stub 就能测，
# 无需构造完整的 OrchestratorService（它会拉起泵、相机和后台线程）。
_STUB = SimpleNamespace(
    _flow_from_channel_params_strict=PumpHardwareService.flow_from_channel_params_strict,
)
_verified_flow_actual = OrchestratorService._verified_flow_actual


def make_params(**overrides: object) -> ChannelParams:
    """构造一组可解析的回读参数：1000 × 1.0 uL / (100 × 0.1 min) = 100 uL/min。"""
    values: dict[str, object] = {
        "channel": 1,
        "mode": 1,
        "syringe_code": 0x18,
        "dispense_value": 1000,
        "dispense_unit": 4,
        "infuse_time_value": 100,
        "infuse_time_unit": 2,
        "withdraw_time_value": 100,
        "withdraw_time_unit": 1,
        "repeat_count": 10,
        "interval_value": 50,
    }
    values.update(overrides)
    return ChannelParams(**values)  # type: ignore[arg-type]


def test_valid_readback_parses_to_expected_flow() -> None:
    assert PumpHardwareService.flow_from_channel_params_strict(make_params()) == pytest.approx(100.0)


def test_absent_readback_is_none_and_not_an_error() -> None:
    assert PumpHardwareService.flow_from_channel_params_strict(None) is None
    assert PumpHardwareService.flow_from_channel_params(None) is None


def test_non_positive_infuse_time_is_a_parse_error() -> None:
    with pytest.raises(ChannelFlowParseError):
        PumpHardwareService.flow_from_channel_params_strict(make_params(infuse_time_value=0))


def test_unparseable_readback_is_a_parse_error() -> None:
    with pytest.raises(ChannelFlowParseError):
        PumpHardwareService.flow_from_channel_params_strict(make_params(dispense_value=None))


def test_lenient_parse_is_unchanged_for_logging_and_display() -> None:
    """宽松路径给日志/展示用，行为必须与改动前完全一致：损坏时返回 None，不抛。"""
    assert PumpHardwareService.flow_from_channel_params(make_params(dispense_value=None)) is None
    assert PumpHardwareService.flow_from_channel_params(make_params(infuse_time_value=0)) is None
    assert PumpHardwareService.flow_from_channel_params(make_params()) == pytest.approx(100.0)


def test_corrupt_readback_no_longer_becomes_the_commanded_value() -> None:
    """回归主断言：损坏回读必须失败，而不是被替换成指令值。"""
    commanded = 100.0
    with pytest.raises(ChannelFlowParseError):
        _verified_flow_actual(_STUB, make_params(dispense_value=None), commanded)


def test_absent_readback_is_rejected_not_replaced_by_the_command() -> None:
    """回读缺失不能算验证成功：指令值不是实测值。

    曾按「缺失时沿用指令值」实现；那与损坏回读一样会让 ``_flow_matches``
    退化成拿指令值和自己比，因此一并改为拒绝。
    """
    with pytest.raises(FlowReadbackUnavailableError):
        _verified_flow_actual(_STUB, None, 42.0)


def test_readback_mismatch_is_still_detected() -> None:
    """修复不能把校验改成恒假：真实不符仍必须被抓出来。"""
    commanded = 100.0
    actual = _verified_flow_actual(_STUB, make_params(dispense_value=2000), commanded)
    ok, reason = OrchestratorService._flow_matches("Q1", commanded, actual)
    assert ok is False
    assert "Q1" in reason


def test_matching_readback_is_still_accepted() -> None:
    """正常一致的回读仍判通过，避免修复过度报警。"""
    commanded = 100.0
    actual = _verified_flow_actual(_STUB, make_params(), commanded)
    ok, reason = OrchestratorService._flow_matches("Q1", commanded, actual)
    assert ok is True
    assert reason == ""


def test_the_old_fallback_made_the_check_vacuous() -> None:
    """钉住本次修复的动机：拿指令值和自己比，``_flow_matches`` 必然通过。

    修复前链路是：损坏回读 → ``None`` → ``or float(q)`` → ``actual == commanded``
    → 回读校验恒真 → 报告「更新成功」。这个反例固化下来，防止兜底逻辑被重新引入。
    """
    commanded = 100.0
    ok, reason = OrchestratorService._flow_matches("Q1", commanded, commanded)
    assert ok is True
    assert reason == ""


@pytest.mark.parametrize(
    "overrides",
    [
        {"dispense_unit": 99},          # 未知容积单位码
        {"infuse_time_unit": 99},       # 未知时间单位码
        {"dispense_unit": "abc"},       # 单位码不是整数
        {"dispense_value": float("nan")},
        {"dispense_value": float("inf")},
        {"dispense_value": float("-inf")},
        {"infuse_time_value": float("nan")},
        {"dispense_value": -100},       # 负容积
        {"infuse_time_value": -100},    # 负时间
        {"dispense_value": None},
        {"infuse_time_value": "abc"},
    ],
)
def test_strict_parse_rejects_invalid_readback(overrides: dict) -> None:
    """未知单位、非有限值、负容积、非正时间都必须被拒绝，不能静默通过。"""
    params = make_params(**overrides)
    with pytest.raises(ChannelFlowParseError):
        PumpHardwareService.flow_from_channel_params_strict(params)
    # 宽松接口仍供日志/展示使用：捕获后返回 None，不抛给调用方。
    assert PumpHardwareService.flow_from_channel_params(params) is None


def test_unknown_unit_code_is_not_silently_treated_as_one() -> None:
    """``.get(code, 1.0)`` 的默认倍率不得进入严格解析。

    未知容积单位码曾退化成 ×1.0，可能让换算流量差几个数量级。
    """
    with pytest.raises(ChannelFlowParseError):
        PumpHardwareService.flow_from_channel_params_strict(make_params(dispense_unit=99))


def test_flow_update_path_fails_when_readback_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """调用方必须进入失败路径，而不是把缺失回读报告成写入成功。"""
    service = OrchestratorService()
    monkeypatch.setattr(service._safety, "permits", lambda _token: True)
    monkeypatch.setattr(
        service,
        "_update_flow_with_lifecycle_guard",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True, verified_q1=None, verified_q2=None, reason=None, error=None,
        ),
    )
    with pytest.raises(FlowReadbackUnavailableError):
        service._apply_optimizer_flow(100.0, 5.0, 0, object())
