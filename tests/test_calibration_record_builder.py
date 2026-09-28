"""标定记录公共 builder 的回归测试。

覆盖 P0-1：``schema_version=3`` 的记录曾由两条辨识分支（``linear`` 与
``quadratic_response``）各自复制一份字段装配代码，结果漂移：

- 二次响应曲面分支漏建 per-channel 延迟与基线生成频率/CV；
- 该分支的 ``calibration_id`` 是裸 uuid，无法回溯到具体运行；
- ``linear`` 传归一化后的值，二次分支传 ``config`` 原始值。

这里钉住修复后的不变量：

1. 曾经取值不同的四个字段**没有默认值**，调用方无法静默继承（字段静默缺失正是漂移成因）；
2. ``calibration_id`` 可回溯，含 UTC 时间戳与 session 前缀；
3. ``config`` 派生的字符串/数值归一化只在 builder 内做一份；
4. builder 负责的那些固定字段不会被分支各写各的。

两条分支的端到端调用点由 ``test_nonlinear_calibration.py`` 与
``test_plant_calibration_experiment.py`` 覆盖——若哪个分支漏传必填参数，那些测试会直接
抛 ``TypeError``，因此这里不必重复构造两套测量数据。
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backend.pid_control.calibration import build_calibration_record
from test_nonlinear_calibration import build as build_nonlinear, nonlinear_measurements
from test_plant_calibration_experiment import _config

TRACEABLE_ID = re.compile(r"^plant-cal-\d{8}T\d{6}Z-.{1,8}$")

# 这两条分支的字段曾因「靠 dataclass 默认值兜底」而漂移。
DRIFTED_FIELDS = (
    "q1_response_delay_ms",
    "q2_response_delay_ms",
    "baseline_generation_frequency_hz",
    "baseline_diameter_cv",
)

_BASE_KWARGS: dict[str, object] = {
    "config": _config(),
    "session_id": "session-abcdefgh",
    "completed_at": datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
    "measurement_source": "test_source",
    "response_delay_median_ms": 1000.0,
    "response_delay_uncertainty_ms": 50.0,
    "diameter_sensitivity_um_per_output": -2.0,
    "q1_control_sign": -1.0,
    "q2_control_sign": 1.0,
    "q1_output_gain": 0.5,
    "q2_output_gain": 0.25,
    "q1_min": 15.0,
    "q1_max": 100.0,
    "q2_min": 5.0,
    "q2_max": 25.0,
    "total_flow_max": 125.0,
    "min_q1_q2_gap": 1.0,
    "baseline_q1": 50.0,
    "baseline_q2": 20.0,
    "baseline_diameter_um": 60.0,
    "q1_log_diameter_sensitivity": -0.4,
    "q2_log_diameter_sensitivity": 0.2,
    "response_time_constant_ms": 1200.0,
    "controller_kp": 0.7,
    "controller_ki": 0.05,
    "response_time_constant_uncertainty_ms": 80.0,
    "q1_log_sensitivity_uncertainty": 0.02,
    "q2_log_sensitivity_uncertainty": 0.01,
    "model_fit_method": "robust_fopdt_grid",
    "model_fit_mae_um": 1.0,
    "model_fit_nrmse": 0.02,
    "validation_mae_um": 1.2,
    "validation_nrmse": 0.03,
    "validation_sample_count": 14,
    "validated_for_pi": True,
    "validated_for_mpc": False,
    "q1_response_delay_ms": 900.0,
    "q2_response_delay_ms": 950.0,
    "baseline_generation_frequency_hz": 1200.0,
    "baseline_diameter_cv": 0.04,
}


def _build(**overrides: object):
    kwargs = dict(_BASE_KWARGS)
    kwargs.update(overrides)
    return build_calibration_record(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", DRIFTED_FIELDS)
def test_drifted_fields_have_no_default_so_they_cannot_be_omitted(name: str) -> None:
    """四个漂移字段必须显式传入：默认值兜底让一条分支漏建了字段而无人察觉。"""
    parameter = inspect.signature(build_calibration_record).parameters[name]
    assert parameter.default is inspect.Parameter.empty, f"{name} 不应有默认值"

    kwargs = {key: value for key, value in _BASE_KWARGS.items() if key != name}
    with pytest.raises(TypeError):
        build_calibration_record(**kwargs)  # type: ignore[arg-type]


def test_calibration_id_is_traceable_to_the_run() -> None:
    record = _build()
    assert TRACEABLE_ID.match(record.calibration_id), record.calibration_id
    assert record.calibration_id == "plant-cal-20260920T120000Z-session-"


def test_nonlinear_branch_record_id_is_also_traceable() -> None:
    """修复前二次分支用裸 uuid 作 id，记录无法回溯到具体运行。"""
    record = build_nonlinear(nonlinear_measurements()).record
    assert TRACEABLE_ID.match(record.calibration_id), record.calibration_id
    assert "nonlinea" in record.calibration_id


def test_builder_owns_the_shared_constant_fields() -> None:
    """这些字段两条分支取值相同，改由 builder 负责，避免各写各的。"""
    record = _build()
    assert record.schema_version == 3
    assert record.measurement_region == "generation"
    assert record.controller_kd == 0.0
    assert record.flow_measurement_kind == "device_parameter_readback"
    assert record.measurement_source == "test_source"


def test_config_derived_values_are_normalized_in_the_builder() -> None:
    """归一化只有一份：带空白或字符串形态的 config 值必须被规整。"""
    config = SimpleNamespace(
        plant_id="  rig-a  ",
        chip_id="chip-a\n",
        fluid_id=" water-oil-a ",
        pump_model="pump-a",
        syringe_profile=" 10ml-glass ",
        channel_height_um="120",
        channel_width_um="400.0",
        volume_correction_factor="0.98",
        sensitivity_allocation_regularization="0.02",
        continuous_phase_oil=" fluorinated oil ",
        surfactant_name=" span-80 ",
        surfactant_concentration_percent="2.0",
        surfactant_concentration_basis="v/v",
        aqueous_phase=" water ",
        temperature_c="25.5",
    )
    record = _build(config=config)
    assert record.plant_id == "rig-a"
    assert record.chip_id == "chip-a"
    assert record.fluid_id == "water-oil-a"
    assert record.syringe_profile == "10ml-glass"
    assert record.continuous_phase_oil == "fluorinated oil"
    assert record.surfactant_name == "span-80"
    assert record.aqueous_phase == "water"
    assert record.channel_height_um == pytest.approx(120.0)
    assert record.channel_width_um == pytest.approx(400.0)
    assert record.volume_correction_factor == pytest.approx(0.98)
    assert record.sensitivity_allocation_regularization == pytest.approx(0.02)
    assert record.surfactant_concentration_percent == pytest.approx(2.0)
    assert record.temperature_c == pytest.approx(25.5)


def test_nonlinear_model_is_carried_through_and_validated() -> None:
    model = {
        "kind": "quadratic_flow_response",
        "center": [50.0, 20.0],
        "scale": [2.0, 1.0],
        "coefficients": [0.0] * 5,
    }
    assert _build(nonlinear_model=model).nonlinear_model == model
    assert _build().nonlinear_model is None
    with pytest.raises(ValueError):
        _build(nonlinear_model={"kind": "unsupported"})
