"""定位模式不得被几何更新静默关闭（2026-09-24 修复的缺陷）。

修复前：``set_recognition_roi`` 对缺失的 ``strict_detection_localization`` 取默认
``False``，而 ``configure_generation_measurement`` 发送的是**不含该键**的配置副本，
于是「改一次生成区几何」就把严格当前帧定位关掉，且无任何提示。

修复后：模式开关只能经 ``set_localization_mode`` 显式切换，并经
``localization_mode_state`` 查询；ROI 字典携带模式键会被直接拒绝。
"""
from __future__ import annotations

import pytest

import backend.vision.service as vision_service_module
from backend.orchestrator.models import SystemConfig
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.orchestrator.vision_adapter import PipelineVisionService

CHANNEL_UM = 50.0
MODE_KEYS = ("localization_enabled", "strict_detection_localization")


class _StubCameraService:
    def __init__(self, logger=None) -> None:
        self._logger = logger

    def select_camera(self, unique_id, backend_name=None):
        return {"ok": True, "unique_id": unique_id, "backend": backend_name,
                "selected_backend": backend_name or "hikrobot",
                "device_type": "industrial_camera"}

    def open_camera(self):
        return {"ok": True}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: {}

    def stop(self) -> None:
        pass


def _roi(**overrides) -> dict:
    values = {"enabled": True, "user_defined": True, "x_start_ratio": 0.0, "x_end_ratio": 1.0,
              "y_start_ratio": 0.1, "y_end_ratio": 0.6, "channel_width_um": CHANNEL_UM,
              "wall_lines": [], "flow_direction": "negative",
              "generation_measurement_enabled": True, "scale_validated": True}
    values.update(overrides)
    for key in MODE_KEYS:
        values.pop(key, None)
    return values


@pytest.fixture()
def adapter(monkeypatch) -> PipelineVisionService:
    monkeypatch.setattr(vision_service_module, "VisionCameraService", _StubCameraService)
    instance = PipelineVisionService()
    instance._log = lambda _message: None
    instance._video_source_type = "camera"
    instance._video_source = "synthetic"
    return instance


@pytest.fixture()
def service(adapter) -> OrchestratorService:
    instance = OrchestratorService(vision_service=adapter)
    instance._log = lambda _message: None
    instance._cfg = SystemConfig(60.0, 1.25, "camera", "synthetic", 50.0, 20.0, 7500,
                                 pump_port="COM99", recognition_roi=_roi())
    instance._state = SystemState.STOPPED
    return instance


# ------------------------------------------------------------------ 初始化

def test_localization_mode_is_off_by_default(adapter) -> None:
    state = adapter.localization_mode_state()
    assert state["strict_detection_localization"] is False
    assert state["localization_enabled"] is False
    assert state["revision"] == 0
    assert state["last_change"] is None


# --------------------------------------------------------- 几何更新不得改动模式

def test_geometry_update_via_roi_does_not_touch_the_localization_mode(adapter) -> None:
    adapter.set_localization_mode(strict=True, reason="initial")
    before = adapter.localization_mode_state()
    adapter.set_recognition_roi(_roi(channel_width_um=60.0, generation_channel_height_um=60.0))
    after = adapter.localization_mode_state()
    assert after == before


def test_roi_update_rejects_mode_keys_instead_of_silently_defaulting(adapter) -> None:
    for key in MODE_KEYS:
        payload = _roi()
        payload[key] = True  # 绕过 _roi 的剥离，直接构造非法载荷
        with pytest.raises(ValueError, match=key):
            adapter.set_recognition_roi(payload)
    # 拒绝后模式仍未改变，也没有变成“未声明”状态
    assert adapter.localization_mode_state()["strict_detection_localization"] is False


def test_service_geometry_update_keeps_strict_localization(service) -> None:
    """修复前的直接反例：配置严格模式 → 改生成区几何 → 严格模式必须仍然生效。"""
    service.set_detection_localization_mode(strict=True, reason="test")
    assert service.detection_localization_mode()["strict_detection_localization"] is True
    service.configure_generation_measurement(channel_height_um=50.0, channel_width_um=50.0,
                                            volume_correction_factor=1.0)
    state = service.detection_localization_mode()
    assert state["strict_detection_localization"] is True
    assert state["localization_enabled"] is True
    assert state["last_change"]["reason"] == "test", "几何更新不得新增模式变更记录"


# ------------------------------------------------------------------ 显式切换

def test_explicit_switch_is_recorded_and_queryable(adapter) -> None:
    first = adapter.set_localization_mode(strict=True, reason="unit-test")
    assert first["strict_detection_localization"] is True
    assert first["localization_enabled"] is True
    assert first["revision"] == 1
    assert first["last_change"]["changed"] is True
    assert first["last_change"]["previous_strict"] is False

    second = adapter.set_localization_mode(strict=False, enabled=False, reason="unit-test-off")
    assert second["strict_detection_localization"] is False
    assert second["localization_enabled"] is False
    assert second["revision"] == 2
    assert second["last_change"]["previous_strict"] is True


def test_strict_implies_enabled_and_conflicting_request_is_refused(adapter) -> None:
    adapter.set_localization_mode(strict=True, enabled=True, reason="ok")
    assert adapter.localization_mode_state()["localization_enabled"] is True
    with pytest.raises(ValueError):
        adapter.set_localization_mode(strict=True, enabled=False, reason="conflict")


def test_turning_off_strict_keeps_the_enabled_bit(adapter) -> None:
    adapter.set_localization_mode(strict=True, reason="on")
    state = adapter.set_localization_mode(strict=False, reason="off-strict-only")
    assert state["strict_detection_localization"] is False
    assert state["localization_enabled"] is True


def test_mode_change_rebuilds_the_localizer(adapter) -> None:
    adapter.set_localization_mode(strict=True, reason="on")
    adapter._localizer = object()
    adapter.set_localization_mode(strict=False, reason="off")
    assert adapter._localizer is None, "模式改变必须重建 localizer（contrast_enhance 变了）"


# ------------------------------------------------------------------ 服务层状态限制

def test_service_refuses_mode_change_while_running(service) -> None:
    service._state = SystemState.RUNNING
    with pytest.raises(RuntimeError):
        service.set_detection_localization_mode(strict=True, reason="attempt")
    service._state = SystemState.CALIBRATING
    with pytest.raises(RuntimeError):
        service.set_detection_localization_mode(strict=True, reason="attempt")


def _stub_adapter_video_preparation(service, monkeypatch) -> None:
    """把适配器的 prepare_video 打成 no-op。

    定位模式的提交发生在 ``adapter.prepare_video(...)`` **之前**，因此只隔离后续的
    真实相机流水线，模式逻辑仍走真实 service/adapter 链。
    """
    monkeypatch.setattr(service.vision_adapter, "prepare_video", lambda *_a, **_k: None)


def test_full_config_establishes_strict_for_a_live_source(service, monkeypatch) -> None:
    """完整配置 + 视频准备：实时来源声明严格定位。

    定位模式原先是在 ``prepare_video()`` 里经 ROI 字典注入的；现在它经显式接口提交。
    """
    _stub_adapter_video_preparation(service, monkeypatch)
    cfg = SystemConfig(60.0, 1.25, "camera", "synthetic", 50.0, 20.0, 7500,
                       pump_port="COM99", recognition_roi=_roi())
    service.configure(cfg)
    service.prepare_video()
    state = service.detection_localization_mode()
    assert state["strict_detection_localization"] is True
    assert state["localization_enabled"] is True
    assert state["last_change"]["reason"].startswith("prepare_video:")


def test_full_config_does_not_enable_strict_for_a_video_source(service, monkeypatch) -> None:
    _stub_adapter_video_preparation(service, monkeypatch)
    cfg = SystemConfig(60.0, 1.25, "video", "sample.mp4", 50.0, 20.0, 7500,
                       recognition_roi=_roi())
    service.configure(cfg)
    service.prepare_video()
    state = service.detection_localization_mode()
    assert state["strict_detection_localization"] is False
    assert state["last_change"]["reason"].startswith("prepare_video:")


def test_prepare_video_declared_modes_are_honoured_for_a_video_source(service, monkeypatch) -> None:
    """非实时来源：显式声明的 localization_enabled 必须被尊重，不被默认值覆盖。"""
    _stub_adapter_video_preparation(service, monkeypatch)
    roi = _roi()
    roi["localization_enabled"] = True
    cfg = SystemConfig(60.0, 1.25, "video", "sample.mp4", 50.0, 20.0, 7500,
                       recognition_roi=roi)
    service.configure(cfg)
    service.prepare_video()
    state = service.detection_localization_mode()
    assert state["strict_detection_localization"] is False
    assert state["localization_enabled"] is True
