"""生成区测量接入采集路径的无硬件端到端测试。

被测路径：原始帧 → `PipelineVisionService.measure_generation_zone` /
`_snapshot_from_frame` → `RecognitionSnapshot.generation_measurement`。
用合成帧驱动，不打开相机、不连泵、不写任何原始数据目录。
"""
from __future__ import annotations

import numpy as np
import pytest

import backend.vision.service as vision_service_module
from backend.orchestrator.vision_adapter import PipelineVisionService

WIDTH = 720
HEIGHT = 540
CHANNEL_UM = 50.0
DUCT_PX = 40.0
SCALE = CHANNEL_UM / DUCT_PX
CHANNEL_OFFSET = 60.0


def _transverse_profile(rows: int, rng: np.random.Generator, contrast: float) -> np.ndarray:
    profile = np.full(rows, 48.0, np.float32)
    outer = max(1, int(round(rows * 0.22)))
    profile[:outer] += contrast
    profile[rows - outer:] += contrast
    profile[outer:rows - outer] -= contrast
    return profile + rng.normal(0.0, 0.6, rows)


def _channel_frame(*, offset: float = CHANNEL_OFFSET, separation: float = DUCT_PX,
                   plugs=((80, 200), (240, 360), (400, 560)), seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = int(round(separation))
    canvas = np.full((HEIGHT, WIDTH), 20.0, np.float32) + rng.normal(0.0, 0.6, (HEIGHT, WIDTH))
    for x in range(WIDTH):
        y0 = int(round(offset))
        canvas[y0:y0 + rows, x] = 40.0 + rng.normal(0.0, 0.6, rows)
    for left, right in plugs:
        for x in range(left, right):
            y0 = int(round(offset))
            canvas[y0:y0 + rows, x] = _transverse_profile(rows, rng, 9.0)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _walls(*, offset: float = CHANNEL_OFFSET, separation: float = DUCT_PX):
    def norm(y: float) -> float:
        return float(y) / float(HEIGHT)

    return [
        {"x1": 0.0, "y1": norm(offset), "x2": 1.0, "y2": norm(offset)},
        {"x1": 0.0, "y1": norm(offset + separation), "x2": 1.0, "y2": norm(offset + separation)},
    ]


class _StubCameraService:
    def __init__(self, logger=None) -> None:
        self._logger = logger
        self.selected = None

    def select_camera(self, unique_id, backend_name=None):
        self.selected = (unique_id, backend_name)
        return {"ok": True, "unique_id": unique_id, "backend": backend_name,
                "selected_backend": backend_name or "hikrobot",
                "device_type": "industrial_camera"}

    def open_camera(self):
        return {"ok": True}

    def __getattr__(self, name):
        """其余相机服务方法一律当作无操作，避免把桩写成整份相机 API。"""
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: {}

    def stop(self) -> None:
        pass


MODE_KEYS = ("localization_enabled", "strict_detection_localization")


def _roi_config(**overrides) -> dict:
    """ROI 几何配置。定位模式键**不在这里**——它们走 set_localization_mode()。"""
    values = {
        "enabled": True,
        "user_defined": True,
        "x_start_ratio": 0.0,
        "x_end_ratio": 1.0,
        "y_start_ratio": 0.1,
        "y_end_ratio": 0.6,
        "channel_calibration_enabled": True,
        "channel_width_um": CHANNEL_UM,
        "wall_lines": _walls(),
        "flow_direction": "negative",
        "generation_measurement_enabled": True,
        "scale_validated": True,
    }
    values.update(overrides)
    for key in MODE_KEYS:
        values.pop(key, None)
    return values


def _apply_roi(instance, **overrides) -> None:
    """几何与定位模式分开提交，与生产配置链一致。

    ``localization_enabled`` 缺省时传 None（不改该位），让 ``strict=True`` 自行蕴含
    ``enabled=True``；传 False 会与 strict 冲突并被显式接口拒绝。
    """
    values = dict(overrides)
    mode = {key: values.pop(key) for key in MODE_KEYS if key in values}
    instance.set_recognition_roi(_roi_config(**values))
    if mode:
        instance.set_localization_mode(
            strict=bool(mode.get("strict_detection_localization", False)),
            enabled=(bool(mode["localization_enabled"]) if "localization_enabled" in mode else None),
            reason="test_apply_roi")


@pytest.fixture()
def service(monkeypatch):
    """真实 PipelineVisionService + 桩相机；**不声明芯片深度**。

    刻意保持“只设置标尺、从未设置芯片深度”的状态：这是第二轮审核用来复现默认深度
    回退的场景，任何需要体积等效尺寸的测试必须自己显式声明深度。
    """
    return _build_service(monkeypatch)


@pytest.fixture()
def service_with_chip_depth(monkeypatch):
    """同 `service`，但通过公开配置显式声明了芯片深度与来源。"""
    return _build_service(monkeypatch, chip_depth_um=CHANNEL_UM)


def _build_service(monkeypatch, **roi_overrides) -> PipelineVisionService:
    monkeypatch.setattr(vision_service_module, "VisionCameraService", _StubCameraService)
    instance = PipelineVisionService()
    instance._log = lambda _message: None
    instance._video_source_type = "camera"
    instance._video_source = "synthetic"
    _apply_roi(instance, **roi_overrides)
    instance._channel_calibration_status = "calibrated"
    instance._channel_calibration_confidence = 1.0
    instance._channel_width_px = DUCT_PX
    instance._pixel_to_micron = SCALE
    instance._configured_pixel_to_micron = 1.725
    instance._pinned_batch_metadata = {}
    instance._frame_metadata = {frame_id: {"capture_monotonic": 100.0 + frame_id,
                                          "hardware_frame_id": frame_id}
                                for frame_id in (11, 12, 13)}
    instance._ensure_pipeline().detector.configure_expected_diameter(0.0, SCALE)
    return instance


def _measure(service, frame, **kwargs):
    defaults = dict(frame_id=11, hardware_frame_id=11, capture_monotonic=100.0,
                    time_source="camera_frame_timestamp")
    defaults.update(kwargs)
    return service.measure_generation_zone(frame, **defaults)


def test_scale_evidence_is_reported_and_unvalidated_scale_produces_no_microns(service) -> None:
    service._scale_declared = False
    service._scale_validated = False
    payload = _measure(service, _channel_frame())
    assert payload["valid"] is False, payload
    assert payload["reason"] == "scale_unvalidated"
    assert payload["scale"]["validated"] is False
    assert payload["plug_lengths_um"] == []
    assert payload["scale"]["source"] == "channel_width_reference"
    assert payload["scale"]["reference_um"] == CHANNEL_UM


def test_validated_scale_and_declared_depth_produce_microns(service_with_chip_depth) -> None:
    payload = _measure(service_with_chip_depth, _channel_frame())
    assert payload["valid"] is True, payload
    assert payload["duct"]["width_source"] == "rectified_image_measurement"
    assert payload["duct"]["depth_source"] == "declared_chip_geometry"
    assert payload["duct"]["depth_validated"] is False
    assert payload["duct"]["width_px"] == pytest.approx(DUCT_PX, abs=1.0)
    assert payload["duct"]["depth_px"] == pytest.approx(CHANNEL_UM / SCALE, abs=1.0)
    assert payload["duct"]["detector_width_axis_consistent"] is True
    assert payload["duct"]["detector_depth_axis_consistent"] is True
    assert payload["frame"]["frame_id"] == 11
    assert payload["frame"]["hardware_frame_id"] == 11
    assert payload["frame"]["localization_frame_id"] == 11
    assert len(payload["equivalent_diameters_um"]) == payload["plug_count"] == 3
    assert payload["wall_verification"]["status"] == "verified"


def test_undeclared_chip_depth_refuses_volume_sizes_even_with_validated_scale(service) -> None:
    """只设置标尺、从未设置芯片深度：不得把 DetectorConfig 默认值当成声明深度。

    Codex 第二轮复现（UNDECLARED_DEPTH）：修复前返回 valid=true、depth_um=50、
    depth_source=declared_chip_geometry 并给出三项等效直径。
    """
    assert service._chip_depth_um is None
    payload = _measure(service, _channel_frame())
    assert payload["valid"] is False
    assert payload["reason"] == "duct_depth_unknown"
    assert payload["duct"]["depth_um"] is None
    assert payload["duct"]["depth_source"] == "unknown"
    assert payload["duct"]["usable_for_volume"] is False
    assert payload["equivalent_diameters_px"] == []
    assert payload["equivalent_diameters_um"] == []
    # 轴向长度不需要深度，仍然给出
    assert len(payload["plug_lengths_um"]) == payload["plug_count"] == 3
    # detector 内部仍有默认通道尺寸，但那不是“已声明深度”
    assert payload["duct"]["detector_duct_geometry_px"] is not None


def test_chip_depth_declaration_is_honoured_and_validated_flag_reported(monkeypatch) -> None:
    monkeypatch.setattr(vision_service_module, "VisionCameraService", _StubCameraService)
    instance = PipelineVisionService()
    instance._log = lambda _message: None
    instance.set_recognition_roi(_roi_config(
        chip_depth_um=CHANNEL_UM, chip_depth_source="measured_depth",
        chip_depth_validated=True))
    assert instance._chip_depth_um == CHANNEL_UM
    assert instance._chip_depth_source == "measured_depth"
    assert instance._chip_depth_validated is True


@pytest.mark.parametrize("bad_depth", (None, "", 0.0, -3.0, float("nan"), "abc"))
def test_invalid_chip_depth_declarations_fall_back_to_unknown(monkeypatch, bad_depth) -> None:
    monkeypatch.setattr(vision_service_module, "VisionCameraService", _StubCameraService)
    instance = PipelineVisionService()
    instance.set_recognition_roi(_roi_config(chip_depth_um=bad_depth))
    assert instance._chip_depth_um is None
    assert instance._chip_depth_source == "unknown"
    assert instance._chip_depth_validated is False


def test_unknown_depth_source_label_is_not_accepted_as_declared(monkeypatch) -> None:
    monkeypatch.setattr(vision_service_module, "VisionCameraService", _StubCameraService)
    instance = PipelineVisionService()
    instance.set_recognition_roi(_roi_config(chip_depth_um=CHANNEL_UM,
                                             chip_depth_source="whatever"))
    assert instance._chip_depth_source == "unknown"


def test_wall_verification_runs_on_the_measured_frame_and_blocks_a_stale_roi(service) -> None:
    """复用几何与实际通道差 90 px 时必须拒绝，而不是退回默认几何。"""
    service.set_recognition_roi(_roi_config(chip_depth_um=CHANNEL_UM,
                                            wall_lines=_walls(offset=150.0)))
    payload = _measure(service, _channel_frame())
    assert payload["valid"] is False
    assert payload["reason"] in {"wall_geometry_stale", "wall_geometry_unverified"}
    assert payload["wall_verification"]["consistent"] is False


def test_wall_verification_is_bound_to_the_measured_frame_id(service) -> None:
    payload = _measure(service, _channel_frame(), frame_id=12)
    assert payload["wall_verification"]["verified_frame_id"] == 12


def test_zero_hardware_frame_id_is_refused_on_the_acquisition_path(service) -> None:
    payload = _measure(service, _channel_frame(), hardware_frame_id=0)
    assert payload["valid"] is False
    assert payload["reason"] == "hardware_frame_id_invalid"


def test_undeclared_time_source_is_refused(service) -> None:
    payload = _measure(service, _channel_frame(), time_source="wall_clock")
    assert payload["valid"] is False
    assert payload["reason"] == "time_source_undeclared"


def test_field_style_misconfigured_detector_duct_is_refused(service) -> None:
    """2026-09-21 的现场错误：detector 按 50/1.725 ≈ 29 px 建模，图上通道是 40 px。"""
    pipeline = service._ensure_pipeline()
    pipeline.detector.configure_expected_diameter(0.0, 1.725)
    payload = _measure(service, _channel_frame())
    assert payload["valid"] is False
    assert payload["reason"] == "detector_duct_geometry_mismatch"
    assert payload["duct"]["detector_geometry_consistent"] is False
    assert payload["duct"]["detector_duct_geometry_px"][0] == pytest.approx(50.0 / 1.725, abs=0.5)


def test_explicit_depth_override_is_used(service) -> None:
    payload = _measure(service, _channel_frame(), duct_depth_um=100.0,
                       duct_depth_source="measured_depth")
    assert payload["duct"]["depth_um"] == 100.0
    assert payload["duct"]["depth_source"] == "measured_depth"


# ---------------------------------------------------------------- 快照输出

def test_snapshot_carries_the_measurement_when_enabled(service_with_chip_depth) -> None:
    snapshot = service_with_chip_depth._snapshot_from_frame(
        _channel_frame(), frame_id=11, timestamp=100.0)
    payload = snapshot.generation_measurement
    assert payload is not None
    assert payload["source"] == "vision_adapter.measure_generation_zone"
    assert payload["frame"]["frame_id"] == 11
    assert payload["frame"]["hardware_frame_id"] == 11
    assert payload["frame"]["time_source"] == "host_clock_proxy"
    assert payload["frame"]["time_is_proxy"] is True
    assert payload["scale"]["um_per_px"] == pytest.approx(SCALE)
    assert payload["duct"]["depth_source"] == "declared_chip_geometry"
    assert payload["valid"] is True
    assert payload["reason"] == "ok"


def test_snapshot_omits_the_measurement_when_disabled(service) -> None:
    service._generation_measurement_enabled = False
    snapshot = service._snapshot_from_frame(_channel_frame(), frame_id=11, timestamp=100.0)
    assert snapshot.generation_measurement is None


def test_snapshot_records_the_rejection_reason_from_the_acquisition_path(service) -> None:
    service.set_recognition_roi(_roi_config(chip_depth_um=CHANNEL_UM,
                                            wall_lines=_walls(offset=150.0)))
    snapshot = service._snapshot_from_frame(_channel_frame(), frame_id=11, timestamp=100.0)
    payload = snapshot.generation_measurement
    assert payload is not None
    assert payload["valid"] is False
    assert payload["reason"] in {"wall_geometry_stale", "wall_geometry_unverified"}


def test_snapshot_records_missing_wall_geometry_as_an_explicit_reason(service) -> None:
    service.set_recognition_roi(_roi_config(chip_depth_um=CHANNEL_UM, wall_lines=[]))
    snapshot = service._snapshot_from_frame(_channel_frame(), frame_id=11, timestamp=100.0)
    assert snapshot.generation_measurement is not None
    assert snapshot.generation_measurement["reason"] == "wall_geometry_missing"


def test_measurement_is_off_by_default_after_construction(monkeypatch) -> None:
    monkeypatch.setattr(vision_service_module, "VisionCameraService", _StubCameraService)
    instance = PipelineVisionService()
    assert instance._generation_measurement_enabled is False
    assert instance._scale_validated is False
    assert instance._chip_depth_um is None
    assert instance._localization_enabled is False


# ---------------------------------------------------------------- 证据适用范围失效

def test_scale_validation_expires_when_the_image_size_changes(service_with_chip_depth) -> None:
    """标尺的“已验证”只对它被声明时的成像范围有效；图像尺寸一变就作废。"""
    import cv2

    service = service_with_chip_depth
    first = _measure(service, _channel_frame())
    assert first["valid"] is True, first
    assert first["scale"]["validated"] is True

    resized = cv2.resize(_channel_frame(), (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA)
    second = service.measure_generation_zone(
        resized, frame_id=12, hardware_frame_id=12, capture_monotonic=101.0,
        time_source="camera_frame_timestamp")
    assert second["scale"]["validated"] is False
    assert second["scale_scope_change"] is not None
    assert "image_shape" in second["scale_scope_change"]["changed"]
    assert second["valid"] is False
    assert second["reason"] in {"scale_unvalidated", "wall_geometry_stale",
                                "wall_geometry_unverified", "detector_duct_geometry_mismatch"}


def test_same_scope_keeps_the_scale_valid(service) -> None:
    for frame_id in (11, 12, 13):
        payload = service.measure_generation_zone(
            _channel_frame(), frame_id=frame_id, hardware_frame_id=frame_id,
            capture_monotonic=100.0 + frame_id, time_source="camera_frame_timestamp")
        assert payload["scale"]["validated"] is True, payload
        assert payload["scale_scope_change"] is None


def test_reconfiguring_the_roi_resets_the_scale_scope(service_with_chip_depth) -> None:
    """改采集条件（重新保存 ROI）后，旧标尺的“已验证”范围作废，需重新声明。"""
    service = service_with_chip_depth
    assert _measure(service, _channel_frame())["valid"] is True
    assert service._scale_scope is not None

    service.set_recognition_roi(_roi_config(chip_depth_um=CHANNEL_UM))
    assert service._scale_scope is None
    assert service._scale_validated is False

    # 重新声明后，下一次测量重新建立范围
    assert _measure(service, _channel_frame())["valid"] is True
    assert service._scale_validated is True


# ---------------------------------------------------------------- 定位链门控

TRAIN_EARLY = ((40, 180), (250, 390), (460, 600))
TRAIN_LATE = ((110, 250), (320, 460), (530, 670))


def _localization_sequence():
    """交替的两列柱塞，位移 70 px：管内内容明显变化而壁面不动。"""
    return [_channel_frame(plugs=TRAIN_EARLY, seed=11),
            _channel_frame(plugs=TRAIN_LATE, seed=12),
            _channel_frame(plugs=TRAIN_EARLY, seed=13)]


def test_pending_localization_blocks_the_measurement_with_its_reason(service) -> None:
    _apply_roi(service, chip_depth_um=CHANNEL_UM, localization_enabled=True)
    frame = _channel_frame()
    payload = service.measure_generation_zone(
        frame, frame_id=11, hardware_frame_id=11, capture_monotonic=100.0,
        time_source="camera_frame_timestamp")
    assert payload["valid"] is False
    assert payload["reason"].startswith("localization_")
    assert payload["localization"] is not None
    assert payload["localization"]["status"] in {"pending_motion", "insufficient_support",
                                                 "rejected", "ambiguous"}
    # 禁止落回旧 ROI：拒绝时不得给出任何可用壁线
    assert payload.get("plug_lengths_um", []) in ([], None)


def test_localization_rejection_reason_reaches_the_snapshot(service) -> None:
    _apply_roi(service, chip_depth_um=CHANNEL_UM, localization_enabled=True)
    snapshot = service._snapshot_from_frame(_channel_frame(), frame_id=11, timestamp=100.0)
    assert snapshot.generation_measurement is not None
    assert snapshot.generation_measurement["valid"] is False
    assert snapshot.measurement_quality_valid is False
    assert "生成区测量未通过" in snapshot.measurement_quality_reason
    assert snapshot.frame_diameters == []
    assert snapshot.avg_diameter is None
    assert snapshot.valid_for_control is False


def test_localized_sequence_produces_sizes_from_the_new_path(service) -> None:
    _apply_roi(service, chip_depth_um=CHANNEL_UM, localization_enabled=True)
    snapshot = None
    for index, frame in enumerate(_localization_sequence()):
        snapshot = service._snapshot_from_frame(frame, frame_id=11 + index,
                                                timestamp=100.0 + index)
    payload = snapshot.generation_measurement
    assert payload is not None
    assert payload["localization"]["status"] == "localized", payload["localization"]["reason"]
    assert payload["valid"] is True, payload
    assert snapshot.measurement_quality_valid is True
    assert snapshot.frame_diameters, "启用新路径后尺寸列必须来自该路径"
    assert snapshot.avg_diameter is not None


# ---------------------------------------------------------------- 写入门控

def test_enabled_path_clears_the_size_column_when_the_measurement_is_rejected(service) -> None:
    service.set_recognition_roi(_roi_config(chip_depth_um=CHANNEL_UM,
                                            wall_lines=_walls(offset=150.0)))
    snapshot = service._snapshot_from_frame(_channel_frame(), frame_id=11, timestamp=100.0)
    assert snapshot.generation_measurement["valid"] is False
    assert snapshot.frame_diameters == []
    assert snapshot.avg_diameter is None
    assert snapshot.frame_diameter_sum == 0.0
    assert snapshot.measurement_quality_valid is False
    assert snapshot.valid_for_control is False


def test_disabled_path_leaves_the_old_size_column_untouched(service) -> None:
    service._generation_measurement_enabled = False
    snapshot = service._snapshot_from_frame(_channel_frame(), frame_id=11, timestamp=100.0)
    assert snapshot.generation_measurement is None
    assert snapshot.measurement_quality_reason != ""


def test_live_strict_gate_blocks_stale_manual_roi_before_any_detection(service, monkeypatch) -> None:
    _apply_roi(service, chip_depth_um=CHANNEL_UM,
               strict_detection_localization=True,
               wall_lines=_walls(offset=150.0))
    detector = service._ensure_pipeline().detector
    monkeypatch.setattr(detector, "detect", lambda _frame: pytest.fail("stale ROI detected"))
    snapshot = service._snapshot_from_frame(_channel_frame(), frame_id=11, timestamp=100.0)
    assert snapshot.channel_region_status == "rejected"
    assert snapshot.valid_for_control is False
    assert snapshot.frame_droplet_count == 0
    assert snapshot.channel_region_reason


def test_stop_clears_the_localization_evidence(service) -> None:
    _apply_roi(service, chip_depth_um=CHANNEL_UM, localization_enabled=True)
    for index, frame in enumerate(_localization_sequence()):
        service.localize_parallel_walls(frame, frame_id=11 + index, capture_monotonic=100.0 + index)
    assert service._localizer.buffered_frames == 3
    service.stop()
    assert service._localizer.buffered_frames == 0


def test_roi_update_between_measurements_does_not_deadlock_or_mix_geometry(service) -> None:
    """配置更新与测量交替进行：不得死锁，也不得把旧定位几何用到新配置上。"""
    _apply_roi(service, chip_depth_um=CHANNEL_UM, localization_enabled=True)
    snapshot = None
    for index, frame in enumerate(_localization_sequence()):
        snapshot = service._snapshot_from_frame(frame, frame_id=11 + index,
                                                timestamp=100.0 + index)
    assert snapshot.generation_measurement["valid"] is True

    _apply_roi(service, chip_depth_um=CHANNEL_UM,
               localization_enabled=True,
               wall_lines=_walls(offset=150.0))
    # 换 ROI 后定位证据必须重建，不允许沿用上一份几何
    assert service._localizer is None
    second = _measure(service, _channel_frame())
    assert second["localization"]["status"] != "localized"
    assert second["valid"] is False
