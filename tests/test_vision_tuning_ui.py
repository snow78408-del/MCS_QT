from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QMessageBox, QWidget

from backend.vision.config import ChannelRegionConfig, DetectorConfig
from backend.vision.tuning import PipelineStage, TuningFrame
import frontend.vision_tuning as vision_tuning_ui
from frontend.vision_tuning import StageCard, TuningWindow, _validate_tuning_configs
from frontend.vision_tuning_store import TuningLoadStatus, VisionTuningSettingsStore


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _stage(name: str) -> PipelineStage:
    return PipelineStage(
        name=name,
        description="用于验证卡片说明文字不会随相邻参数面板一起拉伸。",
        image=np.zeros((120, 220, 3), dtype=np.uint8),
        parameters="测试参数",
        statistics="测试结果",
    )


def _controls() -> list[dict[str, object]]:
    return [
        {
            "key": "enable_gaussian_blur",
            "label": "高斯模糊",
            "kind": "check",
            "value": True,
            "text": "执行此步骤",
        }
    ]


def test_stage_parameter_toggle_has_clear_collapsed_and_expanded_states() -> None:
    app = _application()
    card = StageCard(_stage("可调步骤"), _controls())

    assert card.parameter_toggle.text() == "＋  参数设置（点击展开）"
    assert not card.parameter_panel.isVisible()
    assert "background:#e7effc" in card.parameter_toggle.styleSheet()
    assert "color:#ffffff" in card.parameter_toggle.styleSheet()

    card.show()
    card.parameter_toggle.click()
    app.processEvents()

    assert card.parameter_toggle.text() == "－  参数设置（点击收起）"
    assert card.parameter_panel.isVisible()
    assert "点击收起" in card.parameter_toggle.accessibleDescription()
    card.close()


def _wait_until(app: QApplication, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert predicate()


def test_invalid_algorithm_parameters_are_rejected_before_inspection() -> None:
    detector = DetectorConfig(sensitivity=float("nan"))
    with pytest.raises(ValueError, match="有限数值"):
        _validate_tuning_configs(detector, ChannelRegionConfig())

    detector = DetectorConfig(min_radius=90, max_radius=20)
    with pytest.raises(ValueError, match="最小半径不能大于最大半径"):
        _validate_tuning_configs(detector, ChannelRegionConfig())

    detector = DetectorConfig(radius_adjustment_percent=21)
    with pytest.raises(ValueError, match="radius_adjustment_percent"):
        _validate_tuning_configs(detector, ChannelRegionConfig())


def test_final_stage_exposes_global_radius_adjustment() -> None:
    window = TuningWindow()

    controls = window._controls_for_stage(11)

    assert [control["key"] for control in controls] == ["radius_adjustment_percent"]
    assert controls[0]["label"] == "液滴整体尺寸调节（%）"
    window.close()


def test_algorithm_exception_rolls_back_without_escaping_gui_thread(monkeypatch) -> None:
    app = _application()
    window = TuningWindow()
    window.frames = [TuningFrame(7, np.zeros((120, 220, 3), dtype=np.uint8))]
    calls = 0

    def inspect(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(centers=[]), [_stage("成功画面")]
        raise RuntimeError("模拟算法故障")

    monkeypatch.setattr(vision_tuning_ui, "inspect_frame", inspect)
    window._redraw()
    _wait_until(app, lambda: window._active_request is None)
    assert window._last_good_stages

    original_dp = window.current_config.hough_dp
    window.current_config.hough_dp = 2.0
    window._config_revision += 1
    window._redraw()
    _wait_until(app, lambda: window._active_request is None)

    assert window.current_config.hough_dp == original_dp
    assert "已回退到上次有效参数" in window.status.text()
    window.close()


def test_incompatible_saved_parameters_prompt_and_can_be_recreated(tmp_path, monkeypatch) -> None:
    _application()
    path = tmp_path / "vision_tuning_parameters.json"
    path.write_text("old algorithm format", encoding="utf-8")
    prompts: list[str] = []

    def answer(_parent, _title, message, *_args):
        prompts.append(message)
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    store = VisionTuningSettingsStore(path)
    window = TuningWindow(settings_store=store)

    assert prompts and "是否删除老版本参数" in prompts[0]
    assert store.load_or_create().status is TuningLoadStatus.LOADED
    assert "重建默认算法参数" in window.status.text()
    window.close()


def test_save_button_overwrites_user_algorithm_parameters(tmp_path) -> None:
    _application()
    store = VisionTuningSettingsStore(tmp_path / "vision_tuning_parameters.json")
    window = TuningWindow(settings_store=store)
    window.current_config.sensitivity = 0.71

    window._save()

    assert store.load_or_create().detector.sensitivity == 0.71
    assert "替代原用户参数" in window.status.text()
    assert not window.reset.isEnabled()
    window.close()


def test_expanding_stage_does_not_stretch_neighbour_card() -> None:
    app = _application()
    container = QWidget()
    grid = QGridLayout(container)
    adjustable = StageCard(_stage("可调步骤"), _controls(), container)
    neighbour = StageCard(_stage("相邻步骤"), [], container)
    grid.addWidget(adjustable, 0, 0, Qt.AlignTop)
    grid.addWidget(neighbour, 0, 1, Qt.AlignTop)
    container.resize(1000, 1000)
    container.show()
    app.processEvents()
    collapsed_neighbour_height = neighbour.height()

    adjustable.parameter_toggle.click()
    app.processEvents()

    assert adjustable.height() > neighbour.height()
    assert neighbour.height() == collapsed_neighbour_height
    container.close()
