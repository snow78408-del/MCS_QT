from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QWidget

from backend.vision.config import ChannelRegionConfig, DetectorConfig
from backend.vision.tuning import PipelineStage, TuningFrame
import frontend.vision_tuning as vision_tuning_ui
from frontend.vision_tuning import StageCard, TuningWindow, _validate_tuning_configs


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
    detector = DetectorConfig(hough_dp=float("nan"))
    with pytest.raises(ValueError, match="有限数值"):
        _validate_tuning_configs(detector, ChannelRegionConfig())

    detector = DetectorConfig(hough_min_radius=90, hough_max_radius=20)
    with pytest.raises(ValueError, match="最小半径不能大于最大半径"):
        _validate_tuning_configs(detector, ChannelRegionConfig())


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
