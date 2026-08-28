from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QWidget

from backend.vision.tuning import PipelineStage
from frontend.vision_tuning import StageCard


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
