from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QToolButton

from backend.vision.tuning import PipelineStage
from frontend.vision_tuning import StageCard


_APPLICATION = QApplication.instance() or QApplication([])


def _card(*, modified: bool) -> StageCard:
    stage = PipelineStage(
        name="测试步骤",
        description="测试恢复按钮布局",
        image=np.zeros((40, 60, 3), dtype=np.uint8),
    )
    return StageCard(
        stage,
        [
            {
                "key": "hough_param2",
                "label": "Hough 累加阈值",
                "kind": "number",
                "value": 28.0 if not modified else 30.0,
                "modified": modified,
            }
        ],
    )


def test_parameter_reset_icon_uses_a_reserved_label_slot() -> None:
    original = _card(modified=False)
    modified = _card(modified=True)

    original_reset = original.findChild(QToolButton, "parameter-reset-hough_param2")
    modified_reset = modified.findChild(QToolButton, "parameter-reset-hough_param2")

    assert original_reset is not None
    assert modified_reset is not None
    assert original_reset.isHidden()
    assert not modified_reset.isHidden()
    assert original_reset.parentWidget().size() == modified_reset.parentWidget().size()
    assert original_reset.parentWidget().width() == 24


def test_parameter_reset_icon_emits_its_parameter_key() -> None:
    card = _card(modified=True)
    emitted: list[str] = []
    card.parameter_reset.connect(emitted.append)

    reset = card.findChild(QToolButton, "parameter-reset-hough_param2")
    assert reset is not None
    reset.click()

    assert emitted == ["hough_param2"]
