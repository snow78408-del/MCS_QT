from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from frontend.qt_app import RoiImageLabel


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_lower_wall_can_be_clicked_from_bottom_letterbox() -> None:
    app = _application()
    label = RoiImageLabel()
    label.resize(800, 500)
    source = QPixmap(640, 360)
    source.fill(QColor("black"))
    label._source = source
    label.set_hough_lines(
        [{"id": 1, "x1": 0.05, "y1": 1.0, "x2": 0.95, "y2": 1.0}]
    )
    selected: list[list[dict[str, float]]] = []
    label.wallLinesChanged.connect(lambda lines: selected.append(list(lines)))
    label.show()
    app.processEvents()

    image_rect = label.image_rect()
    click = QPoint(image_rect.center().x(), image_rect.bottom() + 10)
    assert click.y() < label.height()
    QTest.mouseClick(label, Qt.LeftButton, Qt.NoModifier, click)
    app.processEvents()

    assert len(selected) == 1
    assert selected[0][0]["y1"] == 1.0
    label.close()
