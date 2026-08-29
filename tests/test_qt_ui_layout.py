from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QSpinBox

from backend.orchestrator.state import SystemState
from frontend.qt_app import (
    CollapsibleSection,
    ParameterPage,
    StatusModule,
    StatusPage,
    VideoPage,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


class _PageApp:
    def __init__(self, config=None):
        self.frontend_config = dict(config or {})
        self.saved = {}
        self.shown = ""

    @staticmethod
    def title(heading, subtitle):
        return QLabel(f"{heading}\n{subtitle}")

    def save(self, **values):
        self.saved.update(values)
        self.frontend_config.update(values)

    def show_page(self, key):
        self.shown = key

    @staticmethod
    def error(_title, message):
        raise AssertionError(message)


def test_collapsible_section_has_clear_closed_and_open_states() -> None:
    app = _application()
    section = CollapsibleSection("高级设置")
    section.show()
    app.processEvents()

    assert not section.body.isVisible()
    assert section.toggle.text().startswith("▸")

    section.toggle.click()
    app.processEvents()

    assert section.body.isVisible()
    assert section.toggle.text().startswith("▾")
    section.close()


def test_status_module_keeps_key_metrics_compact() -> None:
    app = _application()
    panel = StatusModule()
    panel.show()
    panel.setPlainText("\n".join(f"指标 {index}" for index in range(9)))
    app.processEvents()

    assert "指标 0" in panel.summary.text()
    assert "指标 5" in panel.summary.text()
    assert "指标 6" not in panel.summary.text()
    assert not panel.details.isVisible()

    panel.toggle.click()
    app.processEvents()
    assert panel.details.isVisible()
    assert "指标 8" in panel.details.toPlainText()
    panel.close()


def test_parameter_page_uses_bounded_inputs_with_units() -> None:
    _application()
    page_app = _PageApp()
    page = ParameterPage(page_app)

    assert isinstance(page.target, QDoubleSpinBox)
    assert isinstance(page.interval, QSpinBox)
    assert page.target.suffix().strip() == "μm"
    assert page.interval.minimum() > 0

    page.submit()
    assert page_app.saved["target_diameter"] == page.target.value()
    assert page_app.shown == "video"
    page.close()


def test_video_page_hides_camera_transport_fields_for_local_video() -> None:
    app = _application()
    page = VideoPage(_PageApp())
    page.show()
    app.processEvents()
    assert page.devices.isVisible()
    assert page.scan_button.isVisible()
    assert "自动检定" in page.advanced_summary.text()

    page.advanced_dialog.open()
    app.processEvents()
    assert page.advanced_dialog.isVisible()
    assert page.advanced_dialog.minimumWidth() >= 780
    page.advanced_dialog.close()

    page.mode.setCurrentIndex(1)
    app.processEvents()

    assert not page.devices.isVisible()
    assert not page.scan_button.isVisible()
    assert page.browse_button.isVisible()
    page.close()


def test_status_overview_prefers_health_summary_over_raw_json() -> None:
    snapshot = SimpleNamespace(
        system_state=SystemState.IDLE,
        config=None,
        recognition=None,
        pump_state=None,
        control=None,
        frame=None,
        message="等待配置",
        error="",
        timestamp=0.0,
    )

    overview = StatusPage._overview(snapshot)

    assert set(overview) == {"system", "vision", "camera", "pump", "pid", "alert"}
    assert overview["system"][1] == "等待"
    assert overview["vision"][1] == "等待"
    assert overview["alert"] == "等待配置"
