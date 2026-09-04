from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QListWidgetItem, QSpinBox

from backend.orchestrator.state import SystemState
from frontend.qt_app import (
    CollapsibleSection,
    FrontendApp,
    InitPage,
    MonitorPage,
    ParameterPage,
    PlantCalibrationExperimentDialog,
    PumpPage,
    StatusModule,
    StatusPage,
    VideoPage,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


class _PageApp:
    def __init__(self, config=None):
        self.frontend_config = dict(config or {})
        self.device_verification = {"camera": False, "pump": False, "pump_write": False}
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

    def refresh_navigation(self, _snapshot=None):
        pass

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


def test_monitor_page_owns_plant_calibration_entry() -> None:
    _application()
    page_app=_PageApp({"initial_q1":50.0,"initial_q2":20.0})
    pump_page=PumpPage(page_app)
    monitor_page=MonitorPage(page_app)

    assert not hasattr(pump_page,"plant_calibration_button")
    assert monitor_page.plant_calibration_button.text()=="打开标定窗口"
    assert "视频" in monitor_page.plant_calibration_summary.text()
    pump_page.close(); monitor_page.close()


def test_monitor_navigation_is_available_before_initialization() -> None:
    _application()
    item = QListWidgetItem("运行监控")
    item.setData(Qt.UserRole + 1, "运行监控")
    item.setData(Qt.UserRole + 2, 5)
    holder = SimpleNamespace(
        frontend_config={},
        device_verification={"camera": False, "pump": False, "pump_write": False},
        _nav_items={"monitor": item},
        _nav_collapsed=False,
    )
    holder._navigation_label = lambda key, label, step: FrontendApp._navigation_label(
        holder, key, label, step
    )

    FrontendApp.refresh_navigation(holder)

    assert item.flags() & Qt.ItemIsEnabled
    assert item.flags() & Qt.ItemIsSelectable
    assert item.data(Qt.UserRole + 3) != "locked"


def test_monitor_page_keeps_pid_start_disabled_before_initialization() -> None:
    _application()
    page = MonitorPage(_PageApp())
    snapshot = SimpleNamespace(
        system_state=SystemState.IDLE,
        config=None,
        plant_calibration_experiment={},
        optimization={},
        disturbance_model={},
    )

    page._update_operation_matrix(snapshot)

    assert page.action_buttons["初始化"].isEnabled()
    assert not page.action_buttons["开始"].isEnabled()
    assert not page.pause_button.isEnabled()
    page.close()


def test_pump_write_readback_is_required_and_invalidated_by_flow_change() -> None:
    _application()
    page_app=_PageApp({"initial_q1":50.0,"initial_q2":20.0})
    page=PumpPage(page_app)
    values={"port":"COM3","address":1,"baudrate":1200,"parity":"N","q1":50.0,"q2":20.0}

    assert (page.q1.minimum(),page.q1.maximum()) == (20.0,200.0)
    assert (page.q2.minimum(),page.q2.maximum()) == (5.0,25.0)

    page.pump_done("读取",values,{"recognized_as_pump":True})
    assert page_app.device_verification["pump"]
    assert not page_app.device_verification["pump_write"]

    page.pump_done("写入",values,{"ok":True})
    assert page_app.device_verification["pump_write"]

    page.q1.setValue(51.0)
    assert not page_app.device_verification["pump_write"]
    page.close()


def test_plant_calibration_dialog_tracks_progress_and_locks_configuration() -> None:
    _application()
    holder={
        "snapshot":SimpleNamespace(
            system_state=SystemState.IDLE,
            plant_calibration=None,
            plant_calibration_experiment={"status":"idle","completed_trials":0,"total_trials":0},
            config=SimpleNamespace(initial_q1=50.0,initial_q2=20.0),
            pump_state=SimpleNamespace(q1_actual=50.0,q2_actual=20.0),
        )
    }
    calls=[]
    def prepare():
        calls.append("prepare")
        holder["snapshot"].system_state=SystemState.INITIALIZED
    def run_calibration(config):
        calls.append(("run",config))
        return "result"
    orchestrator=SimpleNamespace(
        get_snapshot=lambda:holder["snapshot"],
        run_plant_calibration_experiment=run_calibration,
    )
    dialog_app=SimpleNamespace(
        frontend_config={"initial_q1":50.0,"initial_q2":20.0},
        device_verification={"camera":True,"pump":True,"pump_write":True},
        orchestrator=orchestrator,
        pages={},
        save=lambda **_values:None,
        task=lambda *_args,**_kwargs:None,
        error=lambda _title,_message:None,
        configure_prepare_initialize=prepare,
    )
    dialog=PlantCalibrationExperimentDialog(dialog_app)
    dialog.resize(760,700)
    dialog.show()
    _application().processEvents()

    assert dialog.start_button.text()=="运行标定"
    assert dialog.response_limit.value()==30
    assert dialog.stop_button.text()=="停止标定"
    assert dialog.close_button.text()=="关闭窗口"
    assert dialog.config_scroll.widgetResizable()
    assert dialog.close_button.mapTo(dialog,dialog.close_button.rect().bottomRight()).y()<=dialog.contentsRect().bottom()
    assert dialog.start_button.isEnabled()
    assert not dialog.pause_button.isEnabled()
    marker=object()
    assert dialog._prepare_and_run(marker)=="result"
    assert calls==["prepare",("run",marker)]
    holder["snapshot"]=SimpleNamespace(
        system_state=SystemState.CALIBRATING,
        plant_calibration=None,
        plant_calibration_experiment={
            "status":"running","phase":"measuring q1-r1-plus",
            "completed_trials":3,"total_trials":12,
        },
        config=SimpleNamespace(initial_q1=50.0,initial_q2=20.0),
        pump_state=SimpleNamespace(q1_actual=52.0,q2_actual=20.0),
    )
    dialog.refresh_status()

    assert dialog.progress.value()==3
    assert dialog.progress.maximum()==12
    assert dialog.pause_button.isEnabled()
    assert dialog.stop_button.isEnabled()
    assert not dialog.start_button.isEnabled()
    assert all(not widget.isEnabled() for widget in dialog._config_widgets)
    holder["snapshot"].system_state=SystemState.STOPPED
    dialog._running=False
    dialog.close()


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


def test_video_page_keeps_visible_width_and_chip_depth_independent() -> None:
    _application()
    page = VideoPage(
        _PageApp(
            {
                "recognition_roi": {
                    "channel_width_um": 60.0,
                    "generation_volume_correction": 1.07,
                }
            }
        )
    )

    assert page.channel_width.value() == 60.0
    page.channel_width.setValue(70.0)
    page.channel_height.setValue(45.0)
    roi = page._roi_payload()

    assert roi["channel_width_um"] == 70.0
    assert roi["generation_channel_height_um"] == 45.0
    assert roi["generation_channel_width_um"] == 70.0
    assert roi["generation_volume_correction"] == 1.07
    assert roi["previous_channel_width_um"] == 60.0
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


def test_initialization_page_is_read_only_confirmation_for_saved_configuration() -> None:
    _application()
    page_app = _PageApp({
        "video_source_type": "camera",
        "video_source": "camera-001",
        "camera_backend": "hikrobot",
        "recognition_roi": {"enabled": True, "channel_calibration_enabled": True, "channel_width_um": 430.0},
        "pump_port": "COM3",
        "pump_address": 1,
        "pump_baudrate": 1200,
        "pump_parity": "N",
        "initial_q1": 50.0,
        "initial_q2": 20.0,
    })
    page = InitPage(page_app)
    page_app.device_verification.update(camera=True, pump=True)
    page.on_show()

    assert page.button.isEnabled()
    assert page.pump_card.property("health") == "ok"
    assert page.flow_card.property("health") == "idle"
    assert page.flow_card.badge.text() == "已保存"
    assert page.back_button.text() == "返回泵机配置"

    page.back_button.click()
    assert page_app.shown == "pump"
    page.close()


def test_initialization_page_does_not_treat_saved_hardware_as_verified() -> None:
    _application()
    page_app = _PageApp({
        "video_source_type": "camera",
        "video_source": "saved-camera-id",
        "camera_backend": "hikrobot",
        "pump_port": "COM3",
        "pump_address": 1,
        "pump_baudrate": 1200,
        "initial_q1": 50.0,
        "initial_q2": 20.0,
    })
    page = InitPage(page_app)
    page.on_show()

    assert not page.button.isEnabled()
    assert page.source_card.property("health") == "warning"
    assert page.source_card.badge.text() == "仅保存配置"
    assert page.pump_card.property("health") == "warning"
    assert page.pump_card.badge.text() == "仅保存配置"
    assert "尚缺" in page.readiness.text()
    page.close()
