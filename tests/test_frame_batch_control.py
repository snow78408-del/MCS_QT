from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from backend.orchestrator.models import SystemConfig, validate_control_batch
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.orchestrator.vision_adapter import PipelineVisionService
from backend.vision.config import MetricsConfig
from backend.vision.metrics import MetricsCalculator


@pytest.mark.parametrize("capture,analysis", [(4, 5), (5, 6), (65, 5), (10, True), (10.5, 5)])
def test_batch_sizes_reject_invalid_values(capture, analysis):
    with pytest.raises(ValueError):
        validate_control_batch(capture, analysis)


def submit(service, first, last):
    frame = np.zeros((20, 30), dtype=np.uint8)
    for frame_id in range(first, last + 1):
        service._frame_metadata[frame_id] = {
            "capture_monotonic": time.monotonic(), "hardware_frame_id": frame_id,
        }
        service._submit_processing_frame(frame_id, float(frame_id), frame)


def test_collect_all_n_before_enqueue_last_m_and_wait_for_next_request():
    service = PipelineVisionService()
    request = service.request_control_batch(10, 7)
    submit(service, 1, 9)
    assert service._frame_queue.empty()
    submit(service, 10, 10)
    batch = service._frame_queue.get_nowait()
    assert [item[0] for item in batch] == list(range(4, 11))
    assert service._control_batch_assignments[10] == request
    submit(service, 11, 20)
    assert service._frame_queue.empty()
    assert not service._capture_batch


def test_stale_frames_and_hardware_gaps_do_not_complete_a_batch():
    service = PipelineVisionService()
    service.request_control_batch(5, 5)
    frame = np.zeros((20, 30), dtype=np.uint8)
    service._frame_metadata[1] = {"capture_monotonic": service._control_batch_not_before - 1}
    service._submit_processing_frame(1, 1.0, frame)
    assert not service._capture_batch
    submit(service, 2, 4)
    service._frame_metadata[5] = {"capture_monotonic": time.monotonic(), "hardware_frame_id": 50}
    service._submit_processing_frame(5, 5.0, frame)
    assert [item[0] for item in service._capture_batch] == [5]
    assert service._frame_queue.empty()


def test_frame_window_includes_last_frame_and_ignores_seconds():
    metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=10))
    metrics.begin_frame_batch(5)
    for index in range(5):
        result = metrics._update_realtime_window(
            crossed_track_diameters={index: 50 + index}, crossed_track_bead_counts={index: 1},
            frame_crossing_count=1, timestamp=10.0 + index,
            crossed_track_sample_starts={index: 10.0 + index},
            crossed_track_sample_ends={index: 10.0 + index},
        )
        if index < 4:
            assert result[-1] == 0
    assert result == ([50, 51, 52, 53, 54], 5, 5, 5, 1)
    assert metrics._completed_bounds == (10.0, 14.0)
    metrics.begin_frame_batch(5)
    for index in range(5):
        result = metrics._update_realtime_window(
            crossed_track_diameters={7: 70} if index == 4 else {},
            crossed_track_bead_counts={}, timestamp=30.0 + index,
        )
    assert result[0] == [70]
    assert result[-1] == 2


@pytest.mark.parametrize("cancel_at", [None, 2])
def test_analysis_publishes_once_after_all_frames_and_discards_cancelled_batch(cancel_at):
    service = PipelineVisionService()
    request = service.request_control_batch(8, 5)
    submit(service, 1, 8)
    # Metadata can age out of the camera ring while the analysis is pending.
    service._frame_metadata.clear()
    service._droplet_gallery_periods[1] = [{"frame_id": -1}]
    seen = []

    def analyze(frame, *, frame_id, timestamp):
        assert service.get_snapshot().control_batch_id != request
        assert frame_id in service._pinned_batch_metadata
        assert not service._droplet_gallery_periods
        seen.append(frame_id)
        if len(seen) == cancel_at:
            service.cancel_control_batch()
            service._stop_event.set()
        return replace(service._empty_snapshot("test"), frame_id=frame_id)

    service._snapshot_from_frame = analyze
    worker = threading.Thread(target=service._process_loop)
    worker.start()
    try:
        if cancel_at is None:
            result = service.wait_for_control_batch(request, timeout=2.0)
            assert result.control_batch_id == request
            assert result.batch_capture_frames == 8
            assert result.batch_analysis_frames == 5
            assert seen == [4, 5, 6, 7, 8]
        else:
            worker.join(timeout=2.0)
            assert service.get_snapshot().control_batch_id != request
            assert seen == [4, 5]
    finally:
        service._stop_event.set()
        worker.join(timeout=2.0)
    assert not worker.is_alive()


@pytest.mark.parametrize("stop_early", [False, True, "matching"])
def test_orchestrator_waits_for_matching_complete_batch_before_pid(stop_early):
    service = OrchestratorService.__new__(OrchestratorService)
    service._cfg = SimpleNamespace(control_capture_frames=10, control_analysis_frames=7)
    service._lock = threading.RLock()
    service._stop_event = threading.Event()
    service._pause_event = threading.Event()
    service._lifecycle_generation = 3
    service._state = SystemState.RUNNING
    service._safety = SimpleNamespace(heartbeat=lambda *args, **kwargs: True)
    service._log = lambda message: None
    events = []

    def request(capture, analysis):
        events.append(("request", capture, analysis))
        return 12

    def wait(request_id, timeout):
        events.append("analyze")
        if stop_early:
            service._stop_event.set()
            return SimpleNamespace(control_batch_id=12 if stop_early == "matching" else 11)
        return SimpleNamespace(control_batch_id=12 if events.count("analyze") == 7 else 11)

    service.vision_service = SimpleNamespace(request_control_batch=request, wait_for_control_batch=wait)
    service.run_control_step = lambda: events.append("PID/pump")
    service._run_frame_control_cycle(object(), 3, 7.5)
    assert events[0] == ("request", 10, 7)
    if stop_early:
        assert "PID/pump" not in events
    else:
        assert events == [("request", 10, 7)] + ["analyze"] * 7 + ["PID/pump"]


def test_live_settings_replace_config_without_mutating_current_cycle():
    service = OrchestratorService()
    service._cfg = SystemConfig(60, 1, "video", "sample.mp4", 50, 20, 7500)
    current = service._cfg
    service.configure_control_batch(15, 10)
    assert current.control_capture_frames == 5
    assert service._cfg.control_capture_frames == 15
    assert service._cfg.control_analysis_frames == 10
    assert service._cfg.control_batch_enabled


def test_dialog_saves_sizes_and_calls_public_orchestrator_interface():
    from PySide6.QtWidgets import QApplication
    from frontend.qt_app import ControlBatchDialog

    app = QApplication.instance() or QApplication([])
    calls, saved = [], {}
    host = SimpleNamespace(
        frontend_config={"control_capture_frames": 12, "control_analysis_frames": 8},
        orchestrator=SimpleNamespace(configure_control_batch=lambda n, m: calls.append((n, m))),
        save=lambda **values: saved.update(values), error=lambda *args: calls.append("error"),
    )
    dialog = ControlBatchDialog(host)
    assert dialog.capture_frames.value() == 12
    assert dialog.analysis_frames.value() == 8
    dialog.analysis_frames.setValue(13)
    dialog._save()
    assert calls == ["error"]
    assert not saved
    dialog.capture_frames.setValue(15)
    dialog._save()
    assert calls[-1] == (15, 13)
    assert saved == dict(control_batch_enabled=True, control_capture_frames=15, control_analysis_frames=13)
    dialog.close()
    app.processEvents()


def test_frontend_build_restores_saved_batch_settings():
    from frontend.qt_app import FrontendApp

    cfg = dict(target_diameter=60, pixel_to_micron=1, video_source_type="video",
               video_source="sample.mp4", initial_q1=50, initial_q2=20, control_interval_ms=7500,
               control_batch_enabled=True, control_capture_frames=15, control_analysis_frames=10)
    result = FrontendApp.build_system_config(SimpleNamespace(frontend_config=cfg))
    assert result.control_batch_enabled
    assert (result.control_capture_frames, result.control_analysis_frames) == (15, 10)


def test_oversized_capture_batch_is_invalid_instead_of_allocating_more_frames():
    service = PipelineVisionService()
    request = service.request_control_batch(64, 5)
    service._frame_metadata[1] = {"capture_monotonic": time.monotonic()}
    service._submit_processing_frame(1, 1.0, SimpleNamespace(nbytes=8 * 1024 * 1024))
    result = service.wait_for_control_batch(request, timeout=0)
    assert result.control_batch_id == request
    assert not result.valid_for_control
    assert "256 MiB" in result.reason
    assert service._frame_queue.empty()
