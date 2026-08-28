from __future__ import annotations

import cv2
import numpy as np

from backend.vision.channel_region import ChannelRegionDetector, detect_channel_region
from backend.vision.config import ChannelRegionConfig, default_config
from backend.vision.pipeline import VisionPipeline
from backend.orchestrator.vision_adapter import PipelineVisionService


def _channel_frames(count: int = 12) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    random = np.random.default_rng(7)
    channel_polygon = np.array([[0, 55], [639, 75], [639, 240], [0, 220]], dtype=np.int32)
    for _index in range(count):
        image = np.full((300, 640), 180, dtype=np.float32)
        channel_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillConvexPoly(channel_mask, channel_polygon, 255)
        texture = random.normal(0.0, 22.0, image.shape)
        image[channel_mask > 0] += texture[channel_mask > 0]
        frames.append(cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return frames


def test_multiframe_signal_and_straightness_find_and_rectify_channel() -> None:
    result = detect_channel_region(_channel_frames(), ChannelRegionConfig())

    assert result.status == "calibrated"
    assert result.confidence >= 0.58
    assert len(result.wall_lines) == 2
    assert result.high_frequency_map is not None
    assert result.line_overlay is not None
    assert result.region_overlay is not None
    assert result.rectified_frame is not None
    assert 160 <= result.rectified_frame.shape[0] <= 195
    assert result.rectified_frame.shape[1] > 600


def test_broad_high_frequency_channel_wins_over_strong_thin_parallel_distractor() -> None:
    frames: list[np.ndarray] = []
    random = np.random.default_rng(2)
    channel_polygon = np.array([[0, 155], [639, 170], [639, 315], [0, 300]], dtype=np.int32)
    for _index in range(12):
        image = np.full((360, 640), 180, dtype=np.float32)
        cv2.line(image, (3, 25), (636, 35), 20, 4)
        cv2.line(image, (3, 95), (636, 105), 20, 4)
        channel_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillConvexPoly(channel_mask, channel_polygon, 255)
        texture = random.normal(0.0, 25.0, image.shape)
        image[channel_mask > 0] += texture[channel_mask > 0]
        frames.append(cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    result = detect_channel_region(frames, ChannelRegionConfig())
    centers = sorted((line["y1"] + line["y2"]) * 0.5 for line in result.wall_lines)

    assert result.status == "calibrated"
    assert centers[0] > 0.40
    assert centers[1] > 0.80


def test_single_frame_high_frequency_band_is_not_treated_as_persistent_channel() -> None:
    frames = [np.full((300, 640, 3), 180, dtype=np.uint8) for _ in range(12)]
    random = np.random.default_rng(19)
    transient = frames[4].astype(np.float32)
    channel_mask = np.zeros(transient.shape[:2], dtype=np.uint8)
    cv2.rectangle(channel_mask, (0, 70), (639, 230), 255, -1)
    texture = random.normal(0.0, 30.0, transient.shape[:2])
    for channel in range(3):
        transient[:, :, channel][channel_mask > 0] += texture[channel_mask > 0]
    frames[4] = np.clip(transient, 0, 255).astype(np.uint8)

    result = detect_channel_region(frames, ChannelRegionConfig())

    assert result.status == "fallback"
    assert result.wall_lines == []


def test_low_information_frames_fall_back_to_full_frame() -> None:
    frames = [np.full((240, 480, 3), 150, dtype=np.uint8) for _ in range(12)]

    result = detect_channel_region(frames, ChannelRegionConfig())

    assert result.status == "fallback"
    assert result.wall_lines == []
    assert "回退整帧" in result.reason


def test_detector_bounds_startup_storage_for_uhd_frames() -> None:
    config = ChannelRegionConfig(sample_frames=12, work_max_width=960, work_max_height=720)
    detector = ChannelRegionDetector(config)
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    result = detector.add_frame(frame)

    assert result.status == "collecting"
    stored = detector._frames[0]
    assert stored.ndim == 2
    assert stored.shape[1] <= 960
    assert stored.shape[0] <= 720


def test_confidence_is_normalized_when_all_weights_are_maximized() -> None:
    config = ChannelRegionConfig(
        high_frequency_weight=1.0,
        straightness_weight=1.0,
        geometry_weight=1.0,
    )

    result = detect_channel_region(_channel_frames(), config)

    assert 0.0 <= result.confidence <= 1.0


def test_detector_can_be_explicitly_skipped() -> None:
    detector = ChannelRegionDetector(ChannelRegionConfig(enabled=False))

    result = detector.add_frame(_channel_frames(1)[0])

    assert result.status == "skipped"
    assert result.wall_lines == []


def test_pipeline_collects_twelve_frames_then_uses_rectified_region() -> None:
    config = default_config()
    config.channel_region.sample_frames = 12
    pipeline = VisionPipeline(config)
    frames = _channel_frames()

    for frame in frames[:-1]:
        result = pipeline.process_frame(frame)
        assert result.channel_region.status == "collecting"
        assert result.analysis_frame.shape[:2] == frame.shape[:2]
        assert result.detections.centers == []
        assert not result.metrics.control.valid_for_control

    result = pipeline.process_frame(frames[-1])

    assert result.channel_region.status == "calibrated"
    assert result.analysis_frame.shape[0] < frames[-1].shape[0]
    assert result.analysis_frame.shape[1] > 600


def test_channel_region_status_is_exposed_in_recognition_snapshot() -> None:
    service = PipelineVisionService()
    service.set_recognition_roi(
        {"enabled": False, "channel_region_enabled": False}
    )

    snapshot = service._empty_snapshot("waiting")

    assert snapshot.channel_region_status == "skipped"
    assert snapshot.channel_region_confidence == 0.0
    assert "跳过" in snapshot.channel_region_reason


def test_manual_roi_has_priority_over_automatic_calibration() -> None:
    config = default_config()
    config.roi.enabled = True
    config.roi.user_defined = True
    config.roi.x_start_ratio = 0.1
    config.roi.x_end_ratio = 0.9
    config.roi.y_start_ratio = 0.2
    config.roi.y_end_ratio = 0.8
    pipeline = VisionPipeline(config)

    result = pipeline.process_frame(_channel_frames(1)[0])

    assert result.channel_region.status == "manual"
    assert result.analysis_frame.shape[:2] == (180, 512)
