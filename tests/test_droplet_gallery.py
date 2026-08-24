import base64
from types import SimpleNamespace

import cv2
import numpy as np

from backend.orchestrator.vision_adapter import PipelineVisionService
from backend.vision.tracker import DropletTrack


def _result(period_id, valid_track_ids, crossed_track_ids, track):
    control = SimpleNamespace(
        period_id=period_id,
        valid_track_ids=list(valid_track_ids),
        crossed_track_ids=list(crossed_track_ids),
    )
    return SimpleNamespace(
        frame_index=period_id + 1,
        timestamp=float(period_id + 1),
        metrics=SimpleNamespace(control=control),
        tracking=SimpleNamespace(active_tracks=[track]),
        analysis_frame=np.full((120, 180, 3), 90, dtype=np.uint8),
    )


def test_gallery_publishes_crossing_evidence_when_period_completes():
    service = PipelineVisionService()
    service._pixel_to_micron = 2.0
    track = DropletTrack(
        id=7,
        position=np.array([80.0, 60.0]),
        radius=10.0,
        age=5,
        metadata={"observed_radius": 12.0},
    )

    # While period 1 is in progress, the public gallery still represents the
    # last completed period. The next period-id transition publishes it.
    service._update_droplet_gallery(
        _result(0, [7], [7], track),
        frame_id=101,
        timestamp=10.0,
    )
    assert service.get_last_control_period_droplets()["droplet_count"] == 0

    service._update_droplet_gallery(
        _result(1, [], [], track),
        frame_id=102,
        timestamp=11.0,
    )
    gallery = service.get_last_control_period_droplets()

    assert gallery["period_id"] == 1
    assert gallery["droplet_count"] == 1
    assert gallery["sample_frame_count"] == 1
    assert gallery["droplets"] == []
    frame = gallery["frames"][0]
    assert frame["frame_id"] == 101
    assert frame["valid_droplet_count"] == 1
    assert frame["crossed_droplet_count"] == 1
    assert frame["valid_track_ids"] == [7]
    assert frame["average_diameter_um"] == 48.0
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(frame["image_jpeg_base64"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape[:2] == (120, 180)
