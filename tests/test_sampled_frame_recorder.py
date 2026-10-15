from types import SimpleNamespace
import json

import numpy as np
import pytest

from tools.sampled_frame_recorder import SampledFrameRecorder


def test_sparse_sampling_preserves_real_timestamps_and_never_creates_video(tmp_path):
    walls = [{"x1": 0., "x2": 1., "y1": .1, "y2": .1},
             {"x1": 0., "x2": 1., "y1": .9, "y2": .9}]
    sink = SampledFrameRecorder(directory=tmp_path, interval_s=1., walls=walls)
    for timestamp in (10., 10.1, 10.99, 11., 14.):
        sink.submit(SimpleNamespace(image=np.full((40, 200), 90, np.uint8),
                                    host_monotonic_timestamp=timestamp, hardware_frame_id=1))
    sink.close()
    assert sink.accepted == 3
    assert sink.summary()["max_sample_gap_s"] == 3.
    assert not list(tmp_path.glob("*.mkv"))
    assert len((tmp_path / "samples.ndjson").read_text().splitlines()) == 3
    assert sink.summary()["measurement_accepted"] is False
    stats = sink.summary()["candidate_statistics"]
    assert stats["frames"] == stats["phase_unresolved_frames"] == 3
    assert stats["candidate_frames"] == 0
    records = [json.loads(line) for line in (tmp_path / "samples.ndjson").read_text().splitlines()]
    assert all(r["observation_status"] == "phase_unresolved" for r in records)
    assert all(not r["absence_confirmed"] for r in records)
    assert all("phase_diagnostics" in r for r in records)


def test_sample_interval_rejects_invalid_values(tmp_path):
    for interval in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SampledFrameRecorder(directory=tmp_path, interval_s=interval, walls=[])
