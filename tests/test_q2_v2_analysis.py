from __future__ import annotations

from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from tools.analyze_q2_v2 import (
    ROOT, SOURCE_FILES, VARIANTS, analyze, bin_rows, effect_summary, measure_image,
    make_pipeline, select_grid, sha256, timeline, validate_segments, verify_contract,
)


def segments():
    return [dict(label=f"stage_{i}", q1_ul_min=70, q2_ul_min=q2,
                 observation_started_monotonic=start, observation_finished_monotonic=start + 300,
                 planned_duration_s=300) for i, start, q2 in [(1, 1000., 20), (2, 1310., 25)]]


def frame(stamp, identifier):
    return dict(host_monotonic_s=stamp, hardware_frame_id=identifier,
                phase="intentionally_not_used", grid_slot=None)


def test_grid_uses_first_raw_acquisition_in_each_cell_without_interpolation():
    records = [frame(999., 1), frame(1000., 2), frame(1000.19, 3), frame(1000.2, 4),
               frame(1000.61, 5), frame(1300., 6), frame(1310., 7), frame(1609.999, 8)]
    a, b = select_grid(records, segments())
    assert len(a) == len(b) == 1500
    assert [a[i]["hardware_frame_id"] if a[i] else None for i in range(4)] == [2, 4, None, 5]
    assert b[0]["hardware_frame_id"] == 7
    assert b[-1]["hardware_frame_id"] == 8
    assert sum(row is not None for row in a) == 3


@pytest.mark.parametrize("bad", [frame(1000., 2), frame(1001., 1), frame(float("nan"), 2),
                                  frame(1001., 0), frame(1001., 1.5), frame(True, 2)])
def test_grid_rejects_invalid_or_regressed_identity(bad):
    with pytest.raises(ValueError, match="acquisition"):
        select_grid([frame(1000., 1), bad], segments())


def rows():
    return [dict(raw_available=False, frame_median_strip_px=None,
                 physical_state="missing", detection_state="missing") for _ in range(1500)]


def test_bin_has_separate_raw_and_measurement_coverage_and_uses_frame_medians():
    data = rows()
    for i in range(40):
        data[i].update(raw_available=True, physical_state="unreviewed", detection_state="no_complete_candidate")
    for i in range(10):
        data[i].update(frame_median_strip_px=10. if i < 6 else 100., detection_state="complete_candidates")
    result = bin_rows(data)
    assert result[0]["qualified"]
    assert result[0]["median_strip_px"] == 10.
    assert result[1]["median_strip_px"] is None
    data[0]["frame_median_strip_px"] = None
    assert not bin_rows(data)[0]["qualified"]
    data[0]["frame_median_strip_px"] = 10.
    data[39]["raw_available"] = False
    assert not bin_rows(data)[0]["qualified"]


def test_geometry_effect_uses_same_box_statistic_and_common_boxes():
    bins = {name: [[dict(qualified=True, median_strip_px=10. if stage == 0 else value)
                   for _ in range(30)] for stage in range(2)]
            for name, value in zip(VARIANTS, [15., 16., 13.])}
    effect = effect_summary(bins)
    assert effect["delta_strip_px_by_geometry"] == dict(zip(VARIANTS, [5., 6., 3.]))
    assert effect["u_geo_strip_px"] == 2.
    assert effect["eligible_for_engineering_comparison"]
    bins["minus_1_sensor_px"][1][6]["qualified"] = False
    bins["main"][1][6]["median_strip_px"] = 9999.
    effect = effect_summary(bins)
    assert effect["common_post_boxes"] == list(range(7, 18))
    assert effect["delta_strip_px_by_geometry"]["main"] == 5.
    assert not effect["eligible_for_engineering_comparison"]


def test_readback_does_not_fill_stop_restart_or_downtime():
    stages = segments()
    stages[1].update(command_started_monotonic=1300., command_readback_monotonic=1305.)
    result = timeline({"commands": [{"readback_monotonic": 1305.}]}, stages)
    assert result["command_started_monotonic"] == 1300.
    assert result["command_readback_monotonic"] == 1305.
    assert result["downtime_s"] is None
    assert result["stop_confirmed_monotonic"] is None
    assert result["restart_confirmed_monotonic"] is None
    assert result["actual_parameter_writes"] is None


@pytest.fixture
def capture(tmp_path):
    directory = tmp_path / "capture"
    directory.mkdir()
    (directory / "frames").mkdir()
    summary = dict(result_layers=dict(run_completed=True), stop=dict(verified=True), segments=segments())
    (directory / "session_summary.json").write_text(json.dumps(summary))
    geometry = dict(walls=[dict(x1=.1, y1=y, x2=.9, y2=y) for y in (.3, .7)])
    (directory / "registered_geometry.json").write_text(json.dumps(geometry))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(dict(source_hashes={name: sha256(ROOT / name) for name in SOURCE_FILES[:3]},
                                        geometry_sha256=None, camera_settings_sha256=None)))
    return directory, contract


def test_offline_analysis_preserves_gaps_states_ticks_and_never_substitutes(capture, tmp_path):
    directory, contract = capture
    records = []
    for i, stamp in enumerate([1000., 1000.1, 1000.2, 1000.4, 1310.]):
        name = f"frames/{i}.png"
        cv2.imwrite(str(directory / name), np.full((80, 120), 128, dtype=np.uint8))
        record = frame(stamp, i + 1)
        record.update(image=name, raw_shape=[80, 120], hardware_timestamp_ticks=987654321 + i)
        records.append(record)
    (directory / records[0]["image"]).unlink()  # Later same-cell original must not replace it.
    (directory / "frame_manifest.ndjson").write_text("\n".join(json.dumps(r) for r in records))
    review = tmp_path / "review.json"
    review.write_text(json.dumps(dict(frames={"frames/2.png": dict(state="filament"),
                                               "frames/3.png": dict(state="uncertain")})))
    output = tmp_path / "analysis"
    report = analyze(directory, contract, output, review_path=review,
                     pipeline_factory=object, measure=lambda *args: [10., 20., 30.])
    grid = [json.loads(line) for line in (output / "grid_main.ndjson").read_text().splitlines()]
    assert not grid[0]["raw_available"]
    assert grid[0]["raw_record"]["hardware_frame_id"] == 1
    assert grid[1]["raw_record"]["hardware_timestamp_ticks"] == 987654323
    assert grid[1]["candidate_lengths_strip_px"] == [10., 20., 30.]
    assert grid[1]["physical_state"] == "filament"
    assert grid[1]["frame_median_strip_px"] is None
    assert grid[2]["physical_state"] == "uncertain"
    assert grid[2]["frame_median_strip_px"] is None
    assert grid[3]["frame_median_strip_px"] is None
    assert grid[1500]["frame_median_strip_px"] == 20.
    assert report["selected_original_images"] == 3
    assert report["hardware_requested"] is False
    assert report["holdout_passed"] is None
    assert report["measurement_accepted"] is False
    assert report["provenance"]["unbound_fields"] == ["geometry_sha256", "camera_settings_sha256"]
    assert (output / "analysis.json").is_file()
    with pytest.raises(ValueError, match="refuse overwrite"):
        analyze(directory, contract, output)


def test_complete_detector_chain_runs_on_synthetic_mono8_without_hardware():
    walls = [dict(x1=.05, y1=y, x2=.95, y2=y) for y in (.3, .7)]
    pipeline = make_pipeline()
    assert pipeline.detector._phase_refiner is None
    assert measure_image(pipeline, np.full((80, 120), 128, dtype=np.uint8), walls, 1000.) == []


@pytest.mark.parametrize("change", [lambda p: p["stop"].update(verified=False),
                                   lambda p: p["result_layers"].update(run_completed=False),
                                   lambda p: p["segments"][1].update(observation_started_monotonic=1200.),
                                   lambda p: p["segments"][0].update(planned_duration_s=200),
                                   lambda p: p["segments"][0].update(q2_ul_min=18)])
def test_rejects_incomplete_overlapping_or_non_v2_capture(change):
    payload = dict(result_layers=dict(run_completed=True), stop=dict(verified=True), segments=segments())
    change(payload)
    with pytest.raises(ValueError):
        validate_segments(payload)


def test_contract_drift_is_rejected_without_modifying_contract(capture):
    _, contract_path = capture
    contract = json.loads(contract_path.read_text())
    original = deepcopy(contract)
    assert verify_contract(contract)
    assert contract == original
    contract["source_hashes"][SOURCE_FILES[0]] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_contract(contract)


def test_manifest_path_must_stay_inside_capture(capture, tmp_path):
    directory, contract = capture
    record = dict(frame(1000., 1), image="../outside.png", raw_shape=[80, 120])
    (directory / "frame_manifest.ndjson").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="leaves source"):
        analyze(directory, contract, tmp_path / "output", pipeline_factory=object)
