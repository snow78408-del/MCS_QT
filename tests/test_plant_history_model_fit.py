from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import uuid

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import fit_plant_history_model as tool  # noqa: E402


@pytest.fixture()
def tmp_path():
    """Local temp directory.

    The shared pytest temp root is created with owner-only permissions, which some
    sandboxes refuse to scan or write into; use a regular directory instead.
    """
    directory = Path(tempfile.gettempdir()) / f"mcs-history-model-{uuid.uuid4().hex[:12]}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _observations(count: int, *, start: float, step: float, level: float) -> list[dict]:
    return [
        {
            "frame_id": index,
            "capture_monotonic": start + index * step,
            "observed_monotonic": start + index * step,
            "diameter_um": level,
            "droplet_count": 1,
            "droplet_id": index,
        }
        for index in range(count)
    ]


def _trial(
    trial_id: str,
    channel: str,
    direction: int,
    *,
    baseline_flow: tuple[float, float],
    actual_flow: tuple[float, float],
    change: float,
    baseline_diameter: float = 80.0,
    detected: bool = True,
    response_points: int = 12,
    start: float = 100.0,
) -> dict:
    payload = {
        "trial_id": trial_id,
        "channel": channel,
        "direction": direction,
        "baseline_q1": baseline_flow[0],
        "baseline_q2": baseline_flow[1],
        "commanded_q1": actual_flow[0],
        "commanded_q2": actual_flow[1],
        "actual_q1": actual_flow[0],
        "actual_q2": actual_flow[1],
        "actuator_step": None,
        "command_started_monotonic": start,
        "readback_completed_monotonic": start + 1.0,
        "response_started_monotonic": start + 2.0,
        "response_stable_monotonic": start + 5.0,
        "baseline_diameter_um": baseline_diameter,
        "steady_diameter_um": baseline_diameter + change,
        "diameter_change_um": change,
        "response_delay_ms": 900.0,
        "response_detected": detected,
        "response_classification": "detected_stable" if detected else "below_detection_threshold",
        "baseline_observations": _observations(8, start=start - 8.0, step=1.0, level=baseline_diameter),
        "response_observations": _observations(
            response_points, start=start + 2.0, step=1.0, level=baseline_diameter + change),
    }
    return payload


def _session_payload(session_id: str, trials: list[dict]) -> dict:
    return {
        "status": "incomplete",
        "loadable_calibration": False,
        "reason": "test fixture",
        "session_id": session_id,
        "config": {"plant_id": "rig", "chip_id": "chip", "fluid_id": "fluid",
                   "q1_step": 5.0, "q2_step": 5.0},
        "measurements": trials,
    }


def _write_session(directory: Path, session_id: str, trials: list[dict]) -> Path:
    path = directory / f"{session_id}.measurements.json"
    path.write_text(json.dumps(_session_payload(session_id, trials), ensure_ascii=False), encoding="utf-8")
    return path


def _linear_session(session_id: str, index: int, *, gain: tuple[float, float], noise: float = 0.0,
                    seed: int = 0) -> tuple[Path, list[dict]]:
    import random

    rng = random.Random(seed)
    trials = []
    stimuli = [(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0), (5.0, 5.0), (-5.0, -5.0)]
    base = (70.0, 20.0)
    for repeat in range(2):
        for position, (step1, step2) in enumerate(stimuli):
            channel = "combined" if step1 and step2 else ("q1" if step1 else "q2")
            change = gain[0] * step1 + gain[1] * step2 + (rng.gauss(0.0, noise) if noise else 0.0)
            trials.append(_trial(
                f"{channel}-r{repeat}-{position}",
                channel,
                1 if (step1 + step2) >= 0 else -1,
                baseline_flow=base,
                actual_flow=(base[0] + step1, base[1] + step2),
                change=change,
                start=100.0 + index * 200.0 + position * 20.0))
    for direction, sign in (("plus", 1.0), ("minus", -1.0)):
        step1, step2 = 3.0 * sign, 3.0 * sign
        change = gain[0] * step1 + gain[1] * step2 + (rng.gauss(0.0, noise) if noise else 0.0)
        trials.append(_trial(
            f"validation-r1-{direction}",
            "validation",
            1 if sign > 0 else -1,
            baseline_flow=base,
            actual_flow=(base[0] + step1, base[1] + step2),
            change=change,
            start=100.0 + index * 200.0 + 180.0))
    path = _write_session(Path(_linear_session.directory), session_id, trials)
    return path, trials


def _make_linear_history(directory: Path, sessions: int, *, gain, noise: float) -> list[Path]:
    _linear_session.directory = directory
    paths = []
    for index in range(sessions):
        path, _ = _linear_session(f"session-{index}", index, gain=gain, noise=noise, seed=index)
        paths.append(path)
    return paths


def test_recovers_linear_gains_from_synthetic_history(tmp_path):
    _make_linear_history(tmp_path, 4, gain=(0.5, -0.3), noise=0.05)
    payload = tool.run_identification([tmp_path], tmp_path / "out", figure=False)
    linear = payload["flow_model"]["linear_delta"]
    gains = linear["local_gain_um_per_ul_min"]
    assert abs(gains[0] - 0.5) < 0.12
    assert abs(gains[1] + 0.3) < 0.12
    assert payload["signal_test"]["flow_response_identifiable"] is True
    assert payload["app_gates"]["passes"] is True
    assert linear["beats_constant"] is True


def test_no_signal_history_is_not_declared_identifiable(tmp_path):
    _make_linear_history(tmp_path, 4, gain=(0.0, 0.0), noise=4.0)
    payload = tool.run_identification([tmp_path], tmp_path / "out", figure=False)
    assert payload["signal_test"]["flow_response_identifiable"] is False
    flow = payload["flow_model"]["linear_delta"]
    assert abs(flow["cv_uplift_vs_constant_um"]) < 1.0
    assert payload["app_gates"]["available"] is True
    assert payload["app_gates"]["passes"] is False
    assert payload["validity"]["control_authorized"] is False
    assert "控制" in payload["validity"]["statement"]
    assert payload["recommendations"]


def test_loader_accepts_mpc_training_package(tmp_path):
    trials = [
        _trial("q1-r1", "q1", 1, baseline_flow=(70.0, 20.0), actual_flow=(75.0, 20.0), change=1.0),
        _trial("validation-r1", "validation", 1, baseline_flow=(70.0, 20.0),
               actual_flow=(73.0, 21.0), change=0.4),
    ]
    path = tmp_path / "cal.measurements.mpc-training.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "validated_for_mpc": False,
        "training_trials": [trials[0]],
        "validation_trials": [trials[1]],
        "experiment_config": {"plant_id": "rig"},
    }, ensure_ascii=False), encoding="utf-8")
    session = tool.load_history_session(path)
    assert [trial.split for trial in session.trials] == ["model", "validation"]
    assert session.config["plant_id"] == "rig"


def test_loader_skips_trials_without_readback_flow(tmp_path):
    good = _trial("q1-r1", "q1", 1, baseline_flow=(70.0, 20.0), actual_flow=(75.0, 20.0), change=1.0)
    broken = _trial("q1-r2", "q1", 1, baseline_flow=(70.0, 20.0), actual_flow=(75.0, 20.0), change=1.0)
    broken["actual_q1"] = None
    path = _write_session(tmp_path, "session-a", [good, broken])
    session = tool.load_history_session(path)
    assert len(session.trials) == 1
    assert session.skipped_trials == 1


def test_filters_drop_short_curves_and_zero_flow_trials(tmp_path):
    kept = _trial("q1-r1", "q1", 1, baseline_flow=(70.0, 20.0), actual_flow=(75.0, 20.0), change=1.0)
    short = _trial("q1-r2", "q1", 1, baseline_flow=(70.0, 20.0), actual_flow=(75.0, 20.0),
                   change=1.0, response_points=2)
    idle = _trial("q2-r1", "q2", 1, baseline_flow=(70.0, 20.0), actual_flow=(70.0, 20.0), change=0.5)
    path = _write_session(tmp_path, "session-a", [kept, short, idle])
    sessions = [tool.load_history_session(path)]
    dataset = tool.build_dataset(sessions, tool.SelectionFilters(min_response_observations=3))
    assert [trial.trial_id for trial in dataset.model_trials] == ["q1-r1"]
    assert dict(dataset.dropped)["响应曲线点数不足"] == 1
    assert dict(dataset.dropped)["回读流量没有实际改变"] == 1


def test_session_filters_keep_only_requested_session(tmp_path):
    _make_linear_history(tmp_path, 3, gain=(0.4, 0.0), noise=0.05)
    sessions = [tool.load_history_session(path)
                for path in tool.discover_history_files([tmp_path])]
    dataset = tool.build_dataset(sessions, tool.SelectionFilters(include_sessions=("session-1",)))
    assert dataset.session_ids == ("session-1",)


def test_run_identification_writes_artifacts_and_report(tmp_path):
    _make_linear_history(tmp_path, 4, gain=(0.0, 0.0), noise=4.0)
    output = tmp_path / "out"
    payload = tool.run_identification([tmp_path], output, figure=False)
    assert (output / "model.json").exists()
    assert (output / "report.md").exists()
    assert (output / "dataset.csv").exists()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "历史记录拟合报告" in report
    assert "不能据此授权闭环控制" in report
    stored = json.loads((output / "model.json").read_text(encoding="utf-8"))
    assert stored["kind"] == "history_response_model"
    assert stored["schema_version"] == tool.SCHEMA_VERSION
    assert stored["validity"]["control_authorized"] is False
    assert payload["app_gates"]["available"] is True
    assert payload["app_gates"]["passes"] is False


def test_output_directory_must_differ_from_history(tmp_path):
    _make_linear_history(tmp_path, 2, gain=(0.4, 0.0), noise=0.05)
    with pytest.raises(tool.HistoryLoadError):
        tool.run_identification([tmp_path], tmp_path, figure=False)


def test_huber_estimator_resists_a_single_gross_outlier(tmp_path):
    directory = tmp_path
    stimuli = [(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)]
    base = (70.0, 20.0)
    clean = []
    for index, (step1, step2) in enumerate(stimuli):
        clean.append(_trial(f"t{index}", "combined", 1, baseline_flow=base,
                            actual_flow=(base[0] + step1, base[1] + step2),
                            change=0.5 * step1 - 0.25 * step2, start=100.0 + index * 20.0))
    outlier = _trial("outlier", "combined", 1, baseline_flow=base, actual_flow=(75.0, 25.0),
                     change=40.0, start=300.0)
    sessions = [tool.load_history_session(_write_session(directory, "session-a", clean + [outlier]))]
    dataset = tool.build_dataset(sessions, tool.SelectionFilters())
    robust = tool.fit_candidate(dataset.model_trials, tool.CandidateSpec("linear_delta", "huber"))
    ordinary = tool.fit_candidate(dataset.model_trials, tool.CandidateSpec("linear_delta", "ols"))
    assert abs(float(robust.coefficients[0]) - 0.5) < abs(float(ordinary.coefficients[0]) - 0.5)


def test_power_law_grid_recovers_known_exponent(tmp_path):
    base = (70.0, 20.0)
    trials = []
    for index, step in enumerate((-8.0, -4.0, 4.0, 8.0, 0.0)):
        ratio = (base[0] + step) / base[0]
        trials.append(_trial(f"t{index}", "combined", 1, baseline_flow=base,
                             actual_flow=(base[0] + step, base[1]),
                             change=6.0 * (ratio ** 1.5 - 1.0), start=100.0 + index * 20.0))
    sessions = [tool.load_history_session(_write_session(tmp_path, "session-a", trials))]
    dataset = tool.build_dataset(sessions, tool.SelectionFilters())
    fitted = tool.fit_candidate(dataset.model_trials, tool.CandidateSpec("power_law", "ols"))
    assert fitted.state.powers is not None
    assert abs(fitted.state.powers[0] - 1.5) <= 0.25
    assert abs(float(fitted.coefficients[0]) - 6.0) < 0.6


def test_quadratic_history_is_emitted_in_application_form(tmp_path):
    base = (70.0, 20.0)
    trials = []
    for repeat in range(2):
        for index, (step1, step2) in enumerate(((5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0),
                                                (5.0, 5.0), (-5.0, -5.0))):
            change = 0.4 * step1 - 0.2 * step2 + 0.05 * step1 * step2 + 0.02 * step1 ** 2
            trials.append(_trial(f"t{repeat}-{index}", "combined", 1, baseline_flow=base,
                                 actual_flow=(base[0] + step1, base[1] + step2),
                                 change=change, start=100.0 + repeat * 100.0 + index * 10.0))
    sessions = [tool.load_history_session(_write_session(tmp_path, "session-a", trials))]
    dataset = tool.build_dataset(sessions, tool.SelectionFilters())
    quadratic = next(spec for spec in tool.candidate_specs(False)
                     if spec.family == "quadratic_delta" and spec.estimator == "ols")
    fitted = tool.fit_candidate(dataset.model_trials, quadratic)
    block = tool._app_nonlinear_model(fitted)
    assert block is not None
    assert block["kind"] == "quadratic_flow_response"
    assert len(block["coefficients"]) == 5
    assert math.isclose(block["center"][0], base[0], rel_tol=0, abs_tol=1e-6)
