"""Identify a plant response model from saved calibration history.

The tool is read-only with respect to the calibration archive. It loads every
``*.measurements.json`` record (falling back to the matching
``*.measurements.mpc-training.json`` package when the primary record is absent),
builds a trial table from readback flows and droplet-diameter responses, fits a
small set of candidate response models and selects one with session-grouped
cross validation.

Outputs are written to a separate output directory. Fitting history never
authorises closed-loop control: a fresh independent validation run remains the
only path that may mark a calibration usable, so the artifacts carry explicit
validity statements instead of a loadable calibration.

Usage::

    .venv/Scripts/python.exe tools/fit_plant_history_model.py \
        --history D:/MCS_QT_Data/calibrations --output output/history-model
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
NRMSE_FLOOR_UM = 0.25
APP_VALIDATION_MAE_LIMIT_UM = 2.0
APP_VALIDATION_NRMSE_LIMIT = 0.25
POWER_GRID = tuple(float(value) for value in np.arange(-2.0, 2.501, 0.25))
_T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}


class HistoryLoadError(ValueError):
    """A calibration record cannot be used as modelling input."""


@dataclass(frozen=True, slots=True)
class HistoryTrial:
    session_id: str
    trial_id: str
    split: str
    channel: str
    direction: int
    baseline_q1: float
    baseline_q2: float
    actual_q1: float
    actual_q2: float
    commanded_q1: float
    commanded_q2: float
    baseline_diameter_um: float
    steady_diameter_um: float
    diameter_change_um: float
    response_detected: bool
    response_classification: str
    response_observations: int
    baseline_observations: int
    source: str
    payload: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def delta_q1(self) -> float:
        return self.actual_q1 - self.baseline_q1

    @property
    def delta_q2(self) -> float:
        return self.actual_q2 - self.baseline_q2


@dataclass(frozen=True, slots=True)
class HistorySession:
    session_id: str
    source: str
    status: str
    loadable: bool
    config: dict[str, Any]
    trials: tuple[HistoryTrial, ...]
    skipped_trials: int
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SelectionFilters:
    min_response_observations: int = 3
    only_detected: bool = False
    include_sessions: tuple[str, ...] = ()
    exclude_sessions: tuple[str, ...] = ()
    require_flow_change: bool = True


@dataclass(frozen=True, slots=True)
class Dataset:
    sessions: tuple[HistorySession, ...]
    model_trials: tuple[HistoryTrial, ...]
    validation_trials: tuple[HistoryTrial, ...]
    dropped: tuple[tuple[str, int], ...]

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(trial.session_id for trial in self.model_trials))


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _trial_from_payload(
    payload: dict[str, Any], *, session_id: str, source: str, split: str,
) -> HistoryTrial | None:
    required = ("baseline_q1", "baseline_q2", "actual_q1", "actual_q2",
                "baseline_diameter_um", "steady_diameter_um", "diameter_change_um")
    values = {name: _as_float(payload.get(name)) for name in required}
    if any(value is None for value in values.values()):
        return None
    trial_id = str(payload.get("trial_id") or "")
    if not trial_id:
        return None
    channel = str(payload.get("channel") or "")
    if channel not in {"q1", "q2", "combined", "validation"}:
        return None
    direction = _as_float(payload.get("direction"))
    return HistoryTrial(
        session_id=session_id,
        trial_id=trial_id,
        split=split,
        channel=channel,
        direction=-1 if direction is not None and direction < 0 else 1,
        baseline_q1=float(values["baseline_q1"]),
        baseline_q2=float(values["baseline_q2"]),
        actual_q1=float(values["actual_q1"]),
        actual_q2=float(values["actual_q2"]),
        commanded_q1=_as_float(payload.get("commanded_q1")) or float(values["actual_q1"]),
        commanded_q2=_as_float(payload.get("commanded_q2")) or float(values["actual_q2"]),
        baseline_diameter_um=float(values["baseline_diameter_um"]),
        steady_diameter_um=float(values["steady_diameter_um"]),
        diameter_change_um=float(values["diameter_change_um"]),
        response_detected=bool(payload.get("response_detected", False)),
        response_classification=str(payload.get("response_classification") or "unknown"),
        response_observations=len(payload.get("response_observations") or ()),
        baseline_observations=len(payload.get("baseline_observations") or ()),
        source=source,
        payload=dict(payload),
    )

def load_history_session(path: Path) -> HistorySession:
    """Load one saved calibration record; raise HistoryLoadError when unusable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoryLoadError(f"{Path(path).name}: 无法读取或解析 JSON（{error}）") from error
    if not isinstance(payload, dict):
        raise HistoryLoadError(f"{Path(path).name}: 顶层结构不是对象")
    session_id = str(payload.get("session_id")
                     or Path(path).name.replace(".measurements", "").replace(".json", ""))
    source = str(path)
    trials: list[HistoryTrial] = []
    skipped = 0
    if isinstance(payload.get("measurements"), list):
        for item in payload["measurements"]:
            if not isinstance(item, dict):
                skipped += 1
                continue
            split = "validation" if item.get("channel") == "validation" else "model"
            trial = _trial_from_payload(item, session_id=session_id, source=source, split=split)
            if trial is None:
                skipped += 1
            else:
                trials.append(trial)
    elif isinstance(payload.get("training_trials"), list):
        for key, split in (("training_trials", "model"), ("validation_trials", "validation")):
            for item in payload.get(key) or ():
                if not isinstance(item, dict):
                    skipped += 1
                    continue
                trial = _trial_from_payload(item, session_id=session_id, source=source, split=split)
                if trial is None:
                    skipped += 1
                else:
                    trials.append(trial)
    else:
        raise HistoryLoadError(f"{Path(path).name}: 既没有 measurements 也没有 training_trials")
    config = payload.get("config") if isinstance(payload.get("config"), dict) else payload.get("experiment_config")
    return HistorySession(
        session_id=session_id,
        source=source,
        status=str(payload.get("status") or "unknown"),
        loadable=bool(payload.get("loadable_calibration", False)),
        config=dict(config or {}),
        trials=tuple(trials),
        skipped_trials=skipped,
        reason=str(payload.get("reason") or ""),
    )


def discover_history_files(roots: Iterable[Path]) -> list[Path]:
    """Collect measurement files, preferring the primary record over its MPC copy."""
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.append(root)
        elif root.is_dir():
            primary = {path.name.removesuffix(".measurements.json")
                       for path in root.glob("*.measurements.json")
                       if not path.name.endswith(".mpc-training.json")}
            found.extend(sorted(path for path in root.glob("*.measurements.json")
                                if not path.name.endswith(".mpc-training.json")))
            found.extend(sorted(path for path in root.glob("*.measurements.mpc-training.json")
                                if path.name.removesuffix(".measurements.mpc-training.json") not in primary))
        else:
            raise HistoryLoadError(f"历史目录不存在：{root}")
    return sorted(dict.fromkeys(found))


def build_dataset(sessions: Sequence[HistorySession], filters: SelectionFilters) -> Dataset:
    dropped: dict[str, int] = {}
    selected: list[HistorySession] = []
    model_trials: list[HistoryTrial] = []
    validation_trials: list[HistoryTrial] = []
    include = set(filters.include_sessions)
    exclude = set(filters.exclude_sessions)

    def note(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for session in sessions:
        if include and session.session_id not in include:
            note("会话不在 --include-session 中")
            continue
        if session.session_id in exclude:
            note("会话被 --exclude-session 排除")
            continue
        kept: list[HistoryTrial] = []
        for trial in session.trials:
            if filters.only_detected and not trial.response_detected:
                note("未检测到可信响应（--only-detected）")
                continue
            if trial.response_observations < filters.min_response_observations:
                note("响应曲线点数不足")
                continue
            if filters.require_flow_change and abs(trial.delta_q1) + abs(trial.delta_q2) <= 1e-9:
                note("回读流量没有实际改变")
                continue
            kept.append(trial)
        if not kept:
            note("会话没有可用试验")
            continue
        selected.append(replace(session, trials=tuple(kept)))
        for trial in kept:
            (validation_trials if trial.split == "validation" else model_trials).append(trial)
    return Dataset(
        sessions=tuple(selected),
        model_trials=tuple(model_trials),
        validation_trials=tuple(validation_trials),
        dropped=tuple(sorted(dropped.items())),
    )


@dataclass(frozen=True, slots=True)
class FeatureState:
    center: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    powers: tuple[float, float] | None = None


class ResponseModel:
    """One candidate response-model family (linear in its coefficients)."""

    name = "base"
    terms: tuple[str, ...] = ()

    def prepare(self, trials: Sequence[HistoryTrial]) -> FeatureState:
        return FeatureState()

    def design(self, trials: Sequence[HistoryTrial], state: FeatureState) -> np.ndarray:
        raise NotImplementedError

    def predict(
        self, trials: Sequence[HistoryTrial], coefficients: np.ndarray, state: FeatureState,
    ) -> np.ndarray:
        return self.design(trials, state) @ np.asarray(coefficients, dtype=float)


class ConstantModel(ResponseModel):
    name = "constant"
    terms = ("1",)

    def design(self, trials: Sequence[HistoryTrial], state: FeatureState) -> np.ndarray:
        return np.ones((len(trials), 1), dtype=float)


class LinearDeltaModel(ResponseModel):
    name = "linear_delta"
    terms = ("dq1", "dq2")

    def design(self, trials: Sequence[HistoryTrial], state: FeatureState) -> np.ndarray:
        return np.asarray([[trial.delta_q1, trial.delta_q2] for trial in trials], dtype=float)


class QuadraticDeltaModel(ResponseModel):
    """Application response surface: quadratic in centred, scaled readback flow."""

    name = "quadratic_delta"
    terms = ("x1", "x2", "x1^2", "x1*x2", "x2^2")

    def prepare(self, trials: Sequence[HistoryTrial]) -> FeatureState:
        center = (float(median(trial.baseline_q1 for trial in trials)),
                  float(median(trial.baseline_q2 for trial in trials)))
        scale = (max(max(abs(trial.actual_q1 - center[0]) for trial in trials), 1e-9),
                 max(max(abs(trial.actual_q2 - center[1]) for trial in trials), 1e-9))
        return FeatureState(center=center, scale=scale)

    @staticmethod
    def features(state: FeatureState, q1: float, q2: float) -> np.ndarray:
        first = (q1 - state.center[0]) / state.scale[0]
        second = (q2 - state.center[1]) / state.scale[1]
        return np.asarray([first, second, first * first, first * second, second * second], dtype=float)

    def design(self, trials: Sequence[HistoryTrial], state: FeatureState) -> np.ndarray:
        return np.asarray([
            self.features(state, trial.actual_q1, trial.actual_q2)
            - self.features(state, trial.baseline_q1, trial.baseline_q2)
            for trial in trials
        ], dtype=float)


def _log_ratios(trials: Sequence[HistoryTrial]) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray([math.log(max(trial.actual_q1, 1e-9) / max(trial.baseline_q1, 1e-9)) for trial in trials])
    second = np.asarray([math.log(max(trial.actual_q2, 1e-9) / max(trial.baseline_q2, 1e-9)) for trial in trials])
    return first, second


class LogLinearModel(ResponseModel):
    name = "log_linear"
    terms = ("ln(q1/q1b)", "ln(q2/q2b)")

    def design(self, trials: Sequence[HistoryTrial], state: FeatureState) -> np.ndarray:
        first, second = _log_ratios(trials)
        return np.stack([first, second], axis=1)


class LogQuadraticModel(ResponseModel):
    name = "log_quadratic"
    terms = ("z1", "z2", "z1^2", "z1*z2", "z2^2")

    def design(self, trials: Sequence[HistoryTrial], state: FeatureState) -> np.ndarray:
        first, second = _log_ratios(trials)
        return np.stack([first, second, first ** 2, first * second, second ** 2], axis=1)


class PowerLawModel(ResponseModel):
    """Multiplicative power-law family: d = a1*(q1/q1b)^p1 + a2*(q2/q2b)^p2 + c."""

    name = "power_law"
    terms = ("(q1/q1b)^p1-1", "(q2/q2b)^p2-1")

    def __init__(self, powers: tuple[float, float] = (1.0, 1.0)) -> None:
        self.powers = (float(powers[0]), float(powers[1]))

    def prepare(self, trials: Sequence[HistoryTrial]) -> FeatureState:
        return FeatureState(powers=self.powers)

    def design(self, trials: Sequence[HistoryTrial], state: FeatureState) -> np.ndarray:
        powers = state.powers or self.powers
        rows = []
        for trial in trials:
            ratio1 = max(trial.actual_q1, 1e-9) / max(trial.baseline_q1, 1e-9)
            ratio2 = max(trial.actual_q2, 1e-9) / max(trial.baseline_q2, 1e-9)
            rows.append([ratio1 ** powers[0] - 1.0, ratio2 ** powers[1] - 1.0])
        return np.asarray(rows, dtype=float)

def _huber_irls(matrix: np.ndarray, observed: np.ndarray, *, iterations: int = 60) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(matrix, observed, rcond=None)
    for _ in range(iterations):
        residual = observed - matrix @ coefficients
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if scale <= 1e-9:
            break
        cutoff = 1.345 * scale
        magnitude = np.abs(residual)
        weights = np.where(magnitude <= cutoff, 1.0, cutoff / np.maximum(magnitude, 1e-12))
        updated, *_ = np.linalg.lstsq(
            matrix * np.sqrt(weights)[:, None], observed * np.sqrt(weights), rcond=None)
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def _solve(matrix: np.ndarray, observed: np.ndarray, estimator: str) -> np.ndarray:
    if estimator == "huber":
        return _huber_irls(matrix, observed)
    coefficients, *_ = np.linalg.lstsq(matrix, observed, rcond=None)
    return coefficients


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    family: str
    estimator: str = "ols"
    session_offsets: bool = False

    @property
    def label(self) -> str:
        parts = [self.family, self.estimator]
        if self.session_offsets:
            parts.append("session_offsets")
        return "+".join(parts)


@dataclass(frozen=True, slots=True)
class FittedModel:
    spec: CandidateSpec
    state: FeatureState
    coefficients: np.ndarray
    coefficient_names: tuple[str, ...]
    reference_session: str | None
    offset_sessions: tuple[str, ...]

    def predict(self, trials: Sequence[HistoryTrial]) -> np.ndarray:
        model = _family_for(self.spec.family)
        base = model.predict(trials, self.coefficients[:len(model.terms)], self.state)
        if not self.spec.session_offsets or self.reference_session is None:
            return base
        offsets = self.coefficients[len(model.terms):]
        adjustment = np.asarray([
            0.0 if trial.session_id == self.reference_session
            else float(offsets[self.offset_sessions.index(trial.session_id)])
            if trial.session_id in self.offset_sessions else 0.0
            for trial in trials
        ])
        return base + adjustment


def _family_for(name: str) -> ResponseModel:
    if name == "constant":
        return ConstantModel()
    if name == "linear_delta":
        return LinearDeltaModel()
    if name == "quadratic_delta":
        return QuadraticDeltaModel()
    if name == "log_linear":
        return LogLinearModel()
    if name == "log_quadratic":
        return LogQuadraticModel()
    if name == "power_law":
        return PowerLawModel()
    raise ValueError(f"unknown model family {name}")


def _offset_matrix(trials: Sequence[HistoryTrial], sessions: Sequence[str]) -> np.ndarray:
    if not sessions:
        return np.zeros((len(trials), 0))
    return np.asarray([
        [1.0 if trial.session_id == session else 0.0 for session in sessions] for trial in trials
    ])


def fit_candidate(trials: Sequence[HistoryTrial], spec: CandidateSpec) -> FittedModel:
    if not trials:
        raise HistoryLoadError("没有用于拟合的试验")
    model = _family_for(spec.family)
    observed = np.asarray([trial.diameter_change_um for trial in trials], dtype=float)
    if spec.family == "power_law":
        best: tuple[float, FittedModel] | None = None
        for first in POWER_GRID:
            for second in POWER_GRID:
                candidate = _fit_with_state(
                    trials, spec, FeatureState(powers=(first, second)), observed)
                residual = candidate.predict(trials) - observed
                score = float(residual @ residual)
                if best is None or score < best[0]:
                    best = (score, candidate)
        assert best is not None
        return best[1]
    return _fit_with_state(trials, spec, model.prepare(trials), observed)


def _fit_with_state(
    trials: Sequence[HistoryTrial], spec: CandidateSpec, state: FeatureState, observed: np.ndarray,
) -> FittedModel:
    model = _family_for(spec.family)
    matrix = model.design(trials, state)
    names = list(model.terms)
    reference: str | None = None
    sessions: tuple[str, ...] = ()
    if spec.session_offsets:
        ordered = tuple(dict.fromkeys(trial.session_id for trial in trials))
        reference = ordered[0]
        sessions = ordered[1:]
        matrix = np.hstack([matrix, _offset_matrix(trials, sessions)])
        names.extend(f"offset[{session}]" for session in sessions)
    coefficients = _solve(matrix, observed, spec.estimator)
    return FittedModel(
        spec=spec, state=state, coefficients=np.asarray(coefficients, dtype=float),
        coefficient_names=tuple(names), reference_session=reference, offset_sessions=sessions,
    )


def _design_with_offsets(trials: Sequence[HistoryTrial], fitted: FittedModel) -> np.ndarray:
    model = _family_for(fitted.spec.family)
    matrix = model.design(trials, fitted.state)
    if fitted.spec.session_offsets and fitted.reference_session is not None:
        matrix = np.hstack([matrix, _offset_matrix(trials, fitted.offset_sessions)])
    return matrix


def error_metrics(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float).ravel()
    observed = np.asarray(observed, dtype=float).ravel()
    if prediction.size == 0 or prediction.size != observed.size:
        return {"count": 0.0, "mae_um": math.nan, "rmse_um": math.nan, "nrmse": math.nan,
                "nrmse_of_observed_amplitude": math.nan, "bias_um": math.nan, "r2": math.nan}
    residual = prediction - observed
    scale = max(NRMSE_FLOOR_UM, float(np.median(np.abs(prediction))))
    amplitude = max(NRMSE_FLOOR_UM, float(np.median(np.abs(observed))))
    total = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "count": float(observed.size),
        "mae_um": float(np.mean(np.abs(residual))),
        "rmse_um": float(np.sqrt(np.mean(residual ** 2))),
        "nrmse": float(np.sqrt(np.mean(residual ** 2))) / scale,
        "nrmse_of_observed_amplitude": float(np.sqrt(np.mean(residual ** 2))) / amplitude,
        "bias_um": float(np.mean(residual)),
        "r2": 1.0 - float(np.sum(residual ** 2)) / total if total > 1e-12 else math.nan,
    }


def in_sample_metrics(dataset: Dataset, fitted: FittedModel) -> dict[str, float]:
    observed = np.asarray([trial.diameter_change_um for trial in dataset.model_trials], dtype=float)
    return error_metrics(fitted.predict(dataset.model_trials), observed)


def aicc(dataset: Dataset, fitted: FittedModel) -> float:
    observed = np.asarray([trial.diameter_change_um for trial in dataset.model_trials], dtype=float)
    residual = fitted.predict(dataset.model_trials) - observed
    count = max(1, residual.size)
    parameters = fitted.coefficients.size
    rss = max(float(residual @ residual), 1e-12)
    penalty = 2 * parameters * (parameters + 1) / max(1, count - parameters - 1)
    return count * math.log(rss / count) + 2 * parameters + penalty


def cross_validate(dataset: Dataset, spec: CandidateSpec) -> dict[str, Any]:
    """Leave-one-session-out validation with pooled and per-session errors."""
    pooled_prediction: list[float] = []
    pooled_observed: list[float] = []
    centred_prediction: list[float] = []
    per_session: list[dict[str, Any]] = []
    folds = 0
    for held_out in dataset.session_ids:
        train = [trial for trial in dataset.model_trials if trial.session_id != held_out]
        test = [trial for trial in dataset.model_trials if trial.session_id == held_out]
        if not train or not test:
            continue
        try:
            fitted = fit_candidate(train, spec)
        except (HistoryLoadError, np.linalg.LinAlgError, ValueError):
            continue
        folds += 1
        prediction = fitted.predict(test)
        observed = np.asarray([trial.diameter_change_um for trial in test], dtype=float)
        pooled_prediction.extend(prediction.tolist())
        pooled_observed.extend(observed.tolist())
        centred_prediction.extend((prediction - float(np.mean(prediction - observed))).tolist())
        per_session.append({
            "session_id": held_out, "count": int(observed.size), **error_metrics(prediction, observed)})
    return {
        "folds": folds,
        "pooled": error_metrics(np.asarray(pooled_prediction), np.asarray(pooled_observed)),
        "session_centred": error_metrics(np.asarray(centred_prediction), np.asarray(pooled_observed)),
        "per_session": per_session,
    }


def candidate_specs(include_session_offsets: bool = True) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for family in ("constant", "linear_delta", "quadratic_delta", "log_linear", "log_quadratic", "power_law"):
        for estimator in ("ols", "huber"):
            specs.append(CandidateSpec(family=family, estimator=estimator))
            if include_session_offsets and family != "constant":
                specs.append(CandidateSpec(family=family, estimator=estimator, session_offsets=True))
    return specs


def _t_critical(degrees_of_freedom: int) -> float:
    for key in sorted(_T_CRITICAL_95):
        if degrees_of_freedom <= key:
            return _T_CRITICAL_95[key]
    return 1.96


def _cluster_robust_covariance(
    matrix: np.ndarray, residual: np.ndarray, groups: Sequence[str],
) -> np.ndarray | None:
    """Session-clustered sandwich covariance; repeats inside a session are not independent."""
    try:
        inverse = np.linalg.inv(matrix.T @ matrix)
    except np.linalg.LinAlgError:
        return None
    meat = np.zeros((matrix.shape[1], matrix.shape[1]))
    for group in dict.fromkeys(groups):
        mask = np.asarray([item == group for item in groups])
        block = matrix[mask].T @ residual[mask]
        meat += np.outer(block, block)
    return inverse @ meat @ inverse


def coefficient_table(dataset: Dataset, fitted: FittedModel) -> list[dict[str, Any]]:
    matrix = _design_with_offsets(dataset.model_trials, fitted)
    observed = np.asarray([trial.diameter_change_um for trial in dataset.model_trials], dtype=float)
    residual = fitted.predict(dataset.model_trials) - observed
    degrees = max(1, observed.size - matrix.shape[1])
    classical_scale = float(residual @ residual) / degrees
    try:
        classical_covariance = classical_scale * np.linalg.inv(matrix.T @ matrix)
    except np.linalg.LinAlgError:
        classical_covariance = np.full((matrix.shape[1], matrix.shape[1]), math.nan)
    clusters = max(1, len(set(trial.session_id for trial in dataset.model_trials)) - 1)
    critical = _t_critical(clusters)
    robust_covariance = _cluster_robust_covariance(
        matrix, residual, [trial.session_id for trial in dataset.model_trials])
    rows = []
    for index, name in enumerate(fitted.coefficient_names):
        standard_error = math.sqrt(max(0.0, float(classical_covariance[index, index])))
        robust_error = (math.sqrt(max(0.0, float(robust_covariance[index, index])))
                        if robust_covariance is not None else math.nan)
        interval = critical * robust_error if math.isfinite(robust_error) else math.nan
        estimate = float(fitted.coefficients[index])
        rows.append({
            "name": name,
            "estimate": estimate,
            "standard_error": standard_error,
            "clustered_standard_error": robust_error,
            "ci95_low": estimate - interval if math.isfinite(interval) else math.nan,
            "ci95_high": estimate + interval if math.isfinite(interval) else math.nan,
            "sign_identified": bool(math.isfinite(interval) and abs(estimate) > interval),
        })
    return rows


def probe_trial(fitted: FittedModel, q1: float, q2: float, *, q1_step: float = 0.0,
                q2_step: float = 0.0) -> HistoryTrial:
    return HistoryTrial(
        session_id="__probe__", trial_id="__probe__", split="model", channel="combined", direction=1,
        baseline_q1=q1, baseline_q2=q2, actual_q1=q1 + q1_step, actual_q2=q2 + q2_step,
        commanded_q1=q1 + q1_step, commanded_q2=q2 + q2_step,
        baseline_diameter_um=0.0, steady_diameter_um=0.0, diameter_change_um=0.0,
        response_detected=True, response_classification="detected_stable",
        response_observations=0, baseline_observations=0, source="__probe__",
    )


def model_local_gains(dataset: Dataset, fitted: FittedModel, *, step: float = 0.5) -> tuple[float, float]:
    """Local d(diameter)/d(flow) at the median working point, for reporting only."""
    centre1 = float(median(trial.baseline_q1 for trial in dataset.model_trials))
    centre2 = float(median(trial.baseline_q2 for trial in dataset.model_trials))
    gains = []
    for axis in (0, 1):
        plus = probe_trial(fitted, centre1, centre2,
                           q1_step=step if axis == 0 else 0.0,
                           q2_step=step if axis == 1 else 0.0)
        minus = probe_trial(fitted, centre1, centre2,
                            q1_step=-step if axis == 0 else 0.0,
                            q2_step=-step if axis == 1 else 0.0)
        gains.append(float((fitted.predict((plus,))[0] - fitted.predict((minus,))[0]) / (2 * step)))
    return gains[0], gains[1]


def _flow_model_entry(dataset: Dataset, row: dict[str, Any], fitted: FittedModel) -> dict[str, Any]:
    gains = model_local_gains(dataset, fitted)
    reference_diameter = float(median(trial.baseline_diameter_um for trial in dataset.model_trials))
    reference_q = (
        float(median(trial.baseline_q1 for trial in dataset.model_trials)),
        float(median(trial.baseline_q2 for trial in dataset.model_trials)),
    )
    log_sensitivity = tuple(
        gains[index] * reference_q[index] / reference_diameter if reference_diameter > 1e-9 else math.nan
        for index in (0, 1))
    return {
        "label": row["label"],
        "family": row["family"],
        "estimator": row["estimator"],
        "session_offsets": row["session_offsets"],
        "coefficients": [float(value) for value in fitted.coefficients],
        "coefficient_names": list(fitted.coefficient_names),
        "coefficient_table": coefficient_table(dataset, fitted),
        "cv_rmse_um": row["cv_rmse_um"],
        "cv_mae_um": row["cv_mae_um"],
        "local_gain_um_per_ul_min": [gains[0], gains[1]],
        "log_sensitivity": [log_sensitivity[0], log_sensitivity[1]],
        "reference_q": [reference_q[0], reference_q[1]],
        "reference_diameter_um": reference_diameter,
    }


def flow_model_payload(
    dataset: Dataset, ranking: Sequence[dict[str, Any]], fitted_models: dict[str, FittedModel],
) -> dict[str, Any] | None:
    """Keep the interpretable flow models next to the CV winner, with the constant comparison."""
    constant_rmse = next((row["cv_rmse_um"] for row in ranking if row["family"] == "constant"), math.nan)
    entries: dict[str, Any] = {}
    for family in ("linear_delta", "quadratic_delta"):
        rows = [row for row in ranking if row["family"] == family]
        if not rows:
            continue
        best = rows[0]
        entry = _flow_model_entry(dataset, best, fitted_models[best["label"]])
        entry["cv_uplift_vs_constant_um"] = best["cv_rmse_um"] - constant_rmse
        entry["beats_constant"] = bool(math.isfinite(constant_rmse) and best["cv_rmse_um"] < constant_rmse)
        entry["variants"] = [
            {
                "label": row["label"],
                "cv_rmse_um": row["cv_rmse_um"],
                "local_gain_um_per_ul_min": list(
                    model_local_gains(dataset, fitted_models[row["label"]])),
            }
            for row in rows
        ]
        entries[family] = entry
    return entries or None


def signal_test(trials: Sequence[HistoryTrial]) -> dict[str, Any]:
    """Is any flow response identifiable at all, given the recorded scatter?"""
    if len(trials) < 4:
        return {"available": False, "reason": "试验数不足"}
    matrix = np.asarray([[1.0, trial.delta_q1, trial.delta_q2] for trial in trials], dtype=float)
    observed = np.asarray([trial.diameter_change_um for trial in trials], dtype=float)
    coefficients, *_ = np.linalg.lstsq(matrix, observed, rcond=None)
    residual = matrix @ coefficients - observed
    covariance = _cluster_robust_covariance(matrix, residual, [trial.session_id for trial in trials])
    clusters = max(1, len(set(trial.session_id for trial in trials)) - 1)
    critical = _t_critical(clusters)
    rows = []
    for index, name in enumerate(("const", "dq1", "dq2")):
        error = (math.sqrt(max(0.0, float(covariance[index, index])))
                 if covariance is not None else math.nan)
        interval = critical * error if math.isfinite(error) else math.nan
        estimate = float(coefficients[index])
        rows.append({
            "name": name,
            "estimate": estimate,
            "clustered_standard_error": error,
            "ci95_low": estimate - interval if math.isfinite(interval) else math.nan,
            "ci95_high": estimate + interval if math.isfinite(interval) else math.nan,
            "sign_identified": bool(math.isfinite(interval) and abs(estimate) > interval),
        })
    return {
        "available": True,
        "degrees_of_freedom": clusters,
        "coefficients": rows,
        "flow_response_identifiable": all(row["sign_identified"] for row in rows[1:]),
        "residual_std_um": float(np.std(residual, ddof=1)) if residual.size > 1 else math.nan,
        "observed_std_um": float(np.std(observed, ddof=1)) if observed.size > 1 else math.nan,
    }

def session_diagnostics(dataset: Dataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in dataset.sessions:
        trials = [trial for trial in dataset.model_trials if trial.session_id == session.session_id]
        if not trials:
            continue
        repeats: dict[tuple[Any, ...], list[float]] = {}
        for trial in trials:
            key = (trial.channel, trial.direction, round(trial.delta_q1, 3), round(trial.delta_q2, 3))
            repeats.setdefault(key, []).append(trial.diameter_change_um)
        spreads = [float(np.ptp(values)) for values in repeats.values() if len(values) > 1]
        mismatches = []
        for trial in trials:
            changed_q1 = abs(trial.delta_q1) > 1e-6
            changed_q2 = abs(trial.delta_q2) > 1e-6
            expected = {
                "q1": changed_q1 and not changed_q2,
                "q2": changed_q2 and not changed_q1,
                "combined": changed_q1 and changed_q2,
            }.get(trial.channel, True)
            if not expected:
                mismatches.append(trial.trial_id)
        changes = np.asarray([trial.diameter_change_um for trial in trials], dtype=float)
        rows.append({
            "session_id": session.session_id,
            "status": session.status,
            "loadable": session.loadable,
            "trials": len(trials),
            "validation_trials": sum(1 for trial in dataset.validation_trials
                                     if trial.session_id == session.session_id),
            "median_abs_change_um": float(np.median(np.abs(changes))),
            "change_std_um": float(np.std(changes, ddof=1)) if changes.size > 1 else math.nan,
            "median_repeat_spread_um": float(np.median(spreads)) if spreads else math.nan,
            "max_repeat_spread_um": float(np.max(spreads)) if spreads else math.nan,
            "stimulus_label_mismatches": mismatches,
            "q1_solution": sorted({round(trial.delta_q1, 3) for trial in trials}),
            "q2_solution": sorted({round(trial.delta_q2, 3) for trial in trials}),
            "skipped_trials": session.skipped_trials,
        })
    return rows


def fit_dynamics(dataset: Dataset) -> dict[str, Any]:
    """Fit the shared first-order-plus-delay shape with the application fitter."""
    try:
        from backend.pid_control.calibration_experiment import (
            CalibrationIdentificationError, PlantCalibrationMeasurement, PlantCalibrationObservation, _fit_fopdt,
        )
    except Exception as error:  # noqa: BLE001 - the tool must stay usable standalone
        return {"available": False, "reason": f"无法导入应用动态拟合器：{error}"}
    fields = PlantCalibrationObservation.__dataclass_fields__
    measurements = []
    skipped = 0
    for trial in dataset.model_trials:
        payload = dict(trial.payload)
        baseline = tuple(PlantCalibrationObservation(**{
            key: value for key, value in item.items() if key in fields})
            for item in payload.get("baseline_observations") or ())
        response = tuple(PlantCalibrationObservation(**{
            key: value for key, value in item.items() if key in fields})
            for item in payload.get("response_observations") or ())
        try:
            measurements.append(PlantCalibrationMeasurement(
                **{key: value for key, value in payload.items()
                   if key in PlantCalibrationMeasurement.__dataclass_fields__
                   and key not in {"baseline_observations", "response_observations"}},
                baseline_observations=baseline, response_observations=response,
            ))
        except (TypeError, ValueError):
            skipped += 1
    if not measurements:
        return {"available": False, "reason": "没有可重建的测量记录"}
    try:
        dynamics = _fit_fopdt(measurements, nonlinear=True)
    except CalibrationIdentificationError as error:
        return {"available": False, "reason": str(error), "skipped_trials": skipped}
    return {
        "available": True,
        "delay_ms": float(dynamics.delay_ms),
        "time_constant_ms": float(dynamics.time_constant_ms),
        "time_constant_uncertainty_ms": float(dynamics.time_constant_uncertainty_ms),
        "mae_um": float(dynamics.mae_um),
        "nrmse": float(dynamics.nrmse),
        "sample_count": int(dynamics.sample_count),
        "method": dynamics.method,
        "skipped_trials": skipped,
    }


def app_gate_metrics(dataset: Dataset, fitted: FittedModel, dynamics: dict[str, Any]) -> dict[str, Any]:
    trials = dataset.validation_trials
    if not trials:
        return {"available": False, "reason": "历史记录中没有 validation 通道试验"}
    observed = np.asarray([trial.diameter_change_um for trial in trials], dtype=float)
    metrics = error_metrics(fitted.predict(trials), observed)
    dynamic_ok = (not dynamics.get("available", False)) or dynamics.get("nrmse", math.inf) <= APP_VALIDATION_NRMSE_LIMIT
    passes = bool(
        metrics["count"] > 0
        and metrics["mae_um"] <= APP_VALIDATION_MAE_LIMIT_UM
        and metrics["nrmse"] <= APP_VALIDATION_NRMSE_LIMIT
        and dynamic_ok
    )
    return {
        "available": True,
        "mae_limit_um": APP_VALIDATION_MAE_LIMIT_UM,
        "nrmse_limit": APP_VALIDATION_NRMSE_LIMIT,
        "metrics": metrics,
        "passes": passes,
        "note": "这里的 validation 试验来自同一批历史会话，只用于复核；应用要求的独立验证仍需重新采集。",
    }


def _validity(gates: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    reasons = ["模型来自历史记录拟合，没有新的独立验证；不能据此授权闭环控制。"]
    if not gates.get("passes", False):
        reasons.append(
            f"历史 validation 复核未达到应用门槛（MAE≤{APP_VALIDATION_MAE_LIMIT_UM:.1f} μm，"
            f"NRMSE≤{APP_VALIDATION_NRMSE_LIMIT:.2f}）。")
    if signal.get("available") and not signal.get("flow_response_identifiable", False):
        reasons.append("两路流量增益的 95% 置信区间覆盖 0，当前记录无法辨识响应方向。")
    return {
        "control_authorized": False,
        "usable_as_candidate": bool(gates.get("passes", False)),
        "statement": " ".join(reasons),
    }


def _recommendations(dataset: Dataset, diagnostics: Sequence[dict[str, Any]],
                     signal: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    advice: list[str] = []
    if signal.get("available") and not signal.get("flow_response_identifiable", False):
        advice.append("提高激励幅度或重复次数：当前阶跃的响应低于会话间与重复间的漂移量级，"
                      "建议先做一组同刺激重复实验确认响应可重复，再扩大标定范围。")
    spreads = [row["median_repeat_spread_um"] for row in diagnostics
               if row["median_repeat_spread_um"] is not None and math.isfinite(row["median_repeat_spread_um"])]
    if spreads and max(spreads) > 2.0:
        advice.append("同一刺激重复间的尺寸差异超过 2 µm：标定前确认基线稳定（温度、表面活性剂、"
                      "检测 ROI 与生成区状态），并考虑在同一轮内交替执行正负阶跃以抵消慢漂移。")
    mismatched = [row["session_id"] for row in diagnostics if row["stimulus_label_mismatches"]]
    if mismatched:
        advice.append("部分会话的试验标签与实际流量变化不一致（combined 通道只改变了一路流量）："
                      "建模已按回读流量处理，但这些会话不能用于评估通道交叉作用。")
    if not gates.get("passes", False):
        advice.append("在新的独立验证通过之前，只把本模型当作候选，不要写入基础参数页用于闭环控制。")
    return advice

def _app_nonlinear_model(fitted: FittedModel) -> dict[str, Any] | None:
    if fitted.spec.family != "quadratic_delta" or fitted.spec.session_offsets:
        return None
    return {
        "kind": "quadratic_flow_response",
        "center": [float(value) for value in fitted.state.center],
        "scale": [float(value) for value in fitted.state.scale],
        "coefficients": [float(value) for value in fitted.coefficients[:5]],
        "terms": list(QuadraticDeltaModel.terms),
        "output": "diameter_change_um",
        "baseline_handling": "per_trial_feature_difference",
        "source": "history_fit",
    }


def _trial_table(dataset: Dataset) -> list[dict[str, Any]]:
    rows = []
    for trial in (*dataset.model_trials, *dataset.validation_trials):
        rows.append({
            "session_id": trial.session_id,
            "trial_id": trial.trial_id,
            "split": trial.split,
            "channel": trial.channel,
            "direction": trial.direction,
            "delta_q1": trial.delta_q1,
            "delta_q2": trial.delta_q2,
            "baseline_q1": trial.baseline_q1,
            "baseline_q2": trial.baseline_q2,
            "baseline_diameter_um": trial.baseline_diameter_um,
            "steady_diameter_um": trial.steady_diameter_um,
            "diameter_change_um": trial.diameter_change_um,
            "response_detected": trial.response_detected,
            "response_classification": trial.response_classification,
            "response_observations": trial.response_observations,
            "source": trial.source,
        })
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.3f}"


def _render_csv(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    lines = [",".join(fieldnames)]
    for row in rows:
        values = []
        for name in fieldnames:
            value = row.get(name)
            text = "" if value is None else str(value)
            if any(character in text for character in ',"\n'):
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def _render_report(payload: dict[str, Any]) -> str:
    data = payload["data"]
    chosen = payload["selected_model"]
    lines = [
        "# 历史记录拟合报告",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 数据来源：{len(payload['sources'])} 个标定记录文件",
        f"- 建模试验：{data['model_trials']} 个；validation 通道试验：{data['validation_trials']} 个",
        f"- 排除：{'、'.join(f'{reason} {count}' for reason, count in payload['dropped']) or '无'}",
        "",
        "## 数据质量",
        "",
        "| 会话 | 试验 | 中位绝对值 Δd µm | Δd 标准差 µm | 重复极差中位 µm | 最大重复极差 µm | 标签与流量不一致 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["diagnostics"]:
        lines.append(
            f"| {row['session_id'][:28]} | {row['trials']} | {_fmt(row['median_abs_change_um'])} | "
            f"{_fmt(row['change_std_um'])} | {_fmt(row['median_repeat_spread_um'])} | "
            f"{_fmt(row['max_repeat_spread_um'])} | {len(row['stimulus_label_mismatches'])} |")
    signal = payload["signal_test"]
    lines += ["", "## 响应可辨识性", ""]
    if signal.get("available"):
        lines.append(f"- 观测 Δd 标准差 {_fmt(signal['observed_std_um'])} µm，"
                     f"线性模型残差标准差 {_fmt(signal['residual_std_um'])} µm")
        for row in signal["coefficients"]:
            if row["name"] == "const":
                continue
            lines.append(
                f"- {row['name']}：{row['estimate']:+.4f} µm/(µL/min)，会话聚类标准误 "
                f"{_fmt(row['clustered_standard_error'])}，95% 置信区间 "
                f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]，"
                f"{'方向可辨识' if row['sign_identified'] else '方向不可辨识'}")
        lines.append("- 结论：" + ("两路流量增益方向可辨识"
                                   if signal["flow_response_identifiable"]
                                   else "两路流量增益方向不可辨识（置信区间覆盖 0）"))
    else:
        lines.append(f"- 无法计算：{signal.get('reason', '未知原因')}")
    lines += [
        "",
        "## 候选模型比较（按留一会话交叉验证排序）",
        "",
        "| 模型 | 参数 | 拟合 RMSE µm | 交叉验证 RMSE µm | 交叉验证去会话偏置 RMSE µm | AICc |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["selection"]["ranking"][:10]:
        lines.append(
            f"| {row['label']} | {row['parameters']} | {_fmt(row['in_sample_rmse_um'])} | "
            f"{_fmt(row['cv_rmse_um'])} | {_fmt(row['cv_session_centred_rmse_um'])} | {_fmt(row['aicc'])} |")
    lines += [
        "",
        "## 选定模型",
        "",
        f"- 形式：{chosen['family']}（估计器 {chosen['estimator']}"
        f"{'，含会话偏置项' if chosen['session_offsets'] else ''}）",
    ]
    if chosen.get("powers"):
        lines.append(f"- 幂指数：p1={chosen['powers'][0]:.2f}，p2={chosen['powers'][1]:.2f}")
    if chosen.get("center") and chosen["family"] in {"quadratic_delta"}:
        lines.append(f"- 工作点中心：Q1={chosen['center'][0]:.3f}，Q2={chosen['center'][1]:.3f} µL/min；"
                     f"尺度：{chosen['scale'][0]:.3f} / {chosen['scale'][1]:.3f}")
    lines += ["", "| 系数 | 估计 | 标准误 | 会话聚类标准误 | 95% 置信区间 | 方向可辨识 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in payload["coefficient_table"]:
        lines.append(
            f"| {row['name']} | {row['estimate']:+.4f} | {_fmt(row['standard_error'])} | "
            f"{_fmt(row['clustered_standard_error'])} | "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] | "
            f"{'是' if row['sign_identified'] else '否'} |")
    flow_models = payload.get("flow_model")
    if flow_models:
        lines += ["", "## 最拟合的流量模型（参考）", ""]
        for family, title in (("linear_delta", "线性增益模型"), ("quadratic_delta", "二次响应面（应用同形式）")):
            entry = flow_models.get(family)
            if entry is None:
                continue
            lines.append(
                f"- {title}：{entry['label']}；留一会话交叉验证 RMSE={_fmt(entry['cv_rmse_um'])} µm，"
                f"相对常数模型 {'更优' if entry['beats_constant'] else '更差'} "
                f"{_fmt(abs(entry['cv_uplift_vs_constant_um']))} µm")
            lines.append(
                f"  - 工作点局部增益：d(尺寸)/d(Q1)={_fmt(entry['local_gain_um_per_ul_min'][0])}，"
                f"d(尺寸)/d(Q2)={_fmt(entry['local_gain_um_per_ul_min'][1])} µm/(µL/min)")
            lines.append(
                f"  - 对数灵敏度（应用同口径）：Q1={_fmt(entry['log_sensitivity'][0])}，"
                f"Q2={_fmt(entry['log_sensitivity'][1])}；参考工作点 Q1={_fmt(entry['reference_q'][0])}、"
                f"Q2={_fmt(entry['reference_q'][1])} µL/min，参考尺寸 {_fmt(entry['reference_diameter_um'])} µm")
        for family in ("linear_delta", "quadratic_delta"):
            entry = flow_models.get(family)
            if entry is None:
                continue
            alternatives = [item for item in entry["variants"] if item["label"] != entry["label"]][:3]
            if alternatives:
                lines.append("  - 同族其它候选：" + "；".join(
                    f"{item['label']}（CV RMSE={_fmt(item['cv_rmse_um'])} µm，"
                    f"增益 [{_fmt(item['local_gain_um_per_ul_min'][0])}, "
                    f"{_fmt(item['local_gain_um_per_ul_min'][1])}]）" for item in alternatives))
        lines.append("- 这些模型都没有优于常数模型：其系数只能作为上界参考，不能当作已辨识的增益。")
    training = payload["metrics"]["training"]
    lines.append("")
    lines.append(f"- 拟合（同批数据）：MAE={_fmt(training['mae_um'])} µm，RMSE={_fmt(training['rmse_um'])} µm，"
                 f"NRMSE={_fmt(training['nrmse'])}")
    pooled = payload["metrics"]["leave_one_session_out"]["pooled"]
    lines.append(f"- 留一会话交叉验证：MAE={_fmt(pooled['mae_um'])} µm，RMSE={_fmt(pooled['rmse_um'])} µm，"
                 f"NRMSE={_fmt(pooled['nrmse'])}，R²={_fmt(pooled['r2'])}")
    lines.append("- NRMSE 按应用口径以预测幅度归一化；当模型增益接近 0 时该值会急剧放大，此处只用于门槛复核。")
    dynamics = payload["dynamics"]
    if dynamics.get("available"):
        lines.append(f"- 动态拟合：延迟 {dynamics['delay_ms']:.1f} ms，时间常数 "
                     f"{dynamics['time_constant_ms']:.1f} ms，动态 NRMSE={dynamics['nrmse']:.3f}"
                     f"（{dynamics['sample_count']} 个采样点）")
    else:
        lines.append(f"- 动态拟合不可用：{dynamics.get('reason', '未知原因')}")
    lines.append("- 动态参数由池化历史曲线拟合，延迟与时间常数在漂移主导的记录中不可分辨，仅作参考。")
    gates = payload["app_gates"]
    lines += ["", "## 应用门槛复核", ""]
    if gates.get("available"):
        metrics = gates["metrics"]
        lines.append(f"- 历史 validation 试验：MAE={_fmt(metrics['mae_um'])} µm（上限 "
                     f"{gates['mae_limit_um']:.1f}），NRMSE={_fmt(metrics['nrmse'])}（上限 "
                     f"{gates['nrmse_limit']:.2f}）→ {'达到门槛' if gates['passes'] else '未达到门槛'}")
        lines.append(f"- {gates['note']}")
    else:
        lines.append(f"- {gates.get('reason', '无 validation 试验')}")
    lines += ["", "## 适用性结论", "", f"- {payload['validity']['statement']}"]
    if payload["recommendations"]:
        lines += ["", "## 建议", ""]
        lines += [f"- {item}" for item in payload["recommendations"]]
    lines.append("")
    return "\n".join(lines)

def _tick_label(value: float) -> str:
    """Avoid printing negative zero on tick labels."""
    return "0" if abs(value) < 0.5 else f"{value:.0f}"


def write_figure(path: Path, dataset: Dataset, fitted: FittedModel) -> str | None:
    """Render a predicted-vs-observed and stimulus-map figure with OpenCV."""
    try:
        import cv2
    except Exception:  # noqa: BLE001 - figures are optional
        return None
    width, height = 1180, 420
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    sessions = dataset.session_ids
    colours = [(200, 80, 60), (60, 150, 60), (200, 140, 40), (140, 80, 200), (60, 160, 200), (120, 120, 120)]
    observed = np.asarray([trial.diameter_change_um for trial in dataset.model_trials], dtype=float)
    predicted = fitted.predict(dataset.model_trials)
    panels = ((60, 60, 520, 360), (660, 60, 1120, 360))
    for left, top, right, bottom in panels:
        cv2.rectangle(canvas, (left, top), (right, bottom), (190, 190, 190), 1)
    low = float(min(observed.min(), predicted.min())) - 2.0
    high = float(max(observed.max(), predicted.max())) + 2.0
    span = max(1e-6, high - low)
    left, top, right, bottom = panels[0]

    def plot_to_pixel(x_value: float, y_value: float) -> tuple[int, int]:
        return (int(left + (x_value - low) / span * (right - left)),
                int(bottom - (y_value - low) / span * (bottom - top)))

    cv2.line(canvas, plot_to_pixel(low, low), plot_to_pixel(high, high), (210, 210, 210), 1)
    for index, trial in enumerate(dataset.model_trials):
        colour = colours[sessions.index(trial.session_id) % len(colours)]
        cv2.circle(canvas, plot_to_pixel(float(observed[index]), float(predicted[index])), 4, colour, -1)
    cv2.putText(canvas, "predicted [um] vs observed [um]", (left + 8, top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    for fraction in (0.0, 0.5, 1.0):
        value = low + fraction * span
        x_pixel = int(left + fraction * (right - left))
        y_pixel = int(bottom - fraction * (bottom - top))
        cv2.line(canvas, (x_pixel, bottom), (x_pixel, bottom + 5), (150, 150, 150), 1)
        cv2.line(canvas, (left - 5, y_pixel), (left, y_pixel), (150, 150, 150), 1)
        cv2.putText(canvas, _tick_label(value), (x_pixel - 10, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, _tick_label(value), (left - 40, y_pixel + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)

    left, top, right, bottom = panels[1]
    q1_values = [trial.delta_q1 for trial in dataset.model_trials]
    q2_values = [trial.delta_q2 for trial in dataset.model_trials]
    q1_low, q1_high = min(q1_values) - 1.0, max(q1_values) + 1.0
    q2_low, q2_high = min(q2_values) - 1.0, max(q2_values) + 1.0
    q1_span = max(1e-6, q1_high - q1_low)
    q2_span = max(1e-6, q2_high - q2_low)
    for index, trial in enumerate(dataset.model_trials):
        x_pixel = int(left + (trial.delta_q1 - q1_low) / q1_span * (right - left))
        y_pixel = int(bottom - (trial.delta_q2 - q2_low) / q2_span * (bottom - top))
        colour = colours[sessions.index(trial.session_id) % len(colours)]
        radius = int(max(3, min(9, 3 + abs(float(observed[index])))))
        cv2.circle(canvas, (x_pixel, y_pixel), radius, colour, -1)
    cv2.putText(canvas, "stimulus: x=dQ1, y=dQ2, size=|dD|", (left + 8, top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(canvas, (int(left + (0.0 - q1_low) / q1_span * (right - left)), top),
             (int(left + (0.0 - q1_low) / q1_span * (right - left)), bottom), (225, 225, 225), 1)
    cv2.line(canvas, (left, int(bottom - (0.0 - q2_low) / q2_span * (bottom - top))),
             (right, int(bottom - (0.0 - q2_low) / q2_span * (bottom - top))), (225, 225, 225), 1)
    for fraction in (0.0, 0.5, 1.0):
        cv2.putText(canvas, _tick_label(q1_low + fraction * q1_span),
                    (int(left + fraction * (right - left)) - 10, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, _tick_label(q2_low + fraction * q2_span),
                    (left - 40, int(bottom - fraction * (bottom - top)) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    return str(path) if cv2.imwrite(str(path), canvas) else None


def run_identification(
    history_roots: Sequence[Path],
    output_dir: Path,
    *,
    filters: SelectionFilters | None = None,
    include_session_offsets: bool = True,
    figure: bool = True,
) -> dict[str, Any]:
    """Fit every candidate model, select one and write the artifacts."""
    filters = filters or SelectionFilters()
    files = discover_history_files(history_roots)
    if not files:
        raise HistoryLoadError("没有找到任何 *.measurements.json 记录")
    sessions: list[HistorySession] = []
    warnings: list[str] = []
    for path in files:
        try:
            sessions.append(load_history_session(path))
        except HistoryLoadError as error:
            warnings.append(str(error))
    dataset = build_dataset(sessions, filters)
    if not dataset.model_trials:
        raise HistoryLoadError("筛选后没有可用的建模试验")

    ranking: list[dict[str, Any]] = []
    fitted_models: dict[str, FittedModel] = {}
    for spec in candidate_specs(include_session_offsets):
        try:
            fitted = fit_candidate(dataset.model_trials, spec)
        except (HistoryLoadError, np.linalg.LinAlgError, ValueError):
            continue
        fitted_models[spec.label] = fitted
        validation = cross_validate(dataset, spec)
        metrics = in_sample_metrics(dataset, fitted)
        ranking.append({
            "label": spec.label,
            "family": spec.family,
            "estimator": spec.estimator,
            "session_offsets": spec.session_offsets,
            "parameters": int(fitted.coefficients.size),
            "in_sample_rmse_um": metrics["rmse_um"],
            "cv_rmse_um": validation["pooled"]["rmse_um"],
            "cv_session_centred_rmse_um": validation["session_centred"]["rmse_um"],
            "cv_mae_um": validation["pooled"]["mae_um"],
            "aicc": aicc(dataset, fitted),
            "per_session": validation["per_session"],
        })
    if not ranking:
        raise HistoryLoadError("所有候选模型都拟合失败")
    ranking.sort(key=lambda row: (row["cv_rmse_um"], row["cv_session_centred_rmse_um"],
                                  row["in_sample_rmse_um"]))
    best = ranking[0]
    selected_spec = CandidateSpec(family=best["family"], estimator=best["estimator"],
                                  session_offsets=best["session_offsets"])
    fitted = fitted_models[best["label"]]
    signal = signal_test(dataset.model_trials)
    diagnostics = session_diagnostics(dataset)
    dynamics = fit_dynamics(dataset)
    gates = app_gate_metrics(dataset, fitted, dynamics)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "history_response_model",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [str(path) for path in files],
        "warnings": warnings,
        "filters": {
            "min_response_observations": filters.min_response_observations,
            "only_detected": filters.only_detected,
            "include_sessions": list(filters.include_sessions),
            "exclude_sessions": list(filters.exclude_sessions),
            "require_flow_change": filters.require_flow_change,
        },
        "data": {
            "sessions": [session.session_id for session in dataset.sessions],
            "model_trials": len(dataset.model_trials),
            "validation_trials": len(dataset.validation_trials),
        },
        "dropped": list(dataset.dropped),
        "diagnostics": diagnostics,
        "signal_test": signal,
        "selection": {"ranking": ranking, "selected": best["label"]},
        "selected_model": {
            "family": selected_spec.family,
            "estimator": selected_spec.estimator,
            "session_offsets": selected_spec.session_offsets,
            "coefficients": [float(value) for value in fitted.coefficients],
            "coefficient_names": list(fitted.coefficient_names),
            "center": [float(value) for value in fitted.state.center],
            "scale": [float(value) for value in fitted.state.scale],
            "powers": [float(value) for value in fitted.state.powers] if fitted.state.powers else None,
            "reference_session": fitted.reference_session,
            "offset_sessions": list(fitted.offset_sessions),
            "terms": list(_family_for(selected_spec.family).terms),
        },
        "coefficient_table": coefficient_table(dataset, fitted),
        "metrics": {
            "training": in_sample_metrics(dataset, fitted),
            "leave_one_session_out": cross_validate(dataset, selected_spec),
            "validation_trials": (
                error_metrics(
                    fitted.predict(dataset.validation_trials),
                    np.asarray([trial.diameter_change_um for trial in dataset.validation_trials], dtype=float))
                if dataset.validation_trials else None),
        },
        "app_gates": gates,
        "dynamics": dynamics,
        "trials": _trial_table(dataset),
    }
    flow_models = flow_model_payload(dataset, ranking, fitted_models)
    payload["flow_model"] = flow_models
    payload["app_nonlinear_model"] = _app_nonlinear_model(fitted)
    payload["app_nonlinear_model_candidate"] = None
    if payload["app_nonlinear_model"] is None and flow_models and "quadratic_delta" in flow_models:
        quadratic = flow_models["quadratic_delta"]
        quadratic_row = next(row for row in ranking if row["label"] == quadratic["label"])
        payload["app_nonlinear_model_candidate"] = _app_nonlinear_model(fitted_models[quadratic_row["label"]])
    payload["validity"] = _validity(gates, signal)
    payload["recommendations"] = _recommendations(dataset, diagnostics, signal, gates)

    output_dir = Path(output_dir)
    for root in (Path(item) for item in history_roots):
        if root.is_dir() and output_dir.resolve() == root.resolve():
            raise HistoryLoadError("输出目录不能与历史记录目录相同")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset.csv").write_text(_render_csv(payload["trials"]), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report(payload), encoding="utf-8")
    if figure:
        payload["figure"] = write_figure(output_dir / "response_map.png", dataset, fitted)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a plant response model from saved calibration history (read-only).")
    parser.add_argument("--history", action="append", default=None,
                        help="calibration history directory or file (repeatable); "
                             "defaults to <user data dir>/calibrations")
    parser.add_argument("--output", default=None,
                        help="output directory (default: output/history-model-<UTC timestamp>)")
    parser.add_argument("--only-detected", action="store_true",
                        help="keep only trials classified as detected_stable")
    parser.add_argument("--min-response-observations", type=int, default=3,
                        help="minimum response-curve samples per trial (default: 3)")
    parser.add_argument("--include-session", action="append", default=[],
                        help="only use these session ids (repeatable)")
    parser.add_argument("--exclude-session", action="append", default=[],
                        help="drop these session ids (repeatable)")
    parser.add_argument("--allow-zero-flow-trials", action="store_true",
                        help="keep trials whose readback flow did not change")
    parser.add_argument("--no-session-offsets", action="store_true",
                        help="disable candidates with per-session intercepts")
    parser.add_argument("--no-figure", action="store_true", help="skip the PNG diagnostic figure")
    parser.add_argument("--quiet", action="store_true", help="print only the output directory")
    return parser


def _configure_stdout() -> None:
    """Keep console output usable on hosts whose console code page is not UTF-8."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdout()
    args = _build_parser().parse_args(argv)
    if args.history:
        roots = [Path(item) for item in args.history]
    else:
        from backend.runtime_paths import user_data_dir
        roots = [user_data_dir() / "calibrations"]
    output = Path(args.output) if args.output else (
        REPO_ROOT / "output" / f"history-model-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    filters = SelectionFilters(
        min_response_observations=max(0, args.min_response_observations),
        only_detected=bool(args.only_detected),
        include_sessions=tuple(args.include_session),
        exclude_sessions=tuple(args.exclude_session),
        require_flow_change=not args.allow_zero_flow_trials,
    )
    try:
        payload = run_identification(
            roots, output, filters=filters,
            include_session_offsets=not args.no_session_offsets,
            figure=not args.no_figure)
    except HistoryLoadError as error:
        print(f"拟合失败：{error}", file=sys.stderr)
        return 2
    if args.quiet:
        print(str(output))
        return 0
    best = payload["selection"]["ranking"][0]
    print(f"建模试验 {payload['data']['model_trials']} 个，"
          f"validation 试验 {payload['data']['validation_trials']} 个")
    print(f"选定模型：{best['label']}（留一会话交叉验证 RMSE={best['cv_rmse_um']:.3f} um）")
    print(f"响应方向可辨识：{payload['signal_test'].get('flow_response_identifiable')}")
    print(f"应用门槛复核通过：{payload['app_gates'].get('passes')}")
    print(f"输出目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
