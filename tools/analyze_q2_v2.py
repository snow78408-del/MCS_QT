"""Offline v2 PNG sampling and geometry diagnostics; never opens hardware.

Results remain candidate pixel measurements pending independent image review.
No model fitting, holdout acceptance or physical calibration occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Callable, Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.vision.config import default_config
from backend.vision.pipeline import VisionPipeline
from backend.vision.rectified_roi import wall_line_quad

ROOT = Path(__file__).resolve().parents[1]
GRID_HZ = 5
DURATION_S = 300
VARIANTS = {"main": 0, "minus_1_sensor_px": -1, "plus_1_sensor_px": 1}
REVIEW_STATES = {"complete_droplet", "gap", "filament", "bubble", "uncertain"}
SOURCE_FILES = (
    "backend/vision/capsule_profile.py", "backend/vision/detector.py",
    "backend/vision/pipeline.py", "backend/vision/config.py",
    "backend/vision/rectified_roi.py", "backend/vision/rectified_measurement.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
        if digest is not None:
            return digest.hexdigest()
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
        return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def contained_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if path == directory.resolve() or not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Path leaves source directory: {relative}")
    return path


def verify_contract(contract: dict) -> dict:
    expected = contract.get("source_hashes", {})
    required = {name.replace("/", "\\") for name in SOURCE_FILES[:3]}
    normalized = {name.replace("/", "\\"): digest for name, digest in expected.items()}
    if not required.issubset(normalized):
        raise ValueError("Measurement contract lacks the three frozen vision sources")
    for name, digest in expected.items():
        path = contained_path(ROOT, name.replace("\\", "/"))
        if sha256(path) != digest:
            raise ValueError(f"Measurement source hash mismatch: {name}")
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def validate_segments(summary: dict) -> list[dict]:
    if summary.get("result_layers", {}).get("run_completed") is not True:
        raise ValueError("Completed two-stage capture required")
    if (summary.get("stop") or {}).get("verified") is not True:
        raise ValueError("Verified pump stop required in the saved summary")
    segments = summary.get("segments", [])
    if len(segments) != 2 or len({s["label"] for s in segments}) != 2:
        raise ValueError("Exactly two distinct stage labels required")
    for segment in segments:
        if segment.get("q1_ul_min") != 70 or segment.get("q2_ul_min") not in (20, 25):
            raise ValueError("Only the v2 A/B flow states are supported")
        start = segment["observation_started_monotonic"]
        end = segment["observation_finished_monotonic"]
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or start <= 0 or end - start < DURATION_S - 1e-6
                or segment.get("planned_duration_s") != DURATION_S):
            raise ValueError("Each stage needs a completed 300 s observation")
    if segments[1]["observation_started_monotonic"] < segments[0]["observation_finished_monotonic"]:
        raise ValueError("Stage intervals overlap")
    return segments


def select_grid(records: Iterable[dict], segments: list[dict]) -> list[list[dict | None]]:
    """First *recorded* frame in each half-open 0.2 s cell; no interpolation.

    Recorder phase/grid_slot metadata never defines the analysis time grid.
    Strictly ordered hardware identity and acquisition time are prerequisites.
    """
    grids: list[list[dict | None]] = [[None] * (GRID_HZ * DURATION_S) for _ in segments]
    last_stamp, last_id = -math.inf, -1
    for row in records:
        stamp, identifier = row["host_monotonic_s"], row["hardware_frame_id"]
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) or stamp <= 0 or stamp <= last_stamp
                or isinstance(identifier, bool) or not isinstance(identifier, int)
                or identifier <= 0 or identifier <= last_id):
            raise ValueError("Invalid or regressed acquisition time / hardware frame id")
        last_stamp, last_id = stamp, identifier
        for index, segment in enumerate(segments):
            elapsed = stamp - segment["observation_started_monotonic"]
            if 0 <= elapsed < DURATION_S:
                # Suppress only floating-point cancellation at exact cell edges.
                slot = int(math.floor(elapsed * GRID_HZ + 1e-7))
                if slot < len(grids[index]) and grids[index][slot] is None:
                    grids[index][slot] = row
                break
    return grids


def bin_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != GRID_HZ * DURATION_S:
        raise ValueError("Expected all 1500 time cells, including missing cells")
    bins = []
    for index in range(DURATION_S // 10):
        group = rows[index * 50:(index + 1) * 50]
        saved = sum(row["raw_available"] for row in group)
        lengths = [row["frame_median_strip_px"] for row in group
                   if row["frame_median_strip_px"] is not None]
        qualified = saved >= 40 and len(lengths) >= 10
        bins.append(dict(start_s=index * 10, end_s=(index + 1) * 10,
                         time_center_s=index * 10 + 5, saved_cells=saved,
                         measurable_cells=len(lengths), qualified=qualified,
                         median_strip_px=float(np.median(lengths)) if qualified else None,
                         physical_states=dict(Counter(row["physical_state"] for row in group)),
                         detection_states=dict(Counter(row["detection_state"] for row in group))))
    return bins


def effect_summary(bins: dict[str, list[list[dict]]]) -> dict:
    # All variants use identical qualified box indices for the effect statistic.
    # Incomplete windows remain descriptive and cannot pass the resolution check.
    windows = [(0, list(range(24, 30))), (1, list(range(6, 18)))]
    common = [[i for i in indices if all(bins[name][stage][i]["qualified"] for name in VARIANTS)]
              for stage, indices in windows]
    deltas = {}
    for name in VARIANTS:
        centers = [float(np.median([bins[name][stage][i]["median_strip_px"] for i in indices]))
                   if indices else None for (stage, _), indices in zip(windows, common)]
        deltas[name] = centers[1] - centers[0] if None not in centers else None
    complete = [len(indices) for indices in common] == [6, 12]
    main = deltas["main"]
    sensitivity = max(abs(delta - main) for delta in deltas.values()) if main is not None else None
    return dict(delta_strip_px_by_geometry=deltas, u_geo_strip_px=sensitivity,
                common_pre_boxes=common[0], common_post_boxes=common[1],
                complete_effect_windows=complete,
                eligible_for_engineering_comparison=complete and sensitivity is not None,
                note="Descriptive engineering resolution; no causal sham subtraction or significance claim")


def timeline(summary: dict, segments: list[dict]) -> dict:
    commands = summary.get("commands", [])
    post = segments[1]
    # Readback is not a proxy for stop or restart. Missing fields stay null.
    names = ("command_started_monotonic", "stop_confirmed_monotonic",
             "parameter_verification_completed_monotonic", "restart_confirmed_monotonic")
    result = {name: post.get(name) for name in names}
    result["command_readback_monotonic"] = post.get("command_readback_monotonic")
    result["observation_started_monotonic"] = post["observation_started_monotonic"]
    stop, restart = result["stop_confirmed_monotonic"], result["restart_confirmed_monotonic"]
    result["downtime_s"] = restart - stop if stop is not None and restart is not None else None
    result["actual_parameter_writes"] = post.get("actual_parameter_writes")
    result["raw_command_records"] = commands
    result["missing_fields"] = [name for name in names if result[name] is None]
    return result


def make_pipeline() -> VisionPipeline:
    config = default_config()
    config.detector.measurement_mode = "generation_plug"
    config.detector.generation_phase_mode = "transverse_bounded"
    config.detector.generation_phase_support_fraction = .02
    return VisionPipeline(config, phase_refiner=None)


def measure_image(pipeline: VisionPipeline, image: np.ndarray, walls: list[dict], stamp: float) -> list[float]:
    geometry = wall_line_quad(image.shape[1], image.shape[0], walls)
    if geometry is None:
        raise ValueError("Invalid fixed channel geometry")
    pipeline.detector.declare_pixel_cross_section(geometry[2] - 1, reason="offline_v2_fixed_geometry")
    result = pipeline.process_frame(image, timestamp=stamp, current_wall_lines=walls)
    detected = result.detections
    if len(detected.plug_lengths_px) != len(detected.diameter_valid):
        raise ValueError("Detection completeness flags do not match candidate lengths")
    return [float(length) for length, complete in zip(detected.plug_lengths_px, detected.diameter_valid)
            if complete and math.isfinite(length) and length > 0]


def analyze(directory: Path, contract_path: Path, output: Path, *, review_path: Path | None = None,
            pipeline_factory: Callable = make_pipeline, measure: Callable = measure_image) -> dict:
    directory, output = directory.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Analysis output already exists; refuse overwrite")
    if output == directory or directory.is_relative_to(output):
        raise ValueError("Output must not contain the capture directory")
    contract = read_json(contract_path)
    source_hashes = verify_contract(contract)
    summary_path, geometry_path = directory / "session_summary.json", directory / "registered_geometry.json"
    summary = read_json(summary_path)
    segments = validate_segments(summary)
    geometry = read_json(geometry_path)
    walls = geometry.get("walls", [])
    if len(walls) != 2 or any(set(wall) != {"x1", "y1", "x2", "y2"} for wall in walls):
        raise ValueError("Two fixed registered walls required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for wall in walls for value in wall.values()):
        raise ValueError("Wall coordinates must be finite normalized sensor coordinates")
    geometry_hash = sha256(geometry_path)
    if contract.get("geometry_sha256") not in (None, geometry_hash):
        raise ValueError("Registered geometry differs from bound measurement contract")
    reviews = read_json(review_path)["frames"] if review_path else {}
    for image_name, review in reviews.items():
        contained_path(directory, image_name)
        if review.get("state") not in REVIEW_STATES:
            raise ValueError(f"Unknown independent physical state for {image_name}")
    manifest = directory / "frame_manifest.ndjson"
    with manifest.open(encoding="utf-8") as stream:
        grids = select_grid((json.loads(line) for line in stream if line.strip()), segments)
    payloads: dict[str, list[list[dict]]] = {name: [] for name in VARIANTS}
    image_count = 0
    for stage, grid in enumerate(grids):
        pipelines = {name: pipeline_factory() for name in VARIANTS}
        stage_rows: dict[str, list[dict]] = {name: [] for name in VARIANTS}
        for slot, record in enumerate(grid):
            fact = dict(stage=stage + 1, label=segments[stage]["label"], grid_slot=slot,
                        grid_start_s=slot / GRID_HZ, physical_state="missing",
                        raw_available=False, frame_median_strip_px=None,
                        detection_state="missing", candidate_lengths_strip_px=[])
            image = None
            if record is not None:
                path = contained_path(directory, record["image"])
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
                fact.update(raw_record=dict(record), physical_state=reviews.get(record["image"], {}).get("state", "unreviewed"),
                            review=reviews.get(record["image"]), detection_state="raw_unreadable")
                if image is not None:
                    if image.ndim != 2 or image.dtype != np.uint8 or list(image.shape) != record["raw_shape"]:
                        raise ValueError(f"Original Mono8 format mismatch: {record['image']}")
                    fact.update(raw_available=True, image_sha256=sha256(path))
                    image_count += 1
            for name, shift in VARIANTS.items():
                row = dict(fact, geometry=name)
                if image is not None:
                    shifted = [dict(wall, y1=wall["y1"] + shift / image.shape[0],
                                    y2=wall["y2"] + shift / image.shape[0]) for wall in walls]
                    if any(not 0 <= wall[key] <= 1 for wall in shifted for key in ("y1", "y2")):
                        raise ValueError("Perturbed wall leaves sensor; sensitivity cannot be evaluated")
                    lengths = measure(pipelines[name], image, shifted, record["host_monotonic_s"])
                    row.update(candidate_lengths_strip_px=lengths,
                               detection_state="complete_candidates" if lengths else "no_complete_candidate")
                    if lengths and row["physical_state"] in {"unreviewed", "complete_droplet"}:
                        row["frame_median_strip_px"] = float(np.median(lengths))
                stage_rows[name].append(row)
        for name in VARIANTS:
            payloads[name].append(stage_rows[name])
    bins = {name: [bin_rows(rows) for rows in stages] for name, stages in payloads.items()}
    post = bins["main"][1]
    coverage = dict(valid_post_boxes=sum(b["qualified"] for b in post),
                    valid_tail_boxes=sum(b["qualified"] for b in post[18:]),
                    missing_post_raw_fraction=1 - sum(b["saved_cells"] for b in post) / 1500,
                    nonmeasurable_post_fraction=1 - sum(b["measurable_cells"] for b in post) / 1500)
    coverage["holdout_coverage_sufficient"] = coverage["valid_post_boxes"] >= 24 and coverage["valid_tail_boxes"] >= 10
    report = dict(schema="q2-v2-pixel-analysis-v1", source_directory=str(directory),
                  primary_unit="strip_px", hardware_requested=False, selected_original_images=image_count,
                  measurement_accepted=False, holdout_passed=None, mpc_qualified=False,
                  review_status="Independent measurement review required; candidates are not ground truth",
                  method=dict(sampling_hz=5, cell_rule="first raw record per half-open time cell; no interpolation or substitution",
                              bin_s=10, minimum_saved_cells=40, minimum_measurable_cells=10,
                              statistic="median of per-frame complete-candidate length medians; not independent droplet count",
                              detector=asdict(default_config().detector) | dict(measurement_mode="generation_plug",
                                  generation_phase_mode="transverse_bounded", generation_phase_support_fraction=.02),
                              learned_model_enabled=False, geometry_shift_sensor_px=VARIANTS),
                  coverage=coverage, effect=effect_summary(bins), transaction=timeline(summary, segments), bins=bins,
                  provenance=dict(manifest_sha256=sha256(manifest), session_sha256=sha256(summary_path),
                                  registered_geometry_sha256=geometry_hash, contract_sha256=sha256(contract_path),
                                  independent_review_sha256=sha256(review_path) if review_path else None,
                                  source_sha256=source_hashes, analysis_source_sha256=sha256(Path(__file__)),
                                  camera_settings_sha256=contract.get("camera_settings_sha256"),
                                  unbound_fields=[key for key in ("geometry_sha256", "camera_settings_sha256")
                                                  if not contract.get(key)]))
    output.mkdir(parents=True)
    for name, stages in payloads.items():
        with (output / f"grid_{name}.ndjson").open("x", encoding="utf-8") as stream:
            for rows in stages:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "analysis.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", type=Path, help="Independent frame-state annotations; never the blind review key")
    args = parser.parse_args()
    report = analyze(args.session, args.contract, args.output, review_path=args.review)
    print(json.dumps({key: report[key] for key in ("hardware_requested", "selected_original_images", "coverage", "effect")},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
