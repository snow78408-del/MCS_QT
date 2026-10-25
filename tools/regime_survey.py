"""Offline companion records for joint diameter, dynamics and regime surveys.

No image heuristic can promote a segment to a verified flow regime here.
Transitions require explicit, evidence-backed review of adjacent sweep points.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REGIMES = frozenset({"jetting", "dripping", "mixed", "unbroken_thread", "unknown"})

# Confirmed by the operator for the current chip on 2026-10-07. The split
# describes plumbing; equal physical flow in the two arms is not measured.
CURRENT_PHASE_MAPPING = {
    "q1": "continuous_oil_total_to_two_side_inlets",
    "q2": "dispersed_water",
    "confirmed_on": "2026-10-07",
    "source": "operator_confirmation",
    "equal_side_split_verified": False,
}


def survey_records(segments: list[dict]) -> list[dict]:
    """Keep commanded flow and observation windows; never invent readbacks."""
    records = []
    for segment in segments:
        q1, q2 = float(segment["q1_ul_min"]), float(segment["q2_ul_min"])
        if not all(math.isfinite(q) and q > 0 for q in (q1, q2)):
            raise ValueError("survey flows must be positive and finite")
        records.append({
            "segment_index": segment["index"], "label": segment["label"],
            "q1_command_ul_min": q1, "q2_command_ul_min": q2,
            "q1_over_q2_command": q1 / q2,
            "flow_basis": "command_not_measured_chip_flow",
            "observation_started_monotonic": segment["observation_started_monotonic"],
            "observation_finished_monotonic": segment.get("observation_finished_monotonic"),
            "regime": "unknown", "review_status": "pending",
            "sweep_id": None, "scan_channel": None,
            "stable_window_monotonic": None, "evidence": [],
            "breakup_position_px": None,
            "notes": "Sparse frames and droplet size alone do not establish a regime.",
        })
    return records


def apply_reviews(records: list[dict], reviews: list[dict]) -> list[dict]:
    """Apply manual evidence without allowing edits to source flow or timing."""
    by_index = {row["segment_index"]: dict(row) for row in records}
    if len(by_index) != len(records):
        raise ValueError("duplicate segment index")
    seen = set()
    for review in reviews:
        index = review["segment_index"]
        if index not in by_index or index in seen:
            raise ValueError("unknown or duplicate reviewed segment")
        seen.add(index)
        row = by_index[index]
        regime = review["regime"]
        if regime not in REGIMES:
            raise ValueError("unknown regime label")
        evidence = review.get("evidence", [])
        window = review.get("stable_window_monotonic")
        if (not isinstance(evidence, list) or not evidence
                or not all(isinstance(item, str) and item.strip() for item in evidence)):
            raise ValueError("review requires evidence references")
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("review requires a stable observation window")
        start, stop = map(float, window)
        end = row["observation_finished_monotonic"]
        if (not all(math.isfinite(t) for t in (start, stop)) or end is None
                or not row["observation_started_monotonic"] <= start < stop <= end):
            raise ValueError("review window lies outside the completed segment")
        channel = review.get("scan_channel")
        sweep = review.get("sweep_id")
        if channel not in {"q1", "q2"} or not isinstance(sweep, str) or not sweep.strip():
            raise ValueError("review requires sweep_id and q1/q2 scan_channel")
        row.update(regime=regime, review_status="reviewed", evidence=evidence,
                   stable_window_monotonic=[start, stop], scan_channel=channel,
                   sweep_id=sweep, notes=str(review.get("notes", "")))
        # Positions are optional pixel evidence, not a fixed physical origin.
        position = review.get("breakup_position_px")
        if position is not None:
            if not isinstance(position, list) or len(position) != 2:
                raise ValueError("breakup position must be [minimum, maximum]")
            low, high = map(float, position)
            if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low <= high:
                raise ValueError("invalid breakup position range")
            row["breakup_position_px"] = [low, high]
    return list(by_index.values())


def transition_brackets(records: list[dict]) -> list[dict]:
    """Bracket reviewed transitions, without bridging unknown/mixed points.

    Only consecutive points of a monotonic, single-channel sweep qualify.
    Returns observed brackets, not fitted critical values or MPC permission.
    """
    boundaries = []
    invalid_sweeps = set()
    directions: dict[str, set[str]] = {}
    for left, right in zip(records, records[1:]):
        sweep, channel = left.get("sweep_id"), left.get("scan_channel")
        if not sweep or sweep != right.get("sweep_id"):
            continue
        if channel not in {"q1", "q2"} or channel != right.get("scan_channel"):
            invalid_sweeps.add(sweep)
            continue
        fixed = "q2_command_ul_min" if channel == "q1" else "q1_command_ul_min"
        delta = right[f"{channel}_command_ul_min"] - left[f"{channel}_command_ul_min"]
        if delta == 0 or not math.isclose(left[fixed], right[fixed], rel_tol=0, abs_tol=1e-9):
            invalid_sweeps.add(sweep)
        directions.setdefault(sweep, set()).add("increasing" if delta > 0 else "decreasing")
    invalid_sweeps.update(sweep for sweep, signs in directions.items() if len(signs) > 1)
    for left, right in zip(records, records[1:]):
        sweep = left.get("sweep_id")
        channel = left.get("scan_channel")
        if (not sweep or sweep in invalid_sweeps or sweep != right.get("sweep_id")
                or channel not in {"q1", "q2"} or channel != right.get("scan_channel")):
            continue
        varying = f"{channel}_command_ul_min"
        fixed = "q2_command_ul_min" if channel == "q1" else "q1_command_ul_min"
        delta = right[varying] - left[varying]
        if not math.isclose(left[fixed], right[fixed], rel_tol=0, abs_tol=1e-9) or delta == 0:
            continue
        direction = "increasing" if delta > 0 else "decreasing"
        if (left.get("review_status") != "reviewed" or right.get("review_status") != "reviewed"
                or {left["regime"], right["regime"]} != {"jetting", "dripping"}):
            continue
        boundaries.append({
            "sweep_id": sweep, "scan_channel": channel, "direction": direction,
            "transition": f"{left['regime']}->{right['regime']}",
            "segment_indices": [left["segment_index"], right["segment_index"]],
            "scanned_flow_bracket_ul_min": sorted([left[varying], right[varying]]),
            "fixed_channel_flow_ul_min": left[fixed],
            "q1_over_q2_bracket": sorted([left["q1_over_q2_command"], right["q1_over_q2_command"]]),
            "status": "reviewed_observation_bracket_not_exact_threshold",
        })
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_summary", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = json.loads(args.session_summary.read_text(encoding="utf-8"))
    records = survey_records(session["segments"])
    if args.reviews:
        records = apply_reviews(records, json.loads(args.reviews.read_text(encoding="utf-8")))
    payload = {"schema_version": 1, "source_session": str(args.session_summary.resolve()),
               "phase_mapping": session.get("regime_survey", {}).get("phase_mapping"),
               "records": records, "transition_brackets": transition_brackets(records),
               "validated_for_mpc": False}
    # Derived reports must not overwrite the capture or the review evidence.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
