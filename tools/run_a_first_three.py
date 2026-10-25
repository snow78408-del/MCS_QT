"""Bounded, raw-PNG-only A survey blocks; hardware requires explicit --live.

Reuses the existing verified stop/write/restart lifecycle. This records
exploration evidence and never declares measurement or steady-state acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_r1_front4_live as live
from tools.plant_flow_transient_capture import (
    CaptureLifecycleError, TASK_GOAL_DIAGNOSTIC_CAPTURE,
)


PROFILE = "requested_a1_a3_70_60_50_20261014"
SEQUENCE = [(f"A{i}", float(q), 20.0, i > 1)
            for i, q in enumerate((70, 60, 50), 1)]
REFILL_PAIR_PROFILE = "requested_a4_a5_60_70_20261014"
HIGH_PAIR_PROFILE = "requested_a6_a7_80_90_20261014"
A7_RESUME_PROFILE = "requested_a7_only_90_20261014"
BLOCK_SEQUENCES = {
    PROFILE: SEQUENCE,
    REFILL_PAIR_PROFILE: [("A4", 60., 20., True), ("A5", 70., 20., True)],
    HIGH_PAIR_PROFILE: [("A6", 80., 20., True), ("A7", 90., 20., True)],
    A7_RESUME_PROFILE: [("A7", 90., 20., True)],
}
REFILL_REFERENCE_SECONDS = {REFILL_PAIR_PROFILE: 60., HIGH_PAIR_PROFILE: 30.}


def refill_block_budget_ul(profile: str) -> float:
    """Upper oil estimate including the reference and30s transaction allowance."""
    sequence = BLOCK_SEQUENCES[profile]
    observation = sum((q1 + .5) * 5. for _, q1, _, _ in sequence)
    reference = 70.5 * REFILL_REFERENCE_SECONDS[profile] / 60.
    transaction = max(q1 + .5 for _, q1, _, _ in sequence) * .5
    return observation + reference + transaction


class RawSurveyRecorder:
    """One fixed-grid sample per second plus three short continuous sequences.

    PNG writes run on a bounded worker. Overflow and write errors propagate
    to the camera reader, which triggers the existing verified pump cleanup.
    """

    def __init__(self, *, directory: Path, interval_s: float, walls: list) -> None:
        if interval_s != 1.0:
            raise ValueError("A survey requires a fixed 1Hz grid")
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._phase: tuple[str, float] | None = None
        self._slot = -1
        self._shape = None
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._stream = None
        self._last_stamp: float | None = None
        self._last_id: int | None = None
        self.received = 0
        self.saved = 0
        self.hardware_gaps = 0
        self.saved_by_phase: dict[str, int] = {}
        self.saved_bytes = 0

    def set_phase(self, name: str, start: float) -> None:
        with self._lock:
            self._phase = (name, start)
            self._slot = -1

    def prepare(self, frame: object) -> None:
        image = frame.image
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("Original Mono8 grayscale frames required")
        self._shape = image.shape
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "frames").mkdir()
        if not cv2.imwrite(str(self.directory / "pre_start.png"), image):
            raise OSError("Cannot persist pre-start original image")
        self._stream = (self.directory / "frame_manifest.ndjson").open("x", encoding="utf-8")
        self._thread = threading.Thread(target=self._write_loop, name="a-png-writer", daemon=True)
        self._thread.start()

    def check(self) -> None:
        if self._error is not None:
            raise CaptureLifecycleError(f"PNG writer failed: {self._error!r}") from self._error

    def submit(self, frame: object) -> None:
        self.check()
        image = frame.image
        stamp = float(frame.host_monotonic_timestamp)
        identifier = int(frame.hardware_frame_id)
        if (not math.isfinite(stamp) or stamp <= 0 or identifier <= 0
                or (self._last_stamp is not None and stamp <= self._last_stamp)):
            raise ValueError("Invalid frame identity or nonmonotonic host timestamp")
        if image.shape != self._shape or image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("Original image format changed")
        gap = max(0, identifier - self._last_id - 1) if self._last_id is not None else 0
        self.hardware_gaps += gap
        self._last_stamp, self._last_id = stamp, identifier
        software_id = self.received
        self.received += 1
        with self._lock:
            phase = self._phase
            if phase is None:
                return
            name, start = phase
            elapsed = stamp - start
            if elapsed < 0:
                return
            slot = int(elapsed)
            grid = slot > self._slot
            if grid:
                self._slot = slot
            # Leave late slots absent instead of fabricating past samples.
            grid = grid and elapsed - slot <= .15
        burst = next((i for i, offset in enumerate((250., 260., 270.), 1)
                      if name in {f"A{i}" for i in range(1, 10)}
                      and offset <= elapsed < offset + 2.), None)
        if not grid and burst is None and name != "storage_pilot":
            return
        row = {
            "phase": name, "elapsed_s": elapsed, "grid_slot": slot if grid else None,
            "burst": burst, "software_frame_id": software_id,
            "hardware_frame_id": identifier, "host_monotonic_s": stamp,
            "wall_clock_s": float(getattr(frame, "timestamp", 0.0)),
            "hardware_timestamp_ticks": int(getattr(frame, "hardware_timestamp_ticks", 0)),
            "sdk_host_timestamp_ticks": int(getattr(frame, "sdk_host_timestamp_ticks", 0)),
            "lost_packet_count": int(getattr(frame, "lost_packet_count", 0)),
            "hardware_gap_before_frame": gap,
            "image": f"frames/{name}_sw{software_id:08d}_hw{identifier:08d}.png",
            "raw_shape": list(image.shape), "endpoint_validated": False,
        }
        try:
            self._queue.put_nowait((image.copy(), row))
        except queue.Full as exc:
            raise CaptureLifecycleError("Bounded PNG queue full; stopping capture") from exc

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                image, row = item
                path = self.directory / row["image"]
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f"Cannot save {path.name}")
                row["bytes"] = path.stat().st_size
                self._stream.write(json.dumps(row) + "\n")
                self._stream.flush()
                self.saved += 1
                self.saved_bytes += row["bytes"]
                name = row["phase"]
                self.saved_by_phase[name] = self.saved_by_phase.get(name, 0) + 1
                if row["grid_slot"] is not None:
                    temporary = self.directory / "latest.tmp.png"
                    shutil.copyfile(path, temporary)
                    os.replace(temporary, self.directory / "latest.png")
        except BaseException as exc:
            self._error = exc

    def close(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                try:
                    self._queue.put(None, timeout=5)
                except queue.Full:
                    self._error = self._error or RuntimeError("PNG writer not draining")
                self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("PNG writer did not stop")
            self._thread = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self.check()

    def summary(self) -> dict:
        return {"received": self.received, "saved": self.saved,
                "saved_by_phase": dict(self.saved_by_phase), "saved_bytes": self.saved_bytes,
                "hardware_gaps": self.hardware_gaps, "raw_video_saved": False,
                "sampling_premise_ok": self.saved > 0 and self._error is None,
                "measurement_accepted": False,
                "measurement_acceptance_reason": "exploration_raw_frames_measurement_review_pending"}


class AFirstThreeSession(live.R1FrontFourSession):
    def __init__(self, **kwargs) -> None:
        kwargs["task_goal"] = TASK_GOAL_DIAGNOSTIC_CAPTURE
        super().__init__(**kwargs)
        self._delivery_last: float | None = None
        flow = self.plan.get("flow_plan", {})
        self._upper_rates = (float(flow.get("baseline_q1_ul_min", 70.)) + .5,
                             float(flow.get("baseline_q2_ul_min", 20.)) + .5)
        self._delivery_limit_ul = min(950., float(flow.get("max_cumulative_delivery_each_ul", 950.)))
        self._delivery_ul = [0., 0.]

    def _update_delivery(self) -> None:
        now = self._clock()
        if self._delivery_last is not None:
            dt = max(0., now - self._delivery_last)
            for i, rate in enumerate(self._upper_rates):
                self._delivery_ul[i] += rate * dt / 60.
            self._delivery_last = now

    def _record(self, event: str) -> None:
        super()._record(event)
        if event.startswith("segment_") and event.endswith("_started"):
            self._update_delivery()
            segment = self.segment_records[-1]
            self._upper_rates = (segment["q1_ul_min"] + .5, segment["q2_ul_min"] + .5)
            self.sink.set_phase(segment["label"], segment["observation_started_monotonic"])
        if event.startswith("segment_") and event.endswith("_finished"):
            self._update_delivery()
            # A higher target may already be flowing during restart verification.
            # Charge the entire upcoming transaction at the larger rate.
            steps = self.plan["flow_plan"]["steps"]
            next_index = len(self.segment_records)
            if next_index < len(steps):
                next_step = steps[next_index]
                self._upper_rates = tuple(max(old, float(next_step[key]) + .5)
                                          for old, key in zip(self._upper_rates,
                                                              ("q1_ul_min", "q2_ul_min")))
        self._update_delivery()
        status = {"event": event, "monotonic_s": self._clock(),
                  "segments": self.segment_records, "frames_received": self.frames_seen,
                  "capture": self.sink.summary(), "conservative_delivery_upper_ul": self._delivery_ul,
                  "measurement_accepted": False}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.output_dir / "live_status.tmp.json"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(temporary, self.output_dir / "live_status.json")
        print(json.dumps(status), flush=True)

    def _write_pump_parameters(self) -> None:
        # A ten-second full-frame storage pilot and a separate precommand
        # baseline happen while the existing lifecycle has verified pump stop.
        self.sink.set_phase("storage_pilot", self._clock())
        self._wait_segment(10.)
        if self.sink.saved_by_phase.get("storage_pilot", 0) < 20:
            raise CaptureLifecycleError("Insufficient persisted storage pilot frames")
        self._record("storage_pilot_finished")
        self.sink.set_phase("precommand", self._clock())
        self._wait_segment(30.)
        super()._write_pump_parameters()

    def _start_infusion(self) -> None:
        # Count the full start verification and transition intervals as flowing
        # at the previous upper rate, even while the hardware is stopped.
        self._delivery_last = self._clock()
        super()._start_infusion()
        profile = self.plan["flow_plan"]["campaign_profile"]
        if profile in REFILL_REFERENCE_SECONDS:
            # This short recovery is separately labelled, never counted as a
            # qualified steady-state point or joined to the pre-refill sweep.
            self.sink.set_phase("refill_reference_70_20", self._clock())
            self._record("refill_reference_70_20_started")
            self._wait_segment(REFILL_REFERENCE_SECONDS[profile])
            self._record("refill_reference_70_20_finished")
            self._update_delivery()
            first = self.plan["flow_plan"]["steps"][0]
            self._upper_rates = tuple(max(old, float(first[key]) + .5)
                                      for old, key in zip(self._upper_rates,
                                                          ("q1_ul_min", "q2_ul_min")))
        elif profile == A7_RESUME_PROFILE:
            self._update_delivery()
            self._upper_rates = (90.5, 20.5)

    def _wait_segment(self, duration_s: float) -> None:
        deadline = self._clock() + duration_s
        last_status = self._clock()
        while True:
            self._raise_reader_error()
            self.sink.check()
            self._update_delivery()
            if any(v >= self._delivery_limit_ul for v in self._delivery_ul):
                raise CaptureLifecycleError(f"Conservative {self._delivery_limit_ul:g}uL delivery limit reached")
            if (self.output_dir / "STOP").exists():
                raise CaptureLifecycleError("Operator STOP file received")
            if shutil.disk_usage(self.output_dir).free < 5_000_000_000:
                raise CaptureLifecycleError("Less than5GB free storage")
            if self._clock() - last_status >= 5.:
                self._record("capture_heartbeat")
                last_status = self._clock()
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            self._reader_stop.wait(min(.25, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--live", action="store_true", help="Explicitly allow the bounded hardware session")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    profile = plan.get("flow_plan", {}).get("campaign_profile")
    if profile not in BLOCK_SEQUENCES:
        parser.error("unsupported A block profile")
    if profile == A7_RESUME_PROFILE:
        flow = plan["flow_plan"]
        prior_dir = Path(str(plan.get("A_resume", {}).get("previous_session_directory", "")))
        try:
            prior = json.loads((prior_dir / "session_summary.json").read_text(encoding="utf-8"))
            status = json.loads((prior_dir / "live_status.json").read_text(encoding="utf-8"))
            valid_prior = (prior["stop"]["verified"] is True and not prior["cleanup_errors"]
                           and [s["label"] for s in prior["segments"]] == ["A6"]
                           and prior["segments"][0]["observation_finished_monotonic"] is not None)
            used = max(status["conservative_delivery_upper_ul"])
            limit = float(flow["max_cumulative_delivery_each_ul"])
            valid_budget = math.isfinite(used) and used >= 0 and 90.5 * 5.5 <= limit <= 950. - used
        except (OSError, ValueError, KeyError, TypeError):
            valid_prior = valid_budget = False
        if (not valid_prior or not valid_budget or flow.get("baseline_q1_ul_min") != 80.
                or flow.get("baseline_q2_ul_min") != 20. or not flow.get("transition_review_file")):
            parser.error("A7 resume requires verified stopped A6 evidence and shared refill budget")
    if profile in REFILL_REFERENCE_SECONDS:
        flow = plan["flow_plan"]
        if (flow.get("baseline_q1_ul_min") != 70.
                or flow.get("baseline_q2_ul_min") != 20.):
            parser.error("refilled A block requires a70/20 initial reference")
        # Reference + observations + a conservative30s transaction allowance.
        required = refill_block_budget_ul(profile)
        if float(flow.get("max_cumulative_delivery_each_ul", 0)) < required:
            parser.error("A block delivery budget must cover reference and transactions")
        if profile == HIGH_PAIR_PROFILE and not flow.get("transition_review_file"):
            parser.error("high-side block requires a current visual transition review file")
    # Keep the original plan validation, device locks, provenance and cleanup.
    from tools import sampled_frame_recorder as sampled
    previous_profile = live.CAMPAIGN_PROFILES.get(profile)
    previous_session, previous_recorder = live.R1FrontFourSession, sampled.SampledFrameRecorder
    old_argv = sys.argv
    try:
        live.CAMPAIGN_PROFILES[profile] = BLOCK_SEQUENCES[profile]
        live.R1FrontFourSession = AFirstThreeSession
        sampled.SampledFrameRecorder = RawSurveyRecorder
        sys.argv = [old_argv[0], "--plan", args.plan, "--sample-interval-s", "1"]
        if not args.live:
            sys.argv.append("--preflight-only")
        return live.main()
    finally:
        sys.argv = old_argv
        live.R1FrontFourSession, sampled.SampledFrameRecorder = previous_session, previous_recorder
        if previous_profile is None:
            live.CAMPAIGN_PROFILES.pop(profile, None)
        else:
            live.CAMPAIGN_PROFILES[profile] = previous_profile


if __name__ == "__main__":
    raise SystemExit(main())
