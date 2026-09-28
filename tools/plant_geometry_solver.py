"""Legacy scenario calculator; its assumed bounds are not physical validation.

For measured phase and explicit alias handling use review_periodic_flow.py.
The derivations below are conditional on the stated model assumptions; in
particular k*phi bounds must not be used to identify the dispersed channel.

Why a solver and not a single formula
-------------------------------------
Cubaud & Mason (2008) give ``U_d = J = (Q1 + Q2) / A``: the droplet velocity
equals the *mixture* velocity, so the duct cross-section follows from a measured
pixel velocity.  The catch is that the relation only ever contains products:

    (R1)   A * s = r * Q_total / U_px        s = um/px, r = pump delivery ratio
    (R2)   k * phi = (Q_dis / Q_total) / duty    mass conservation, scale free

where ``phi`` is the fraction of the duct cross-section the droplet occupies and
``k = U_d / J`` is how much faster the droplet travels than the cross-sectional
mean.  (R1) needs two of ``{s, r, A}`` before it determines the third; the video
alone pins none of them.  (R2) needs none of them, which is why it is the one
hard statement the footage can make on its own.

This module takes whatever the bench can supply, then reports exactly what is
determined, what is only bounded, and what is still free -- rather than silently
assuming the convenient values.

Physical bounds used for the verdicts
-------------------------------------
* ``phi <= 1``: a droplet cannot occupy more than the whole cross-section.
* ``k <= 2.096``: in a square duct the centre-line speed is 2.096x the mean, so
  a droplet carried by the core cannot outrun that.
* therefore ``k * phi in [1.0, 2.096]`` for an elongated droplet in the core.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.vision import flow_locking as FL

UL_PER_MIN_TO_UM3_S = 1.0e9 / 60.0
SQUARE_DUCT_PEAK_RATIO = 2.096
K_PHI_MIN = 1.0
K_PHI_MAX = SQUARE_DUCT_PEAK_RATIO


@dataclass
class DutyEstimate:
    """One definition of the droplet duty cycle, with the length it implies."""

    name: str
    value: float
    length_px: float


@dataclass
class ImageMeasurement:
    """Everything the footage alone can state, in pixels and hertz."""

    source: str
    frames: int
    rate_hz: float
    band_rows: tuple
    band_height_px: int
    channel_x0: int
    profile_source: str
    pitch_px: float
    pitch_strength: float
    velocity_px_per_s: float
    velocity_px_per_frame: float
    velocity_residual_px: float
    frequency_hz: float
    duties: list = field(default_factory=list)
    velocity_segments: list = field(default_factory=list)
    duty_frames_used: int = 0

    def duty_range(self):
        """Widest defensible duty interval: best-case low over best-case high."""
        if not self.duties:
            return (0.0, 0.0)
        lows = [item["p10"] for item in self.duties.values()]
        highs = [item["p90"] for item in self.duties.values()]
        return (min(lows), max(highs))

    def duty_central(self):
        if not self.duties:
            return 0.0
        return float(np.median([item["median"] for item in self.duties.values()]))

    def to_dict(self):
        payload = asdict(self)
        payload["band_rows"] = list(self.band_rows)
        payload["duty_range"] = list(self.duty_range())
        payload["duty_central"] = self.duty_central()
        return payload


# --------------------------------------------------------------------------- #
# duty cycle: the single most consequential number, so measure it several ways
# --------------------------------------------------------------------------- #
def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=96)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total == 0:
        return float(np.mean(values))
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * centres) / total
    denominator = omega * (1.0 - omega)
    denominator[denominator <= 1e-12] = 1e-12
    sigma = (mu[-1] * omega - mu) ** 2 / denominator
    return float(centres[int(np.argmax(sigma))])


def fold_waveform(profile: np.ndarray, pitch: float, cycles: int = 3) -> np.ndarray:
    """Average ``cycles`` periods of the axial profile into one waveform.

    Samples are scattered into the phase grid with linear interpolation rather
    than rounded to the nearest bin, because the pitch is not an integer and the
    rounding seam otherwise shows up as a spurious steep edge.
    """
    profile = np.asarray(profile, np.float64)
    span = int(round(pitch * cycles))
    if span < 12 or span > len(profile):
        span = len(profile)
    segment = profile[:span]
    bins = max(6, int(round(pitch)))
    if pitch < 6:
        return segment
    accumulator = np.zeros(bins)
    weight = np.zeros(bins)
    index = np.arange(span, dtype=np.float64)
    phase = np.mod(index, pitch) / pitch * bins
    base = np.floor(phase).astype(int)
    fraction = phase - base
    for offset, share in ((0, 1.0 - fraction), (1, fraction)):
        target = (base + offset) % bins
        np.add.at(accumulator, target, segment * share)
        np.add.at(weight, target, share)
    return accumulator / np.maximum(weight, 1e-9)


def _run_lengths(mask: np.ndarray, minimum: int = 3) -> list:
    runs, start = [], None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        if (not flag or index == len(mask) - 1) and start is not None:
            end = index if not flag else index + 1
            runs.append(end - start)
            start = None
    return [size for size in runs if size >= minimum]


def _length_at_threshold(waveform: np.ndarray, threshold: float, unit: float) -> float:
    """Mean length of the runs that are clearly a droplet, in samples.

    ``unit`` is the span of one period *in the same samples as the waveform*:
    the pitch for a raw profile, the folded bin count for a folded waveform.
    Using ``len(waveform)`` here would treat a 400 px profile as one period and
    silently reject every droplet.
    """
    runs = _run_lengths(waveform >= threshold)
    if not runs:
        return 0.0
    keep = [size for size in runs if 0.15 * unit <= size <= 0.98 * unit]
    if not keep:
        return 0.0
    return float(np.mean(keep))


def duty_estimates(profile: np.ndarray, pitch: float, fold: bool = False,
                   cycles: int = 3) -> list:
    """Duty from three independent threshold definitions.

    The droplet edge in this rig is a gradual ramp, not a step, so no single
    threshold is "the" duty.  The spread between the three is the honest
    uncertainty, and the caller is expected to carry it forward as a range.
    """
    if pitch <= 0:
        return []
    if fold:
        waveform = fold_waveform(profile, pitch, cycles=cycles)
        unit = float(len(waveform))
    else:
        waveform = np.asarray(profile, np.float64)
        unit = float(pitch)
    if waveform.size < 6:
        return []
    low, high = np.percentile(waveform, [2, 98])
    levels = (
        ("mid_p2_p98", 0.5 * (low + high)),
        ("otsu", _otsu(waveform)),
        ("mid_p10_p90", 0.5 * (float(np.percentile(waveform, 10)) + float(np.percentile(waveform, 90)))),
    )
    estimates = []
    for name, threshold in levels:
        length = _length_at_threshold(waveform, threshold, unit)
        if length <= 0:
            continue
        estimates.append(DutyEstimate(name, min(1.0, length / unit), length))
    return estimates


def per_frame_duty(profiles: np.ndarray, pitch: float, fold: bool = False,
                   cycles: int = 3) -> dict:
    """Duty of every frame, summarised.

    Averaging frames before measuring the duty is a trap on this rig: the flow is
    still accelerating, so the droplet pattern smears across the profile and the
    apparent duty collapses.  Measuring each frame separately and taking the
    median avoids that entirely.
    """
    collected = {}
    for row in profiles:
        for item in duty_estimates(row, pitch, fold=fold, cycles=cycles):
            collected.setdefault(item.name, []).append(item.value)
    summary = {}
    for name, values in collected.items():
        array = np.asarray(values, dtype=float)
        summary[name] = {
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
            "n": int(array.size),
        }
    return summary


# --------------------------------------------------------------------------- #
# recording -> pixels
# --------------------------------------------------------------------------- #
def load_stack(path: Path, limit: int | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        stack = np.load(path)
    else:
        import cv2

        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if limit and len(frames) >= limit:
                break
        capture.release()
        if not frames:
            raise RuntimeError(f"no frames could be read from {path}")
        stack = np.stack(frames)
    if limit:
        stack = stack[:limit]
    return stack


def _as_stack(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected a (frames, rows, columns) stack, got {array.shape}")
    if array.shape[0] < 8:
        raise ValueError("need at least 8 frames to measure a flow")
    return array


def measure_stack(stack: np.ndarray, rate: float, direction: float | None = -1.0,
                  duty_frames: int = 60, segments: int = 5,
                  source: str = "<array>") -> ImageMeasurement:
    """Lock the duct, then measure pitch, velocity trend, frequency and duty.

    Velocity is reported per segment.  On the 2026-09-18 rig the flow is still
    accelerating (about -3.9 mm/s^2), so a single number for the whole recording
    is a lie: it is the average of a ramp, and its residual gives it away.
    """
    stack = _as_stack(stack)
    whole = FL.measure_flow(stack, 1.0 / rate, direction=direction)
    if whole.verdict not in ("FLOWING", "WEAK_SIGNAL", "STATIC"):
        raise RuntimeError(f"cannot measure this recording: {whole.verdict} ({whole.reason})")
    pitch = float(whole.pitch_px)
    if pitch <= 0:
        raise RuntimeError("no droplet pitch could be locked")

    top, bot = whole.band_rows
    centre, _ = FL.duct_centre_line(stack, top, bot, x_lo=whole.channel_x0)
    profiles = FL.centreline_profiles(stack, centre, 2, x_lo=whole.channel_x0)
    if profiles.shape[1] < 2 * pitch:
        profiles = FL.channel_profiles(stack, top, bot, x_lo=whole.channel_x0)

    count = int(profiles.shape[0])
    span = max(24, count // max(1, segments))
    velocity_segments = []
    for begin in range(0, count - span + 1, span):
        chunk = profiles[begin:begin + span]
        estimate = FL.estimate_velocity(chunk, 1.0 / rate, pitch=pitch, direction=direction)
        if not estimate.ok:
            continue
        velocity_segments.append({
            "t_start_s": begin / rate,
            "t_end_s": (begin + span) / rate,
            "px_per_frame": float(estimate.px_per_frame),
            "px_per_s": float(estimate.px_per_second),
            "residual_px": float(estimate.residual_px),
            "confidence": estimate.confidence,
        })

    window = max(duty_frames, span)
    tail = profiles[-min(count, window):]
    duties = per_frame_duty(tail, pitch)

    last = velocity_segments[-1] if velocity_segments else {
        "px_per_frame": float(whole.velocity.px_per_frame),
        "px_per_s": float(whole.velocity.px_per_second),
        "residual_px": float(whole.velocity.residual_px),
        "confidence": whole.velocity.confidence,
    }

    return ImageMeasurement(
        source=source,
        frames=count,
        rate_hz=float(rate),
        band_rows=(int(top), int(bot)),
        band_height_px=int(bot - top + 1),
        channel_x0=int(whole.channel_x0),
        profile_source=whole.profile_source,
        pitch_px=pitch,
        pitch_strength=float(whole.pitch_strength),
        velocity_px_per_s=float(last["px_per_s"]),
        velocity_px_per_frame=float(last["px_per_frame"]),
        velocity_residual_px=float(last["residual_px"]),
        frequency_hz=float(abs(last["px_per_s"]) / pitch) if pitch else 0.0,
        duties=duties,
        velocity_segments=velocity_segments,
        duty_frames_used=int(tail.shape[0]),
    )


# --------------------------------------------------------------------------- #
# pixels + bench -> plant parameters
# --------------------------------------------------------------------------- #
def scale_free_verdict(image: ImageMeasurement, q1: float, q2: float) -> dict:
    """The one statement the footage makes without any bench input."""
    total = q1 + q2
    result = {"Q1_ul_per_min": q1, "Q2_ul_per_min": q2, "candidates": {}}
    for label, dispersed, continuous in (("CH1_dispersed", q1, q2), ("CH2_dispersed", q2, q1)):
        share = dispersed / total
        entries = []
        for name, stats in image.duties.items():
            value = float(stats["median"])
            if value <= 0.0:
                continue
            product = share / value
            entries.append({
                "duty_from": name,
                "duty": value,
                "duty_p10": float(stats["p10"]),
                "duty_p90": float(stats["p90"]),
                "k_times_phi": product,
                "k_times_phi_at_p10": share / max(1e-9, float(stats["p10"])),
                "k_times_phi_at_p90": share / max(1e-9, float(stats["p90"])),
                "physically_possible": bool(K_PHI_MIN <= product <= K_PHI_MAX),
            })
        products = [entry["k_times_phi"] for entry in entries]
        result["candidates"][label] = {
            "dispersed_share_of_total": share,
            "k_times_phi_entries": entries,
            "k_times_phi_min": min(products) if products else None,
            "k_times_phi_max": max(products) if products else None,
            "within_core_flow_bounds": bool(
                products and min(products) >= K_PHI_MIN and max(products) <= K_PHI_MAX),
        }
    result["duty_low"], result["duty_high"] = image.duty_range()
    result["duty_central"] = image.duty_central()
    result["bounds"] = {"k_phi_min": K_PHI_MIN, "k_phi_max": K_PHI_MAX,
                        "square_duct_peak_ratio": SQUARE_DUCT_PEAK_RATIO}
    verdicts = [item["within_core_flow_bounds"] for item in result["candidates"].values()]
    if sum(verdicts) == 1:
        result["verdict"] = next(name for name, item in result["candidates"].items()
                                 if item["within_core_flow_bounds"])
    elif not any(verdicts):
        result["verdict"] = "NO_CANDIDATE"
    else:
        result["verdict"] = "AMBIGUOUS"
    return result


def cross_section_from_cubaud(image: ImageMeasurement, q1: float, q2: float,
                              scale_um_per_px: float, pump_ratio: float = 1.0) -> dict:
    """(R1): A * s = r * Q_total / U_px."""
    total = (q1 + q2) * UL_PER_MIN_TO_UM3_S
    speed_px = abs(image.velocity_px_per_s)
    if speed_px <= 0 or scale_um_per_px <= 0:
        raise ValueError("need a positive velocity and scale")
    area = pump_ratio * total / (speed_px * scale_um_per_px)
    width_um = image.band_height_px * scale_um_per_px
    return {
        "scale_um_per_px": scale_um_per_px,
        "pump_ratio": pump_ratio,
        "Q_total_um3_per_s": total,
        "U_um_per_s": speed_px * scale_um_per_px,
        "area_um2": area,
        "band_width_um": width_um,
        "height_if_band_is_full_width_um": area / width_um,
        "band_is_square_at_this_scale": bool(
            abs(area / width_um - width_um) < 0.08 * width_um),
    }


def pump_ratio_bounds(image: ImageMeasurement, q1: float, q2: float,
                      scale_um_per_px: float, width_um: float, height_um: float,
                      k_lo: float = K_PHI_MIN, k_hi: float = K_PHI_MAX) -> dict:
    """Invert (R1) for the pump ratio, over the physical range of ``k``.

    ``r = U_px * s * A / (k * Q_total)``.  A bench measurement of ``s`` and of
    the duct cross-section therefore converts straight into a bound on how much
    the pump actually delivers.
    """
    total = (q1 + q2) * UL_PER_MIN_TO_UM3_S
    speed = abs(image.velocity_px_per_s) * scale_um_per_px
    area = width_um * height_um
    if total <= 0 or k_lo <= 0:
        raise ValueError("need a positive flow and k")
    high = speed * area / (k_lo * total)
    low = speed * area / (k_hi * total)
    return {
        "scale_um_per_px": scale_um_per_px,
        "channel_um": [width_um, height_um],
        "area_um2": area,
        "U_um_per_s": speed,
        "pump_ratio_low": min(low, high),
        "pump_ratio_high": max(low, high),
        "pump_ratio_at_k_1": speed * area / total,
        "k_range": [k_lo, k_hi],
    }


def scale_from_pump(image: ImageMeasurement, q1: float, q2: float,
                    pump_ratio: float, width_um: float, height_um: float) -> dict:
    """Invert (R1) for the pixel scale when the pump and the duct are known."""
    total = (q1 + q2) * UL_PER_MIN_TO_UM3_S
    speed_px = abs(image.velocity_px_per_s)
    if speed_px <= 0:
        raise ValueError("need a positive velocity")
    scale = pump_ratio * total / (speed_px * width_um * height_um)
    return {
        "pump_ratio": pump_ratio,
        "channel_um": [width_um, height_um],
        "scale_um_per_px": scale,
        "band_width_um_at_that_scale": image.band_height_px * scale,
    }


def measure_recording(path: Path, rate: float, direction: float | None = -1.0,
                      limit: int | None = None, duty_frames: int = 60,
                      segments: int = 5) -> ImageMeasurement:
    """Load a recording and measure it."""
    return measure_stack(load_stack(path, limit=limit), rate, direction=direction,
                         duty_frames=duty_frames, segments=segments, source=str(path))


def build_report(image: ImageMeasurement, q1: float, q2: float,
                 scale_um_per_px: float | None = None, pump_ratio: float | None = None,
                 width_um: float | None = None, height_um: float | None = None) -> dict:
    report = {
        "validity": {
            "control_authorized": False,
            "status": "UNVALIDATED_SCENARIO",
            "phase_identity_confirmed": False,
            "velocity_alias_resolved": False,
            "assumptions": [
                "Legacy k*phi bounds are assumed, not universal physical bounds.",
                "Intensity duty is not independently validated droplet volume fraction.",
                "Command flow is not independently measured delivered flow.",
            ],
        },
        "image": image.to_dict(),
        "scale_free": scale_free_verdict(image, q1, q2),
        "given": {"scale_um_per_px": scale_um_per_px, "pump_ratio": pump_ratio,
                  "channel_width_um": width_um, "channel_height_um": height_um},
        "derived": {},
        "missing": [],
    }
    derived = report["derived"]
    if scale_um_per_px:
        for label, ratio in (("pump_ratio_given", pump_ratio),):
            if ratio:
                derived["cross_section_from_cubaud"] = cross_section_from_cubaud(
                    image, q1, q2, scale_um_per_px, pump_ratio=ratio)
        if width_um and height_um:
            derived["pump_ratio_bounds"] = pump_ratio_bounds(
                image, q1, q2, scale_um_per_px, width_um, height_um)
            if not pump_ratio:
                report["missing"].append("pump_ratio (needed to fix the area, not just bound r)")
        else:
            report["missing"].append("channel_width_um and channel_height_um")
    else:
        report["missing"].append("scale_um_per_px (a scale bar under the same objective)")
    if pump_ratio and width_um and height_um and not scale_um_per_px:
        derived["scale_from_pump"] = scale_from_pump(image, q1, q2, pump_ratio, width_um, height_um)
    return report


def format_report(report: dict) -> str:
    image = report["image"]
    lines = []
    lines.append("UNVALIDATED SCENARIO: no phase identity, true speed or control parameters are confirmed.")
    lines.append("Use review_periodic_flow.py for quality-screened offline image measurements.")
    lines.append("=== image (pixels and hertz only) ===")
    lines.append(f"  source          {image['source']}")
    lines.append(f"  frames / rate   {image['frames']} @ {image['rate_hz']:.2f} Hz")
    lines.append(f"  band rows       {image['band_rows'][0]}-{image['band_rows'][1]}"
                 f"  ({image['band_height_px']} px)   x0 {image['channel_x0']}"
                 f"   profile {image['profile_source']}")
    lines.append(f"  pitch           {image['pitch_px']:.1f} px   (strength {image['pitch_strength']:.3f})")
    lines.append(f"  velocity        {image['velocity_px_per_frame']:+.2f} px/frame"
                 f" = {image['velocity_px_per_s']:+.0f} px/s"
                 f"   (last segment, residual {image['velocity_residual_px']:.2f} px)")
    segments = image.get("velocity_segments") or []
    if len(segments) > 1:
        lines.append("  velocity trend:")
        for item in segments:
            lines.append(f"     t {item['t_start_s']:5.2f}-{item['t_end_s']:5.2f} s"
                         f"   {item['px_per_frame']:+7.2f} px/frame"
                         f"   (residual {item['residual_px']:.2f})")
    lines.append(f"  frequency       {image['frequency_hz']:.1f} Hz  (= |v| / pitch)")
    lines.append(f"  duty, measured per frame over the last {image['duty_frames_used']} frames:")
    for name, stats in image["duties"].items():
        lines.append(f"     {name:>12s}  median {stats['median']:.3f}"
                     f"   p10-p90 {stats['p10']:.3f}-{stats['p90']:.3f}   n={stats['n']}")
    low, high = image["duty_range"]
    lines.append(f"  duty range      {low:.3f} - {high:.3f}"
                 f"   (central {image['duty_central']:.3f})")

    lines.append("")
    lines.append("=== conditional scenario comparison (requires independent bench validation) ===")
    free = report["scale_free"]
    lines.append(f"  duty central {free['duty_central']:.3f}"
                 f"   (range {free['duty_low']:.3f}-{free['duty_high']:.3f})")
    lines.append(f"  assumed window for k*phi: [{free['bounds']['k_phi_min']},"
                 f" {free['bounds']['k_phi_max']}]")
    for label, item in free["candidates"].items():
        span = (f"{item['k_times_phi_min']:.3f} - {item['k_times_phi_max']:.3f}"
                if item["k_times_phi_min"] is not None else "n/a")
        mark = "OK " if item["within_core_flow_bounds"] else "NO "
        lines.append(f"  {mark}{label:<14s} share {item['dispersed_share_of_total']:.3f}"
                     f"   k*phi = {span}")
    lines.append(f"  verdict: {free['verdict']}")

    derived = report["derived"]
    if "cross_section_from_cubaud" in derived:
        item = derived["cross_section_from_cubaud"]
        lines.append("")
        lines.append("=== cross-section from Cubaud, given s and r ===")
        lines.append(f"  s = {item['scale_um_per_px']:.4f} um/px, r = {item['pump_ratio']:.3f}")
        lines.append(f"  U = {item['U_um_per_s']:.0f} um/s   A = {item['area_um2']:.0f} um^2")
        lines.append(f"  band {item['band_width_um']:.1f} um wide"
                     f"  -> height {item['height_if_band_is_full_width_um']:.1f} um")
    if "pump_ratio_bounds" in derived:
        item = derived["pump_ratio_bounds"]
        lines.append("")
        lines.append("=== pump delivery ratio, inverted from a known duct ===")
        lines.append(f"  s = {item['scale_um_per_px']:.4f} um/px,"
                     f" channel {item['channel_um'][0]:.0f} x {item['channel_um'][1]:.0f} um")
        lines.append(f"  r = {item['pump_ratio_low']:.3f} - {item['pump_ratio_high']:.3f}"
                     f"   (at k = 1: {item['pump_ratio_at_k_1']:.3f})")
    if "scale_from_pump" in derived:
        item = derived["scale_from_pump"]
        lines.append("")
        lines.append("=== pixel scale, inverted from a known pump and duct ===")
        lines.append(f"  s = {item['scale_um_per_px']:.4f} um/px")
        lines.append(f"  band would be {item['band_width_um_at_that_scale']:.1f} um wide")
    if report["missing"]:
        lines.append("")
        lines.append("=== still needed ===")
        for item in report["missing"]:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="stack .npy or a video file")
    parser.add_argument("--rate", type=float, required=True, help="frames per second")
    parser.add_argument("--q1", type=float, default=70.0, help="channel 1 flow, uL/min")
    parser.add_argument("--q2", type=float, default=20.0, help="channel 2 flow, uL/min")
    parser.add_argument("--scale", type=float, default=None, help="um per pixel, from a scale bar")
    parser.add_argument("--pump-ratio", type=float, default=None,
                        help="delivered/commanded flow, from a syringe displacement test")
    parser.add_argument("--width-um", type=float, default=None, help="duct width in um")
    parser.add_argument("--height-um", type=float, default=None, help="duct height in um")
    parser.add_argument("--direction", choices=("auto", "left", "right"), default="left")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N frames")
    parser.add_argument("--out", default=None, help="write the report as JSON here")
    args = parser.parse_args()

    direction = {"auto": None, "left": -1.0, "right": 1.0}[args.direction]
    image = measure_recording(Path(args.video), args.rate, direction=direction, limit=args.limit)
    report = build_report(image, args.q1, args.q2,
                          scale_um_per_px=args.scale, pump_ratio=args.pump_ratio,
                          width_um=args.width_um, height_um=args.height_um)
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
