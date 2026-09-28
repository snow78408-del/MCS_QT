"""Offline tests for the plant geometry solver.

Everything here runs on synthetic frames: no camera, no pump, no display and no
large recordings, so it works in CI and inside the sandbox.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.plant_geometry_solver import (
    K_PHI_MAX,
    ImageMeasurement,
    build_report,
    cross_section_from_cubaud,
    duty_estimates,
    fold_waveform,
    measure_stack,
    per_frame_duty,
    pump_ratio_bounds,
    scale_free_verdict,
    scale_from_pump,
)


def test_legacy_report_cannot_authorize_control_or_confirm_phase() -> None:
    image = measure_stack(train_stack(), 100.0)
    report = build_report(image, 70.0, 20.0)
    assert report["validity"]["status"] == "UNVALIDATED_SCENARIO"
    assert report["validity"]["control_authorized"] is False
    assert report["validity"]["phase_identity_confirmed"] is False
    assert report["validity"]["velocity_alias_resolved"] is False


def train_frame(phase: float, duty: float, pitch: float, rows: int = 200, cols: int = 420,
                slope: float = 0.0, base: float = 45.0, amplitude: float = 26.0) -> np.ndarray:
    rows_axis = np.arange(rows, dtype=np.float32)[:, None]
    centre = 110.0 + slope * np.arange(cols, dtype=np.float64)
    position = np.mod(np.arange(cols, dtype=np.float64) + phase, pitch)
    gate = (position < duty * pitch).astype(np.float32)
    frame = np.full((rows, cols), base, dtype=np.float32)
    frame += 8.0 * np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    frame += 3.0 * np.sin(np.arange(rows, dtype=np.float32) / 9.0)[:, None]
    frame += amplitude * gate[None, :] * np.exp(-((rows_axis - centre[None, :]) ** 2) / (2 * 5.0 ** 2))
    return frame


def train_stack(frames: int = 60, duty: float = 0.55, pitch: float = 60.0, speed: float = 8.0,
                slope: float = 0.0, noise: float = 0.3, started: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(3)
    stack = np.stack([
        train_frame(started + index * speed, duty, pitch, slope=slope)
        for index in range(frames)
    ])
    return (stack + rng.normal(0.0, noise, stack.shape)).astype(np.float32)


def fake_image(velocity_px_per_s: float = -10000.0, band_height_px: int = 100,
               duty: float = 0.55) -> ImageMeasurement:
    return ImageMeasurement(
        source="<synthetic>", frames=100, rate_hz=320.0, band_rows=(100, 199),
        band_height_px=band_height_px, channel_x0=0, profile_source="band",
        pitch_px=60.0, pitch_strength=0.7, velocity_px_per_s=velocity_px_per_s,
        velocity_px_per_frame=velocity_px_per_s / 320.0, velocity_residual_px=0.2,
        frequency_hz=abs(velocity_px_per_s) / 60.0,
        duties={"mid_p2_p98": {"median": duty, "p10": duty - 0.01, "p90": duty + 0.01, "n": 100}},
    )


def test_fold_waveform_recovers_a_known_duty() -> None:
    stack = train_stack(duty=0.55, slope=0.0)
    profile = stack[0][104:117].mean(axis=0)
    waveform = fold_waveform(profile, pitch=60.0, cycles=4)
    assert waveform.size >= 6
    assert waveform.max() > waveform.min() + 10.0


def test_duty_estimators_agree_on_a_synthetic_train() -> None:
    stack = train_stack(duty=0.55)
    profile = stack[0][104:117].mean(axis=0)
    for fold in (False, True):
        values = [item.value for item in duty_estimates(profile, 60.0, fold=fold)]
        assert values, f"no duty estimate with fold={fold}"
        for value in values:
            assert 0.42 < value < 0.68, f"fold={fold} gave duty {value}"


def test_per_frame_duty_is_stable_across_frames() -> None:
    stack = train_stack(frames=40, duty=0.60)
    profiles = np.stack([frame[104:117].mean(axis=0) for frame in stack])
    summary = per_frame_duty(profiles, 60.0)
    assert summary
    for stats in summary.values():
        assert 0.45 < stats["median"] < 0.75
        assert stats["n"] == 40


def test_scale_free_verdict_prefers_the_dispersed_channel() -> None:
    stack = train_stack(frames=80, duty=0.75, speed=6.0)
    image = measure_stack(stack, rate=320.0, segments=2)
    free = scale_free_verdict(image, 70.0, 20.0)
    assert free["verdict"] == "CH1_dispersed"
    assert free["candidates"]["CH1_dispersed"]["within_core_flow_bounds"] is True


def test_scale_free_verdict_can_pick_the_other_channel() -> None:
    stack = train_stack(frames=80, duty=0.15, speed=6.0)
    image = measure_stack(stack, rate=320.0, segments=2)
    free = scale_free_verdict(image, 70.0, 20.0)
    assert free["verdict"] == "CH2_dispersed"


def test_scale_free_product_does_not_depend_on_a_length_scale() -> None:
    """k*phi is built from shares and a duty ratio, so no um/px enters it."""
    image = fake_image(duty=0.50)
    free = scale_free_verdict(image, 70.0, 20.0)
    entry = free["candidates"]["CH1_dispersed"]["k_times_phi_entries"][0]
    assert entry["k_times_phi"] == pytest.approx((70.0 / 90.0) / 0.50, rel=1e-9)


def test_cross_section_inverts_cubaud() -> None:
    image = fake_image(velocity_px_per_s=-10000.0)
    result = cross_section_from_cubaud(image, 70.0, 20.0, scale_um_per_px=2.0, pump_ratio=1.0)
    total = 90.0 * 1.0e9 / 60.0
    assert result["area_um2"] == pytest.approx(total / (10000.0 * 2.0), rel=1e-9)
    assert result["U_um_per_s"] == pytest.approx(20000.0, rel=1e-9)


def test_pump_ratio_bounds_bracket_a_known_truth() -> None:
    image = fake_image(velocity_px_per_s=-33333.3)
    bounds = pump_ratio_bounds(image, 70.0, 20.0, scale_um_per_px=2.0,
                               width_um=150.0, height_um=150.0)
    assert bounds["pump_ratio_at_k_1"] == pytest.approx(1.0, abs=0.01)
    assert bounds["pump_ratio_low"] == pytest.approx(1.0 / K_PHI_MAX, rel=0.02)
    assert bounds["pump_ratio_high"] == pytest.approx(1.0, rel=0.02)


def test_scale_from_pump_inverts_the_relation() -> None:
    image = fake_image(velocity_px_per_s=-33333.3)
    result = scale_from_pump(image, 70.0, 20.0, pump_ratio=1.0, width_um=150.0, height_um=150.0)
    assert result["scale_um_per_px"] == pytest.approx(2.0, rel=0.02)


def test_measure_stack_reports_an_accelerating_trend() -> None:
    rng = np.random.default_rng(5)
    frames = []
    for index in range(90):
        speed = 4.0 + 0.12 * index
        frames.append(train_frame(index * speed * 0.5, 0.55, 60.0))
    stack = (np.stack(frames) + rng.normal(0.0, 0.3, (90, 200, 420))).astype(np.float32)
    image = measure_stack(stack, rate=320.0, segments=3)
    speeds = [abs(item["px_per_frame"]) for item in image.velocity_segments]
    assert len(speeds) == 3
    assert speeds[-1] > speeds[0] * 1.3


def test_report_without_bench_inputs_names_what_is_missing() -> None:
    image = fake_image()
    report = build_report(image, 70.0, 20.0)
    assert report["missing"]
    assert any("scale" in item for item in report["missing"])
    assert "cross_section_from_cubaud" not in report["derived"]


def test_report_with_scale_and_duct_produces_a_pump_bound() -> None:
    image = fake_image(velocity_px_per_s=-33333.3)
    report = build_report(image, 70.0, 20.0, scale_um_per_px=2.0,
                          width_um=150.0, height_um=150.0)
    bounds = report["derived"]["pump_ratio_bounds"]
    assert bounds["pump_ratio_low"] < bounds["pump_ratio_high"]
    assert "cross_section_from_cubaud" not in report["derived"]
