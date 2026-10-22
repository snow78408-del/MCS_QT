from __future__ import annotations

import numpy as np
import pytest

from backend.vision.learned_phase import LearnedPhaseProposer, build_network, normalized_strip, probability_intervals, refine_existing_intervals


def test_probability_intervals_reject_partial_and_short_bodies() -> None:
    p = np.zeros(100)
    p[:20] = 1
    p[30:50] = 1
    p[60:65] = 1
    p[80:] = 1
    assert probability_intervals(p) == [(30, 49)]


@pytest.mark.parametrize('data', [np.zeros((3, 4, 2)), np.full((24, 100), np.nan), np.zeros((1, 10))])
def test_normalization_rejects_unusable_images(data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalized_strip(data)


def test_normalization_handles_uniform_and_brightness_shift() -> None:
    assert np.isfinite(normalized_strip(np.zeros((24, 120)))).all()
    source = np.arange(24*120).reshape(24, 120).astype(np.float32)
    assert normalized_strip(source).shape == (24, 720)
    np.testing.assert_allclose(normalized_strip(source), normalized_strip(source+20), atol=1e-6)


@pytest.mark.parametrize('p,threshold', [(np.array([0.,np.nan]), .5), (np.zeros((2,3)), .5), (np.zeros(20), 1.)])
def test_probability_validation(p: np.ndarray, threshold: float) -> None:
    with pytest.raises(ValueError):
        probability_intervals(p, threshold=threshold)


def test_model_loader_rejects_bad_metadata_and_preserves_blank_rejection(tmp_path) -> None:
    torch = pytest.importorskip('torch')
    network = build_network()
    with torch.no_grad():
        for param in network.parameters():
            param.zero_()
        # Deliberately hallucinating network: the image-evidence floor must
        # independently reject empty images, regardless of learned biases.
        network[-1].bias.fill_(10)
    path = tmp_path/'phase.pt'
    checkpoint = dict(schema='rough-phase-cnn-v1', production_authorized=False,
                      threshold=.5, state_dict=network.state_dict())
    torch.save(checkpoint, path)
    proposer = LearnedPhaseProposer(path)
    assert proposer.predict(np.zeros((30, 400), np.uint8)) == []
    checkpoint['production_authorized'] = True
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='authorization'):
        LearnedPhaseProposer(path)


def test_replay_geometry_rejects_outside_and_degenerate_channel() -> None:
    from tools.replay_learned_phase import channel_strip

    image = np.zeros((50, 200), np.uint8)
    geometry = dict(quad=[[10,10],[180,10],[180,40],[10,40]], nx=170, ny=30)
    assert channel_strip(image, geometry).shape == (30,170)
    geometry['quad'][0][0] = -1
    with pytest.raises(ValueError, match='outside'):
        channel_strip(image, geometry)
    geometry['quad'] = [[0,0]]*4
    with pytest.raises(ValueError, match='geometry'):
        channel_strip(image, geometry)


def test_refinement_preserves_candidates_and_abstains_on_ambiguous_or_large_moves() -> None:
    gray = np.zeros((31, 300), np.uint8)
    existing = [(50,100),(150,200)]
    assert refine_existing_intervals(gray, existing, [(53,98),(149,202),(230,270)],30) == [(53,98),(149,202)]
    assert refine_existing_intervals(gray, [], [(53,98)],30) == []
    assert refine_existing_intervals(gray, existing, [(60,110),(53,98),(40,90),(145,230)],30) == existing
    assert refine_existing_intervals(gray, existing, [(53,98),(54,99)],30) == existing


def test_refiner_is_not_called_without_rectified_phase_candidates() -> None:
    from backend.vision.config import DetectorConfig, DebugConfig
    from backend.vision.detector import DropletDetector

    class Refiner:
        def refine(self, *args):
            pytest.fail('Missing geometry/candidates must not invoke learned refinement')

    detector = DropletDetector(DetectorConfig(measurement_mode='generation_plug', generation_phase_mode='transverse_bounded'),
                              DebugConfig(), phase_refiner=Refiner())
    detector.detect(np.zeros((31,300), np.uint8))
    detector.detect(np.zeros((31,300), np.uint8), channel_width_px=30, flow_axis='x')


def _phase_frame() -> np.ndarray:
    """Calibrated shoulder ordering with three carrier-separated droplets."""
    x = np.arange(360, dtype=float)
    signal = np.full(x.shape, -24.0)
    for left, right in [(45, 105), (155, 215), (265, 325)]:
        signal += 48 * (1 / (1 + np.exp(-(x - left) / 4))
                        - 1 / (1 + np.exp(-(x - right) / 4)))
    frame = np.full((32, 360), 128.0)
    frame[2:8] -= signal / 2
    frame[24:30] += signal / 2
    return np.rint(frame).astype(np.uint8)


def test_predict_rejects_degenerate_shape_before_normalization(tmp_path) -> None:
    torch = pytest.importorskip('torch')
    network = build_network()
    with torch.no_grad():
        for param in network.parameters():
            param.zero_()
    path = tmp_path/'phase.pt'
    checkpoint = dict(schema='rough-phase-cnn-v1', production_authorized=False,
                      threshold=.5, state_dict=network.state_dict())
    torch.save(checkpoint, path)
    proposer = LearnedPhaseProposer(path)
    # Degenerate 2D shapes return [] instead of raising: the shape/contrast
    # rejection runs before resize/normalization.
    assert proposer.predict(np.zeros((1, 100), np.uint8)) == []
    assert proposer.predict(np.zeros((24, 50), np.uint8)) == []
    # Non-2D input still raises rather than silently proposing nothing.
    with pytest.raises(ValueError):
        proposer.predict(np.zeros((3, 4, 2), np.uint8))


def test_detector_rebounds_aggressive_refiner() -> None:
    from backend.vision.config import DetectorConfig, DebugConfig
    from backend.vision.detector import DropletDetector

    class MergingRefiner:
        def refine(self, gray, existing, reference_width_px):
            if not existing:
                return []
            # An injected refiner must not be able to merge separate bodies.
            return [(existing[0][0], existing[-1][1])]

    detector = DropletDetector(DetectorConfig(measurement_mode='generation_plug', generation_phase_mode='transverse_bounded'),
                              DebugConfig(), phase_refiner=MergingRefiner())
    result = detector.detect(_phase_frame(), channel_width_px=31, flow_axis='x')
    # The detector re-check keeps the original three separate droplets.
    assert len(result.plug_lengths_px) == 3


def test_refiner_does_not_change_diameter_valid() -> None:
    from backend.vision.config import DetectorConfig, DebugConfig
    from backend.vision.detector import DropletDetector

    class InwardRefiner:
        def refine(self, gray, existing, reference_width_px):
            return [(a + 1, b - 1) for a, b in existing]

    base = DropletDetector(DetectorConfig(measurement_mode='generation_plug', generation_phase_mode='transverse_bounded'),
                          DebugConfig())
    refined = DropletDetector(DetectorConfig(measurement_mode='generation_plug', generation_phase_mode='transverse_bounded'),
                             DebugConfig(), phase_refiner=InwardRefiner())
    frame = _phase_frame()
    base_result = base.detect(frame, channel_width_px=31, flow_axis='x')
    refined_result = refined.detect(frame, channel_width_px=31, flow_axis='x')
    assert len(base_result.plug_lengths_px) == len(refined_result.plug_lengths_px) == 3
    # Shadow proposals cannot alter measurements or state used downstream.
    assert base_result.diameter_valid == refined_result.diameter_valid
    assert base_result.plug_lengths_px == refined_result.plug_lengths_px
    assert base_result.equivalent_diameters_px == refined_result.equivalent_diameters_px
    assert base._phase_length_history == refined._phase_length_history


def test_shadow_cannot_mutate_input_or_history_across_frames() -> None:
    from backend.vision.config import DetectorConfig, DebugConfig
    from backend.vision.detector import DropletDetector

    class MutatingRefiner:
        def refine(self, gray, existing, reference_width_px):
            proposals = [(a + 1, b - 1) for a, b in existing]
            existing.clear()
            gray[:] = 0
            return proposals

    def make(refiner=None):
        return DropletDetector(DetectorConfig(measurement_mode='generation_plug',
            generation_phase_mode='transverse_bounded'), DebugConfig(), phase_refiner=refiner)

    base, shadow = make(), make(MutatingRefiner())
    for shift in [0, 1, 3, 5]:
        frame = np.roll(_phase_frame(), shift, axis=1)
        before = frame.copy()
        trace = {}
        a = base.detect(frame, channel_width_px=31, flow_axis='x')
        b = shadow.detect(frame, channel_width_px=31, flow_axis='x', trace=trace)
        assert a.plug_lengths_px == b.plug_lengths_px
        assert a.equivalent_diameters_px == b.equivalent_diameters_px
        assert base._phase_length_history == shadow._phase_length_history
        np.testing.assert_array_equal(frame, before)
        np.testing.assert_array_equal(a.helper_mask, b.helper_mask)
        assert trace['phase_diagnostics']['learned_endpoint_refinement']['mode'] == 'shadow'
