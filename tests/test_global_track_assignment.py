from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from backend.vision.config import TrackerConfig
from backend.vision.kalman_tracker import KalmanTracker
from backend.vision.nearest_tracker import NearestTracker
from backend.vision.tracker import radius_gated_match


def match(xs, ys, radii=None):
    return radius_gated_match(
        [np.array([x, 0.0]) for x in xs], [5.0] * len(xs),
        [np.array([y, 0.0]) for y in ys], radii or [5.0] * len(ys),
        max_distance=5.0, min_distance=5.0,
        distance_radius_ratio=1.0, radius_match_ratio=0.5,
    )


def test_competing_nearest_neighbor_does_not_break_second_track():
    # Greedy takes 4 -> 5, leaving 8 unmatched although 4 -> 0, 8 -> 5 works.
    assert match([4, 8], [0, 5]) == ([(0, 0), (1, 1)], [], [])


@pytest.mark.parametrize("tracker_type", [NearestTracker, KalmanTracker])
def test_track_ids_and_confirmation_survive_competing_observations(tracker_type):
    tracker = tracker_type(TrackerConfig(
        match_distance=5.0, min_match_distance=5.0,
        match_distance_radius_ratio=1.0,
    ))
    tracker.update([np.array([4, 0]), np.array([8, 0])], [5, 5], timestamp=0.0)
    result = tracker.update([np.array([0, 0]), np.array([5, 0])], [5, 5], timestamp=0.1)
    assert result.new_track_ids == []
    assert result.matched_pairs == [(1, 0), (2, 1)]
    assert all(t.is_confirmed for t in result.active_tracks)
    assert result.total_count == 2


@pytest.mark.parametrize("xs,ys", [([], [1]), ([1], []), ([1, 2], [100]), ([0], [0, 1, 100])])
def test_unmatched_partition_and_distance_gate(xs, ys):
    pairs, missing, new = match(xs, ys)
    assert sorted([i for i, _ in pairs] + missing) == list(range(len(xs)))
    assert sorted([j for _, j in pairs] + new) == list(range(len(ys)))
    assert all(abs(xs[i] - ys[j]) <= 5 for i, j in pairs)


def test_size_and_nonfinite_candidates_cannot_steal_a_match():
    assert match([0], [0, 1, 2, float("nan")], [50, float("nan"), 5, 5]) == (
        [(0, 2)], [], [0, 1, 3],
    )


def test_random_small_assignments_against_exhaustive_oracle():
    rng = np.random.default_rng(20261022)
    for _ in range(80):
        xs = rng.uniform(0, 20, size=4)
        ys = rng.uniform(0, 20, size=3)
        pairs, _, _ = match(xs, ys)
        actual = (-len(pairs), sum(abs(xs[i] - ys[j]) for i, j in pairs))
        feasible = []
        for assignment in product(range(-1, len(ys)), repeat=len(xs)):
            used = [j for j in assignment if j >= 0]
            if len(set(used)) != len(used):
                continue
            edges = [(i, j) for i, j in enumerate(assignment) if j >= 0]
            if all(abs(xs[i] - ys[j]) <= 5 for i, j in edges):
                feasible.append((-len(edges), sum(abs(xs[i] - ys[j]) for i, j in edges)))
        expected = min(feasible)
        assert actual[0] == expected[0]
        assert actual[1] == pytest.approx(expected[1])
