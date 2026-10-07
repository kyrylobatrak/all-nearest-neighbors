"""Тести підготовки сцени для інтерфейсу."""

import math
from brute_force import all_nearest_neighbors_brute
from generator import generate
from geometry import Point
from scene import compute_scene

def test_empty_and_single():
    s = compute_scene([], 100)
    assert s.nn == [] and s.tri_edges == []
    s = compute_scene([Point(1, 1)], 100)
    assert s.nn == [None] and math.isnan(s.nn_dist[0])

def test_matches_brute_force_and_mutual_pairs():
    pts = generate("uniform", 300, seed=7, size=10_000)
    s = compute_scene(pts, 10_000)
    assert s.nn == all_nearest_neighbors_brute(pts)
    for i, j in s.mutual:
        assert i < j and s.nn[i] == j and s.nn[j] == i
    # найближча пара всієї множини завжди взаємна
    assert len(s.mutual) >= 1

def test_duplicates_are_neighbours_at_zero():
    pts = [Point(0, 0), Point(5, 5), Point(0, 0)]
    s = compute_scene(pts, 10)
    assert s.unique == 2
    assert s.nn[0] == 2 and s.nn[2] == 0
    assert s.nn_dist[0] == 0.0
    assert len(s.tri_edges) == 1

def test_euler_counts():
    pts = generate("gauss", 500, seed=1, size=10_000, unique=True)
    s = compute_scene(pts, 10_000)
    n = len(pts)
    # E = 3N − 3 − h, T = 2N − 2 − h  ⇒  E − T = N − 1
    assert len(s.tri_edges) - s.n_triangles == n - 1