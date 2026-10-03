"""Тести генератора точок"""

import pytest
from generator import GENERATORS, generate
from geometry import ccw
from nearest import all_nearest_neighbors
from brute_force import all_nearest_neighbors_brute

KINDS = list(GENERATORS)

@pytest.mark.parametrize("kind", KINDS)
def test_count_and_bounds(kind):
    pts = generate(kind, 500, seed=1, size=10_000)
    assert len(pts) == 500
    assert all(0 <= p.x <= 10_000 and 0 <= p.y <= 10_000 for p in pts)
    assert all(isinstance(p.x, int) and isinstance(p.y, int) for p in pts)

@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_same_points(kind):
    assert generate(kind, 200, seed=42) == generate(kind, 200, seed=42)

@pytest.mark.parametrize("kind", KINDS)
def test_zero_points(kind):
    assert generate(kind, 0, seed=1) == []

def test_unique_removes_duplicates():
    pts = generate("uniform", 300, seed=3, size=10, unique=True)   # у квадраті 11x11 лише 121 точка
    assert len(pts) == len(set(pts)) <= 121

def test_collinear_points_are_on_one_line():
    pts = generate("collinear", 100, seed=5)
    a, b = pts[0], pts[1]
    assert all(ccw(a, b, p) == 0 for p in pts)

def test_grid_points_are_distinct():
    pts = generate("grid", 400, seed=6)
    assert len(set(pts)) == 400

def test_unknown_kind():
    with pytest.raises(ValueError):
        generate("spiral", 10)

@pytest.mark.parametrize("kind", KINDS)
def test_algorithm_on_every_distribution(kind):
    pts = generate(kind, 300, seed=7, size=5_000)
    assert all_nearest_neighbors(pts) == all_nearest_neighbors_brute(pts)