"""Юніт-тести геометричних примітивів"""

import math
import random
import pytest

from geometry import Point, ccw, circumcenter, dist2, in_circle


# ---------- Point ----------

def test_point_lexicographic_order():
    pts = [Point(2, 1), Point(1, 5), Point(1, 2), Point(0, 9)]
    assert sorted(pts) == [Point(0, 9), Point(1, 2), Point(1, 5), Point(2, 1)]


def test_point_hashable_and_equal():
    assert Point(3, 4) == Point(3, 4)
    assert len({Point(3, 4), Point(3, 4), Point(4, 3)}) == 2


# ---------- dist2 ----------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (Point(0, 0), Point(3, 4), 25),
        (Point(-1, -1), Point(2, 3), 25),
        (Point(5, 5), Point(5, 5), 0),
    ],
)
def test_dist2(a, b, expected):
    assert dist2(a, b) == expected
    assert dist2(b, a) == expected


# ---------- ccw ----------

def test_ccw_left_right_collinear():
    a, b = Point(0, 0), Point(10, 0)
    assert ccw(a, b, Point(5, 1)) > 0      # зліва
    assert ccw(a, b, Point(5, -1)) < 0     # справа
    assert ccw(a, b, Point(20, 0)) == 0    # на прямій, за b
    assert ccw(a, b, Point(-3, 0)) == 0    # на прямій, перед a


def test_ccw_is_twice_signed_area():
    assert ccw(Point(0, 0), Point(4, 0), Point(0, 3)) == 12


def test_ccw_antisymmetry():
    rnd = random.Random(1)
    for _ in range(1000):
        a, b, c = (Point(rnd.randint(-50, 50), rnd.randint(-50, 50)) for _ in range(3))
        assert ccw(a, b, c) == -ccw(b, a, c)
        assert ccw(a, b, c) == ccw(b, c, a) == ccw(c, a, b)


def test_ccw_exact_on_huge_coordinates():
    # Колінеарні точки з координатами ~10^18: float тут помилився б, int — ні
    big = 10**18
    assert ccw(Point(0, 0), Point(big, big + 1), Point(2 * big, 2 * big + 2)) == 0
    assert ccw(Point(0, 0), Point(big, big + 1), Point(2 * big, 2 * big + 3)) > 0


# ---------- in_circle ----------

# Коло x² + y² = 25, трикутник у порядку проти годинникової стрілки
A, B, C = Point(5, 0), Point(0, 5), Point(-5, 0)


def test_in_circle_inside_outside_on():
    assert ccw(A, B, C) > 0
    assert in_circle(A, B, C, Point(0, 0)) > 0     # центр
    assert in_circle(A, B, C, Point(1, -2)) > 0    # всередині
    assert in_circle(A, B, C, Point(10, 10)) < 0   # зовні
    assert in_circle(A, B, C, Point(0, -5)) == 0   # на колі
    assert in_circle(A, B, C, Point(3, 4)) == 0    # на колі (3²+4²=25)
    assert in_circle(A, B, C, Point(-4, -3)) == 0  # на колі


def test_in_circle_sign_flips_for_clockwise():
    d = Point(0, 0)
    assert in_circle(A, B, C, d) > 0
    assert in_circle(C, B, A, d) < 0


def test_in_circle_invariant_under_rotation_of_triangle():
    rnd = random.Random(2)
    for _ in range(1000):
        a, b, c, d = (Point(rnd.randint(-30, 30), rnd.randint(-30, 30)) for _ in range(4))
        v = in_circle(a, b, c, d)
        assert in_circle(b, c, a, d) == v
        assert in_circle(c, a, b, d) == v


def test_in_circle_matches_geometric_definition():
    """Звірка знаку з прямим обчисленням через центр і радіус."""
    rnd = random.Random(3)
    checked = 0
    while checked < 2000:
        a, b, c, d = (Point(rnd.randint(-100, 100), rnd.randint(-100, 100)) for _ in range(4))
        o = ccw(a, b, c)
        if o == 0:
            continue
        cx, cy = circumcenter(a, b, c)
        r2 = (a.x - cx) ** 2 + (a.y - cy) ** 2
        d2 = (d.x - cx) ** 2 + (d.y - cy) ** 2
        if math.isclose(r2, d2, rel_tol=1e-9, abs_tol=1e-9):
            continue  # майже на колі — float-еталон тут ненадійний
        expected_inside = d2 < r2
        v = in_circle(a, b, c, d) if o > 0 else -in_circle(a, b, c, d)
        assert (v > 0) == expected_inside
        checked += 1


def test_in_circle_cocircular_grid_points():
    # 4 вершини квадрата завжди коциклічні
    assert in_circle(Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)) == 0


# ---------- circumcenter ----------

def test_circumcenter_known():
    assert circumcenter(A, B, C) == (0.0, 0.0)
    assert circumcenter(Point(0, 0), Point(4, 0), Point(0, 4)) == (2.0, 2.0)


def test_circumcenter_equidistant():
    rnd = random.Random(4)
    for _ in range(500):
        a, b, c = (Point(rnd.randint(-100, 100), rnd.randint(-100, 100)) for _ in range(3))
        if ccw(a, b, c) == 0:
            continue
        ux, uy = circumcenter(a, b, c)
        ra = math.hypot(a.x - ux, a.y - uy)
        rb = math.hypot(b.x - ux, b.y - uy)
        rc = math.hypot(c.x - ux, c.y - uy)
        assert math.isclose(ra, rb, rel_tol=1e-9) and math.isclose(ra, rc, rel_tol=1e-9)


def test_circumcenter_collinear_raises():
    with pytest.raises(ValueError):
        circumcenter(Point(0, 0), Point(1, 1), Point(2, 2))
