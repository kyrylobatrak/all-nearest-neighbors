"""Тести побудови діаграми Вороного з тріангуляції Делоне."""

import math
import random
import pytest
from delaunay import delaunay
from geometry import Point, dist2
from quadedge import iter_edges
from voronoi import (_clip, triangles, triangulation_edges, voronoi_edges,
                     voronoi_segments)

def uniform(n, seed, hi=1000):
    r = random.Random(seed)
    return list(dict.fromkeys(Point(r.randint(0, hi), r.randint(0, hi)) for _ in range(n)))

def d(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def assert_bisector_point(v, c, tol=1e-6):
    """Точка c рівновіддалена від a і b (лежить на серединному перпендикулярі)."""
    da, db = d(c, v.a), d(c, v.b)
    assert abs(da - db) <= tol * max(1.0, da)

# ---------- базові випадки ----------

def test_two_points_give_one_line():
    vs = voronoi_edges(delaunay([Point(0, 0), Point(4, 0)]))
    assert len(vs) == 1
    v = vs[0]
    assert v.kind == "line"
    assert v.start == (2.0, 0.0)
    # напрям перпендикулярний до ab
    assert v.direction[0] * 4 + v.direction[1] * 0 == 0

def test_triangle_gives_three_rays_from_circumcenter():
    pts = [Point(0, 0), Point(6, 0), Point(0, 8)]
    vs = voronoi_edges(delaunay(pts))
    assert len(vs) == 3
    assert all(v.kind == "ray" for v in vs)
    # центр описаного кола прямокутного трикутника — середина гіпотенузи
    assert all(v.start == pytest.approx((3.0, 4.0)) for v in vs)

@pytest.mark.parametrize("pts", [
    [Point(0, 0), Point(6, 0), Point(0, 8)],
    [Point(0, 0), Point(0, 8), Point(6, 0)],
    [Point(5, 1), Point(-3, 7), Point(2, -9)],
])
def test_rays_point_away_from_opposite_vertex(pts):
    """Промінь іде від трикутника: третя вершина лежить з протилежного боку від ребра."""
    for v in voronoi_edges(delaunay(pts)):
        third = next(p for p in pts if p not in (v.a, v.b))
        ab = (v.b.x - v.a.x, v.b.y - v.a.y)
        side_dir = ab[0] * v.direction[1] - ab[1] * v.direction[0]
        side_third = ab[0] * (third.y - v.a.y) - ab[1] * (third.x - v.a.x)
        assert side_dir * side_third < 0

def test_collinear_gives_parallel_lines():
    pts = [Point(x, 2 * x + 1) for x in range(6)]
    vs = voronoi_edges(delaunay(pts))
    assert len(vs) == 5
    assert all(v.kind == "line" for v in vs)

# ---------- властивості на випадкових наборах ----------

@pytest.mark.parametrize("seed", range(20))
def test_every_voronoi_edge_lies_on_bisector(seed):
    pts = uniform(200, seed)
    for v in voronoi_edges(delaunay(pts)):
        assert_bisector_point(v, v.start)
        if v.kind == "segment":
            assert_bisector_point(v, v.end)

@pytest.mark.parametrize("seed", range(10))
def test_voronoi_vertices_have_empty_circles(seed):
    """Вершина Вороного рівновіддалена від трьох точок і жодна інша не ближча."""
    pts = uniform(120, seed)
    for t in triangles(delaunay(pts)):
        from geometry import circumcenter
        c = circumcenter(*t)
        r = d(c, t[0])
        assert all(d(c, p) >= r * (1 - 1e-9) for p in pts)

@pytest.mark.parametrize("seed", range(10))
def test_counts_match_euler(seed):
    """Ребер Вороного стільки ж, скільки ребер Делоне; трикутників 2N − 2 − h."""
    pts = uniform(300, seed)
    start = delaunay(pts)
    edges = triangulation_edges(start)
    assert len(voronoi_edges(start)) == len(edges)
    rays = sum(v.kind == "ray" for v in voronoi_edges(start))   # h — ребер оболонки
    assert len(triangles(start)) == 2 * len(pts) - 2 - rays

def test_grid_cocircular_zero_length_edges_skipped():
    pts = [Point(x, y) for x in range(5) for y in range(5)]
    segs = voronoi_segments(delaunay(pts), (-10, -10, 20, 20))
    assert all(a != b for a, b in segs)

# ---------- відсікання ----------

def test_clip_segment_inside_and_outside():
    box = (0, 0, 10, 10)
    assert _clip((1, 1), (2, 2), 0, 1, box) == ((1, 1), (3, 3))
    assert _clip((20, 20), (1, 0), 0, 1, box) is None

def test_clip_ray_and_line():
    box = (0, 0, 10, 10)
    inf = float("inf")
    assert _clip((5, 5), (1, 0), 0, inf, box) == ((5, 5), (10, 5))
    assert _clip((5, 5), (0, 1), -inf, inf, box) == ((5, 0), (5, 10))

def test_segments_inside_box():
    box = (-100.0, -100.0, 1100.0, 1100.0)
    for (x1, y1), (x2, y2) in voronoi_segments(delaunay(uniform(150, 3)), box):
        for x, y in ((x1, y1), (x2, y2)):
            assert box[0] - 1e-6 <= x <= box[2] + 1e-6
            assert box[1] - 1e-6 <= y <= box[3] + 1e-6