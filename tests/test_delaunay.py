"""Тести D&C-тріангуляції Делоне та пошуку всіх найближчих сусідів"""

import random
import pytest
from brute_force import all_nearest_neighbors_brute
from delaunay import delaunay
from geometry import Point, ccw, in_circle
from nearest import all_nearest_neighbors
from quadedge import iter_edges

# ---------- генератори наборів точок ----------

def uniform(n, seed, lo=0, hi=1000):
    r = random.Random(seed)
    return [Point(r.randint(lo, hi), r.randint(lo, hi)) for _ in range(n)]

def grid(w, h, step=1):
    # решітка: максимум коциклічних четвірок і колінеарних трійок
    return [Point(x * step, y * step) for x in range(w) for y in range(h)]

def collinear(n, seed):
    r = random.Random(seed)
    xs = r.sample(range(-500, 500), n)
    return [Point(x, 2 * x + 7) for x in xs]

def circle_points(n, radius=5):
    # усі точки решітки на колі x² + y² = r² — всі коциклічні
    pts = [Point(x, y) for x in range(-radius, radius + 1)
           for y in range(-radius, radius + 1) if x * x + y * y == radius * radius]
    return pts[:n] if n else pts

# ---------- перевірка властивостей тріангуляції ----------

def faces(start):
    """Усі грані як орбіти lnext; кожна грань — список орієнтованих ребер."""
    directed = []
    for e in iter_edges(start):
        directed += [e, e.sym]
    seen, out = set(), []
    for e in directed:
        if id(e) in seen:
            continue
        orbit, cur = [], e
        while id(cur) not in seen:
            seen.add(id(cur))
            orbit.append(cur)
            cur = cur.lnext
        out.append(orbit)
    return out

def check_delaunay(points):
    """Перевіряє, що результат — коректна тріангуляція Делоне множини points."""
    pts = sorted(set(points))
    start = delaunay(pts)
    edges = list(iter_edges(start))

    # усі вершини задіяні
    verts = {e.org for e in edges} | {e.dest for e in edges}
    assert verts == set(pts)

    fs = faces(start)
    triangles = [f for f in fs if len(f) == 3 and ccw(f[0].org, f[1].org, f[2].org) > 0]
    others = [f for f in fs if f not in triangles]

    # рівно одна зовнішня грань, решта — трикутники проти год. стрілки
    assert len(others) == 1

    # формула Ейлера для зв'язного планарного графа: V − E + F = 2
    assert len(verts) - len(edges) + len(fs) == 2

    # умова Делоне: жодна точка не лежить строго всередині описаного кола
    for t in triangles:
        a, b, c = t[0].org, t[1].org, t[2].org
        for p in pts:
            assert in_circle(a, b, c, p) <= 0, f"точка {p} всередині кола {a, b, c}"

    # зовнішня грань обходиться за год. стрілкою і є опуклою: жодного лівого повороту
    outer = [e.org for e in others[0]]
    m = len(outer)
    if len(triangles) > 0:
        for i in range(m):
            assert ccw(outer[i], outer[(i + 1) % m], outer[(i + 2) % m]) <= 0

    return len(edges), len(triangles)

# ---------- тріангуляція ----------

def test_two_points():
    assert check_delaunay([Point(0, 0), Point(5, 5)]) == (1, 0)


@pytest.mark.parametrize("pts, edges, tris", [
    ([Point(0, 0), Point(4, 0), Point(0, 3)], 3, 1),            # лівий поворот
    ([Point(0, 0), Point(0, 3), Point(4, 1)], 3, 1),            # правий поворот
    ([Point(0, 0), Point(1, 1), Point(2, 2)], 2, 0),            # колінеарні
])

def test_three_points(pts, edges, tris):
    assert check_delaunay(pts) == (edges, tris)

def test_square_is_split_into_two_triangles():
    # 4 коциклічні точки: будь-яка з двох діагоналей коректна
    assert check_delaunay(grid(2, 2)) == (5, 2)

def test_all_collinear_is_a_chain():
    for seed in range(5):
        pts = collinear(20, seed)
        assert check_delaunay(pts) == (19, 0)

def test_vertical_line():
    pts = [Point(3, y) for y in range(15)]
    assert check_delaunay(pts) == (14, 0)

@pytest.mark.parametrize("w, h", [(3, 3), (4, 5), (7, 7), (10, 3)])
def test_grid(w, h):
    n = w * h
    edges, tris = check_delaunay(grid(w, h))
    hull = 2 * (w + h) - 4          # точки на межі прямокутника
    assert edges == 3 * n - 3 - hull
    assert tris == 2 * n - 2 - hull

def test_cocircular_points():
    check_delaunay(circle_points(0, radius=5))      # 12 точок на одному колі
    check_delaunay(circle_points(0, radius=5) + [Point(0, 0)])

@pytest.mark.parametrize("n, seed", [(4, 0), (5, 1), (8, 2), (17, 3), (50, 4), (120, 5)])
def test_random(n, seed):
    check_delaunay(uniform(n, seed))

def test_many_small_random_sets():
    # малий діапазон координат → багато колінеарних і коциклічних випадків
    for seed in range(300):
        n = 4 + seed % 20
        pts = list(set(uniform(n, seed, 0, 6)))
        if len(pts) >= 2:
            check_delaunay(pts)

def test_duplicates_rejected():
    with pytest.raises(ValueError):
        delaunay([Point(0, 0), Point(1, 1), Point(0, 0)])

# ---------- усі найближчі сусіди: звірка з наївним алгоритмом ----------

def test_ann_edge_cases():
    assert all_nearest_neighbors([]) == []
    assert all_nearest_neighbors([Point(1, 1)]) == [None]
    assert all_nearest_neighbors([Point(0, 0), Point(3, 4)]) == [1, 0]
    assert all_nearest_neighbors([Point(2, 2)] * 3) == [1, 0, 0]

@pytest.mark.parametrize("pts", [
    [Point(0, 0), Point(7, 7), Point(0, 0), Point(100, 100)],      # дублікати
    [Point(5, 5), Point(5, 6), Point(6, 5), Point(5, 4), Point(4, 5)],   # рівновіддалені
    [Point(0, 0), Point(2, 0), Point(5, 0)],                       # несиметричність
    grid(6, 6),
    grid(5, 5, step=3) + grid(5, 5, step=3),                       # кожна точка двічі
    circle_points(0, radius=5) + [Point(0, 0)],
])
def test_ann_matches_brute_force_special(pts):
    assert all_nearest_neighbors(pts) == all_nearest_neighbors_brute(pts)

def test_ann_matches_brute_force_collinear():
    for seed in range(10):
        pts = collinear(30, seed)
        assert all_nearest_neighbors(pts) == all_nearest_neighbors_brute(pts)

def test_ann_matches_brute_force_random():
    for seed in range(500):
        n = 2 + seed % 60
        hi = 10 if seed % 3 == 0 else 10**6      # кожен третій — щільний набір із дублікатами
        pts = uniform(n, seed, 0, hi)
        assert all_nearest_neighbors(pts) == all_nearest_neighbors_brute(pts), f"seed={seed}"

def test_ann_matches_brute_force_large():
    for seed, n in ((1, 1000), (2, 2000)):
        pts = uniform(n, seed, 0, 10**6)
        assert all_nearest_neighbors(pts) == all_nearest_neighbors_brute(pts)