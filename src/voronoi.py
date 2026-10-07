"""Діаграма Вороного як дуальний граф тріангуляції Делоне.

Кожному ребру Делоне ab відповідає рівно одне ребро Вороного — частина
серединного перпендикуляра до ab, що розділяє області Вороного точок a і b:

    обидві грані ab — трикутники  → відрізок між центрами описаних кіл;
    одна грань — зовнішня         → промінь з центру кола назовні оболонки;
    обидві грані зовнішні         → уся пряма (усі точки S колінеарні).

Вершини Вороного — центри описаних кіл трикутників Делоне.
Перетворення займає O(N): кожне ребро розглядається один раз,
кожен трикутник — сталу кількість разів.

Модуль потрібен лише для візуалізації, тому координати вершин — float.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from geometry import Point, ccw, circumcenter
from quadedge import Edge, iter_edges

Vec = tuple[float, float]

@dataclass(frozen=True)
class VoronoiEdge:
    """Ребро Вороного між областями точок a і b.
    kind = "segment": відрізок start → end;
    kind = "ray":     промінь start + t·direction, t ≥ 0;
    kind = "line":    пряма   start + t·direction, t ∈ ℝ.
    """
    a: Point
    b: Point
    kind: str
    start: Vec
    end: Optional[Vec] = None
    direction: Optional[Vec] = None


def _left_triangle(e: Edge) -> Optional[tuple[Point, Point, Point]]:
    """Трикутник зліва від e або None, якщо зліва зовнішня грань.
    Внутрішня грань тріангуляції — цикл lnext довжини 3 з лівим поворотом.
    Зовнішня грань обходиться за годинниковою стрілкою (ccw < 0), а у
    виродженому випадку колінеарних точок цикл довший за 3.
    """
    e2 = e.lnext
    e3 = e2.lnext
    if e3.lnext is not e:
        return None
    a, b, c = e.org, e.dest, e2.dest
    if ccw(a, b, c) <= 0:
        return None
    return a, b, c

def triangulation_edges(start: Edge) -> list[tuple[Point, Point]]:
    """Усі ребра тріангуляції як пари точок, кожне рівно один раз."""
    return [(e.org, e.dest) for e in iter_edges(start)]

def triangles(start: Edge) -> list[tuple[Point, Point, Point]]:
    """Усі трикутники тріангуляції (вершини проти год. стрілки), кожен один раз."""
    seen: set[frozenset[Point]] = set()
    out = []
    for e in iter_edges(start):
        for d in (e, e.sym):
            t = _left_triangle(d)
            if t is not None:
                key = frozenset(t)
                if key not in seen:
                    seen.add(key)
                    out.append(t)
    return out

def voronoi_edges(start: Edge) -> list[VoronoiEdge]:
    """Ребра діаграми Вороного, побудовані з тріангуляції Делоне за O(N)."""
    centers: dict[frozenset[Point], Vec] = {}

    def center_left_of(e: Edge) -> Optional[Vec]:
        t = _left_triangle(e)
        if t is None:
            return None
        key = frozenset(t)
        if key not in centers:
            centers[key] = circumcenter(*t)
        return centers[key]

    out: list[VoronoiEdge] = []
    for e in iter_edges(start):
        a, b = e.org, e.dest
        dx, dy = b.x - a.x, b.y - a.y
        cl = center_left_of(e)          # трикутник зліва від a → b
        cr = center_left_of(e.sym)      # трикутник справа від a → b

        if cl is not None and cr is not None:
            out.append(VoronoiEdge(a, b, "segment", cl, end=cr))
        elif cl is not None:
            # справа зовнішня грань: промінь іде праворуч від a → b
            out.append(VoronoiEdge(a, b, "ray", cl, direction=(dy, -dx)))
        elif cr is not None:
            # зліва зовнішня грань: промінь іде ліворуч від a → b
            out.append(VoronoiEdge(a, b, "ray", cr, direction=(-dy, dx)))
        else:
            mid = ((a.x + b.x) / 2, (a.y + b.y) / 2)
            out.append(VoronoiEdge(a, b, "line", mid, direction=(-dy, dx)))
    return out

def _clip(p: Vec, d: Vec, t0: float, t1: float,
          box: tuple[float, float, float, float]) -> Optional[tuple[Vec, Vec]]:
    """Відсікання параметричного відрізка p + t·d, t ∈ [t0, t1], прямокутником
    box = (xmin, ymin, xmax, ymax) за алгоритмом Ляна–Барські."""
    xmin, ymin, xmax, ymax = box
    for q, r in ((-d[0], p[0] - xmin), (d[0], xmax - p[0]),
                 (-d[1], p[1] - ymin), (d[1], ymax - p[1])):
        if q == 0:
            if r < 0:
                return None             # паралельно межі й повністю зовні
            continue
        t = r / q
        if q < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return ((p[0] + t0 * d[0], p[1] + t0 * d[1]),
            (p[0] + t1 * d[0], p[1] + t1 * d[1]))

def voronoi_segments(start: Edge,
                     box: tuple[float, float, float, float]) -> list[tuple[Vec, Vec]]:
    """Ребра Вороного, обрізані прямокутником box, — готові відрізки для малювання.
    Ребра нульової довжини (дуальні до діагоналі коциклічної четвірки) пропускаються.
    """
    inf = float("inf")
    out = []
    for v in voronoi_edges(start):
        if v.kind == "segment":
            assert v.end is not None
            d = (v.end[0] - v.start[0], v.end[1] - v.start[1])
            if d == (0.0, 0.0):
                continue
            seg = _clip(v.start, d, 0.0, 1.0, box)
        elif v.kind == "ray":
            assert v.direction is not None
            seg = _clip(v.start, v.direction, 0.0, inf, box)
        else:
            assert v.direction is not None
            seg = _clip(v.start, v.direction, -inf, inf, box)
        if seg is not None:
            out.append(seg)
    return out
