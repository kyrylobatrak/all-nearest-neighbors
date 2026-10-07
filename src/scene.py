"""Підготовка всього, що малює інтерфейс, без залежності від Qt.

compute_scene запускає основний алгоритм (nearest.all_nearest_neighbors) і
окремо будує тріангуляцію Делоне та діаграму Вороного для відмалювання.
Час вимірюється лише для основного алгоритму — саме його складність
O(N log N) заявлено у звіті.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence
from delaunay import delaunay
from geometry import Point
from nearest import all_nearest_neighbors
from voronoi import Vec, triangles, triangulation_edges, voronoi_segments

@dataclass
class Scene:
    points: list[Point]
    nn: list[Optional[int]]
    ann_ms: float                                   # час all_nearest_neighbors, мс
    unique: int                                     # кількість різних точок
    tri_edges: list[tuple[Point, Point]] = field(default_factory=list)
    n_triangles: int = 0
    vor_segments: list[tuple[Vec, Vec]] = field(default_factory=list)
    mutual: list[tuple[int, int]] = field(default_factory=list)    # i < j, nn(i)=j, nn(j)=i
    nn_dist: list[float] = field(default_factory=list)             # |p, nn(p)|

def compute_scene(points: Sequence[Point], size: int, voronoi_margin: float = 1.0) -> Scene:
    """Розв'язує задачу й готує геометрію для малювання.

    Ребра Вороного обрізаються квадратом, ширшим за робочу область
    [0, size]² на voronoi_margin·size з кожного боку, щоб промені
    не обривалися під час зсуву й масштабування.
    """
    pts = list(points)
    t0 = time.perf_counter()
    nn = all_nearest_neighbors(pts)
    ann_ms = (time.perf_counter() - t0) * 1000

    uniq = list(dict.fromkeys(pts))
    scene = Scene(points=pts, nn=nn, ann_ms=ann_ms, unique=len(uniq))

    for i, j in enumerate(nn):
        if j is None:
            scene.nn_dist.append(math.nan)
            continue
        p, q = pts[i], pts[j]
        scene.nn_dist.append(math.hypot(p.x - q.x, p.y - q.y))
        if i < j and nn[j] == i:
            scene.mutual.append((i, j))

    if len(uniq) >= 2:
        start = delaunay(uniq)
        m = voronoi_margin * size
        scene.tri_edges = triangulation_edges(start)
        scene.n_triangles = len(triangles(start))
        scene.vor_segments = voronoi_segments(start, (-m, -m, size + m, size + m))
    return scene
