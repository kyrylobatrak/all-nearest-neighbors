"""Усі найближчі сусіди через тріангуляцію Делоне.

Теорема. Якщо q — найближчий сусід p, то pq — ребро будь-якої тріангуляції
Делоне множини S.
Доведення: у крузі з діаметром pq будь-яка точка r ≠ q (включно з межею)
задовольняє |pr| < |pq|, тобто була б ближчою до p. Отже круг порожній,
pq — ребро графа Габріеля, а він є підграфом тріангуляції Делоне.

Тому для кожної точки досить переглянути лише її сусідів у тріангуляції.
Сумарно це Σ deg(p) = 2E ≤ 2(3N − 6) порівнянь, тобто O(N).
"""

from __future__ import annotations
from typing import Optional, Sequence
from delaunay import delaunay
from geometry import Point, dist2
from quadedge import iter_edges


def all_nearest_neighbors(points: Sequence[Point]) -> list[Optional[int]]:
    """Для кожної точки повертає індекс її найближчого сусіда, за O(N log N).
    Інтерфейс і результат такі самі, як у brute_force.all_nearest_neighbors_brute:
    result[i] = j, j != i; серед рівновіддалених сусідів — менший індекс;
    дублікати — сусіди одне одного; для N = 1 результат [None].
    """
    n = len(points)
    result: list[Optional[int]] = [None] * n
    if n < 2:
        return result

    # ---------- 1. Дублікати, O(N) ----------
    # groups[p] — індекси всіх входжень точки p у порядку зростання
    groups: dict[Point, list[int]] = {}
    for i, p in enumerate(points):
        groups.setdefault(p, []).append(i)

    for idx in groups.values():
        if len(idx) > 1:
            # найближчий сусід — інша копія на відстані 0, з найменшим індексом
            first, second = idx[0], idx[1]
            for i in idx:
                result[i] = second if i == first else first

    unique = list(groups)
    if len(unique) == 1:
        return result

    # представник точки — її перше входження (найменший індекс)
    rep = {p: idx[0] for p, idx in groups.items()}

    # ---------- 2. Тріангуляція Делоне, O(N log N) ----------
    start = delaunay(unique)

    # ---------- 3. Найближчий серед сусідів у тріангуляції, O(N) ----------
    # best[p] = (квадрат відстані, індекс сусіда)
    best: dict[Point, tuple[int, int]] = {}
    for e in iter_edges(start):
        for a, b in ((e.org, e.dest), (e.dest, e.org)):
            cand = (dist2(a, b), rep[b])
            if a not in best or cand < best[a]:
                best[a] = cand

    for p, idx in groups.items():
        if len(idx) == 1:
            result[idx[0]] = best[p][1]

    return result