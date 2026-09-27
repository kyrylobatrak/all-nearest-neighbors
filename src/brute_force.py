"""Наївний розв'язок задачі «Усі найближчі сусіди» за O(N²).
Використовується як еталон для перевірки коректності D&C-алгоритму
і як базова лінія для порівняння часу роботи в бенчмарках.
"""

from __future__ import annotations
from typing import Optional, Sequence
from geometry import Point, dist2


def all_nearest_neighbors_brute(points: Sequence[Point]) -> list[Optional[int]]:
    """Для кожної точки повертає індекс її найближчого сусіда.
    result[i] = j означає, що points[j] — найближчий сусід points[i], j != i.
    Якщо рівновіддалених сусідів кілька, береться той, у кого менший індекс.
    Дублікати (однакові координати) є сусідами одне одного з відстанню 0.
    Для N = 1 сусіда немає: result = [None].
    Складність: N(N-1)/2 обчислень dist2, тобто O(N²) часу і O(N) пам'яті.
    """
    n = len(points)
    best_j: list[Optional[int]] = [None] * n
    best_d: list[int] = [0] * n

    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            d = dist2(pi, points[j])
            # Відстань симетрична, тому одна пара (i, j) оновлює обидві точки.
            # Кожна точка бачить кандидатів у порядку зростання індексу,
            # тому строге «<» залишає рівновіддаленого сусіда з меншим індексом.
            if best_j[i] is None or d < best_d[i]:
                best_j[i], best_d[i] = j, d
            if best_j[j] is None or d < best_d[j]:
                best_j[j], best_d[j] = i, d

    return best_j


def nn_distances2(points: Sequence[Point], nn: Sequence[Optional[int]]) -> list[Optional[int]]:
    """Квадрати відстаней до знайдених сусідів.
    Потрібно для порівняння двох алгоритмів: при рівновіддалених сусідах
    індекси можуть відрізнятися, а відстані — ні.
    """
    return [None if j is None else dist2(points[i], points[j]) for i, j in enumerate(nn)]
