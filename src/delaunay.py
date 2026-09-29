"""Побудова тріангуляції Делоне методом «розділяй і володарюй» (Guibas & Stolfi, 1985).

Рекурсія:
    1. точки відсортовано за (x, y);
    2. множина ділиться навпіл за медіаною;
    3. тріангуляції половин будуються рекурсивно;
    4. злиття: знаходиться нижня спільна дотична, далі половини
       «зшиваються» знизу вгору; ребра, що порушують умову Делоне
       (порожнього кола), видаляються.

Складність: T(N) = 2T(N/2) + O(N) = O(N log N).
"""

from __future__ import annotations
from typing import Sequence
from geometry import Point, ccw, in_circle
from quadedge import Edge, connect, delete_edge, make_edge, splice

def _left_of(p: Point, e: Edge) -> bool:
    """Точка p строго зліва від напрямленого ребра e."""
    return ccw(p, e.org, e.dest) > 0

def _right_of(p: Point, e: Edge) -> bool:
    """Точка p строго справа від напрямленого ребра e."""
    return ccw(p, e.dest, e.org) > 0

def _build(pts: Sequence[Point], lo: int, hi: int) -> tuple[Edge, Edge]:
    """Тріангуляція Делоне точок pts[lo:hi] (відсортованих, без дублікатів, щонайменше 2).
    Повертає (ldo, rdo):
        ldo — ребро опуклої оболонки проти год. стрілки, що виходить з найлівішої точки;
        rdo — ребро опуклої оболонки за год. стрілкою, що виходить з найправішої точки.
    """
    n = hi - lo
    # ---------- база рекурсії ----------
    if n == 2:
        a = make_edge(pts[lo], pts[lo + 1])
        return a, a.sym

    if n == 3:
        p1, p2, p3 = pts[lo], pts[lo + 1], pts[lo + 2]
        a = make_edge(p1, p2)
        b = make_edge(p2, p3)
        splice(a.sym, b)
        o = ccw(p1, p2, p3)
        if o > 0:                       # лівий поворот: трикутник p1 p2 p3
            connect(b, a)
            return a, b.sym
        if o < 0:                       # правий поворот: трикутник p1 p3 p2
            c = connect(b, a)
            return c.sym, c
        return a, b.sym                 # колінеарні: ланцюжок з двох ребер

    # ---------- розділення ----------
    mid = (lo + hi) // 2
    ldo, ldi = _build(pts, lo, mid)
    rdi, rdo = _build(pts, mid, hi)

    # ---------- злиття, крок 1: нижня спільна дотична ----------
    # ldi рухається по оболонці лівої частини, rdi — правої,
    # доки відрізок між ними не залишить обидві оболонки над собою.
    while True:
        if _left_of(rdi.org, ldi):
            ldi = ldi.lnext
        elif _right_of(ldi.org, rdi):
            rdi = rdi.rprev
        else:
            break

    # базове ребро: від правої частини до лівої
    basel = connect(rdi.sym, ldi)
    if ldi.org == ldo.org:
        ldo = basel.sym
    if rdi.org == rdo.org:
        rdo = basel

    # ---------- злиття, крок 2: «застібка» знизу вгору ----------
    def valid(e: Edge) -> bool:
        """Кандидат лежить над базовим ребром."""
        return _right_of(e.dest, basel)

    while True:
        # кандидат з лівої частини; видаляємо ребра, що порушують умову Делоне
        lcand = basel.sym.onext
        if valid(lcand):
            while in_circle(basel.dest, basel.org, lcand.dest, lcand.onext.dest) > 0:
                t = lcand.onext
                delete_edge(lcand)
                lcand = t

        # кандидат з правої частини
        rcand = basel.oprev
        if valid(rcand):
            while in_circle(basel.dest, basel.org, rcand.dest, rcand.oprev.dest) > 0:
                t = rcand.oprev
                delete_edge(rcand)
                rcand = t

        lv, rv = valid(lcand), valid(rcand)
        if not lv and not rv:           # дійшли до верхньої спільної дотичної
            break

        # нове ребро «застібки»: обираємо кандидата, чиє коло порожнє
        if not lv or (rv and in_circle(lcand.dest, lcand.org, rcand.org, rcand.dest) > 0):
            basel = connect(rcand, basel.sym)
        else:
            basel = connect(basel.sym, lcand.sym)

    return ldo, rdo

def delaunay(points: Sequence[Point]) -> Edge:
    """Тріангуляція Делоне множини різних точок (щонайменше 2).
    Повертає будь-яке ребро результату; решту можна обійти через
    quadedge.iter_edges. Точки сортуються всередині.
    """
    pts = sorted(points)
    if len(pts) < 2:
        raise ValueError("потрібно щонайменше 2 точки")
    for a, b in zip(pts, pts[1:]):
        if a == b:
            raise ValueError(f"дублікат точки {a}; дублікати треба прибрати заздалегідь")
    ldo, _ = _build(pts, 0, len(pts))
    return ldo