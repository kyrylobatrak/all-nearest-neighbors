"""
Усі предикати працюють на цілих координатах і використовують точну
арифметику Python int, тому результат не залежить від похибок округлення.
"""

from __future__ import annotations
from typing import NamedTuple


class Point(NamedTuple):
    """Точка площини з цілими координатами.
    NamedTuple дає незмінність, хешування (для словників/множин)
    і лексикографічне порівняння (x, y) «з коробки».
    """

    x: int
    y: int


def dist2(a: Point, b: Point) -> int:
    """Квадрат евклідової відстані |ab|²
    sqrt тут не потрібен, бо функція √t монотонно зростає:
    |ab| < |ac|   ⇔   |ab|² < |ac|²
    """

    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def ccw(a: Point, b: Point, c: Point) -> int:
    """Орієнтація трійки (a, b, c).
    > 0 — лівий поворот (проти годинникової стрілки),
    < 0 — правий поворот (за годинниковою стрілкою),
    = 0 — точки колінеарні.
    Значення дорівнює подвоєній орієнтованій площі трикутника abc.
    """
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def in_circle(a: Point, b: Point, c: Point, d: Point) -> int:
    """Положення точки d відносно кола, описаного навколо a, b, c.
    Для a, b, c у порядку проти годинникової стрілки (ccw(a, b, c) > 0):
    > 0 — d строго всередині кола,
    < 0 — d строго зовні,
    = 0 — d на колі (коциклічні точки).
    Для порядку за годинниковою стрілкою знак змінюється на протилежний.
    Обчислюється як визначник 3x3 після зсуву всіх точок на -d.
    """
    adx, ady = a.x - d.x, a.y - d.y
    bdx, bdy = b.x - d.x, b.y - d.y
    cdx, cdy = c.x - d.x, c.y - d.y

    ad = adx * adx + ady * ady
    bd = bdx * bdx + bdy * bdy
    cd = cdx * cdx + cdy * cdy

    return (
        adx * (bdy * cd - bd * cdy)
        - ady * (bdx * cd - bd * cdx)
        + ad * (bdx * cdy - bdy * cdx)
    )


def circumcenter(a: Point, b: Point, c: Point) -> tuple[float, float]:
    """Центр кола, описаного навколо трикутника abc (вершина Вороного).
    Використовується лише для візуалізації діаграми Вороного, тому
    повертає float. Для колінеарних точок центру не існує → ValueError.
    """
    d = 2 * ccw(a, b, c)
    if d == 0:
        raise ValueError("Колінеарні точки не мають описаного кола")

    a2 = a.x * a.x + a.y * a.y
    b2 = b.x * b.x + b.y * b.y
    c2 = c.x * c.x + c.y * c.y

    ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d
    uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d
    return ux, uy
