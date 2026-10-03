"""Генерація наборів точок для демонстрації та бенчмарків.
Усі координати цілі й лежать у квадраті [0, size] x [0, size].
За однакового seed результат однаковий, тож експерименти відтворювані.

Розподіли:
    uniform   — рівномірний (типовий випадок);
    gauss     — нормальний навколо центру (щільне ядро, розріджені краї);
    clusters  — кілька щільних скупчень (багато дуже близьких пар);
    grid      — квадратна решітка (максимум коциклічних четвірок);
    collinear — усі точки на одній прямій (вироджена тріангуляція-ланцюжок);
    circle    — точки біля одного великого кола (майже коциклічні, усі на опуклій
                оболонці; найважчий випадок для предиката in_circle і для злиття).
"""

from __future__ import annotations
import math
import random
from typing import Callable, Optional
from geometry import Point

DEFAULT_SIZE = 10**6

def _clamp(v: float, size: int) -> int:
    return min(size, max(0, round(v)))

def uniform(n: int, rnd: random.Random, size: int) -> list[Point]:
    return [Point(rnd.randint(0, size), rnd.randint(0, size)) for _ in range(n)]

def gauss(n: int, rnd: random.Random, size: int) -> list[Point]:
    c, s = size / 2, size / 8
    return [Point(_clamp(rnd.gauss(c, s), size), _clamp(rnd.gauss(c, s), size)) for _ in range(n)]

def clusters(n: int, rnd: random.Random, size: int) -> list[Point]:
    k = max(1, min(20, n // 50))                       # кількість скупчень
    centers = [(rnd.uniform(0.1, 0.9) * size, rnd.uniform(0.1, 0.9) * size) for _ in range(k)]
    s = size / 60
    pts = []
    for i in range(n):
        cx, cy = centers[i % k]
        pts.append(Point(_clamp(rnd.gauss(cx, s), size), _clamp(rnd.gauss(cy, s), size)))
    return pts

def grid(n: int, rnd: random.Random, size: int) -> list[Point]:
    side = max(1, math.ceil(math.sqrt(n)))
    step = size // max(1, side - 1) if side > 1 else 0
    cells = [(x, y) for x in range(side) for y in range(side)]
    rnd.shuffle(cells)                                 # випадковий порядок на вході
    return [Point(x * step, y * step) for x, y in cells[:n]]

def collinear(n: int, rnd: random.Random, size: int) -> list[Point]:
    # пряма y = x/2 + size/4, усі точки в межах квадрата
    xs = rnd.sample(range(0, size + 1, 2), min(n, size // 2 + 1))
    return [Point(x, x // 2 + size // 4) for x in xs]

def circle(n: int, rnd: random.Random, size: int) -> list[Point]:
    c, r = size / 2, size * 0.45
    pts = []
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi)
        pts.append(Point(_clamp(c + r * math.cos(a), size), _clamp(c + r * math.sin(a), size)))
    return pts

GENERATORS: dict[str, Callable[[int, random.Random, int], list[Point]]] = {
    "uniform": uniform,
    "gauss": gauss,
    "clusters": clusters,
    "grid": grid,
    "collinear": collinear,
    "circle": circle,
}

def generate(
    kind: str,
    n: int,
    seed: Optional[int] = None,
    size: int = DEFAULT_SIZE,
    unique: bool = False,
) -> list[Point]:
    """Генерує n точок розподілу kind у квадраті [0, size]².
    unique=True прибирає дублікати (точок тоді може бути трохи менше за n).
    """
    if kind not in GENERATORS:
        raise ValueError(f"невідомий розподіл {kind!r}; доступні: {', '.join(GENERATORS)}")
    if n < 0 or size < 1:
        raise ValueError("n має бути ≥ 0, size ≥ 1")
    rnd = random.Random(seed)
    pts = GENERATORS[kind](n, rnd, size)
    if unique:
        pts = list(dict.fromkeys(pts))                 # зберігає порядок
    return pts