"""Структура даних Quad-Edge (Guibas & Stolfi, 1985).

Зберігає планарне розбиття (у нас: тріангуляцію Делоне) і одночасно
дуальне до нього (діаграму Вороного).

Кожне неорієнтоване ребро представлене чотирма орієнтованими «чверть-ребрами»
e[0..3], пов'язаними операцією rot (поворот на 90° проти год. стрілки):

    e[0] — ребро a → b первинного графа (тріангуляції)
    e[1] — дуальне ребро (Вороного), що перетинає e[0] справа наліво
    e[2] — ребро b → a (це sym від e[0])
    e[3] — дуальне ребро у протилежному напрямку

Для кожного чверть-ребра зберігаються лише:
    org  — початкова вершина (для дуальних ребер не використовується),
    next — наступне ребро проти год. стрілки з тим самим початком (onext),
    rot  — сусіднє чверть-ребро в четвірці.
Усі інші переходи виражаються через них за O(1).
"""

from __future__ import annotations
from typing import Iterator, Optional
from geometry import Point


class Edge:
    """Орієнтоване чверть-ребро."""
    __slots__ = ("org", "next", "rot")

    def __init__(self) -> None:
        self.org: Optional[Point] = None
        self.next: Edge = self
        self.rot: Edge = self

    # ---- базові переходи ----

    @property
    def sym(self) -> Edge:
        """Те саме ребро у зворотному напрямку: a→b перетворюється на b→a."""
        return self.rot.rot

    @property
    def inv_rot(self) -> Edge:
        """Поворот на 90° за год. стрілкою (обернений до rot)."""
        return self.rot.rot.rot

    @property
    def dest(self) -> Optional[Point]:
        """Кінцева вершина ребра."""
        return self.sym.org

    # ---- обхід навколо вершин і граней ----

    @property
    def onext(self) -> Edge:
        """Наступне ребро проти год. стрілки навколо org."""
        return self.next

    @property
    def oprev(self) -> Edge:
        """Наступне ребро за год. стрілкою навколо org."""
        return self.rot.next.rot

    @property
    def lnext(self) -> Edge:
        """Наступне ребро вздовж лівої грані (обхід грані проти год. стрілки)."""
        return self.inv_rot.next.rot

    @property
    def lprev(self) -> Edge:
        """Попереднє ребро вздовж лівої грані."""
        return self.next.sym

    @property
    def rnext(self) -> Edge:
        """Наступне ребро вздовж правої грані."""
        return self.rot.next.inv_rot

    @property
    def rprev(self) -> Edge:
        """Попереднє ребро вздовж правої грані."""
        return self.sym.next

    @property
    def dnext(self) -> Edge:
        """Наступне ребро проти год. стрілки навколо dest."""
        return self.sym.next.sym

    @property
    def dprev(self) -> Edge:
        """Наступне ребро за год. стрілкою навколо dest."""
        return self.inv_rot.next.inv_rot

    def __repr__(self) -> str:
        return f"Edge({self.org} -> {self.dest})"

# ---------- топологічні операції ----------

def make_edge(org: Point, dest: Point) -> Edge:
    """Створює нове ізольоване ребро org → dest (окрема компонента зв'язності)."""
    e0, e1, e2, e3 = Edge(), Edge(), Edge(), Edge()
    e0.rot, e1.rot, e2.rot, e3.rot = e1, e2, e3, e0
    # Первинні чверть-ребра: кожне саме у своєму кільці onext (вершина степеня 1)
    e0.next, e2.next = e0, e2
    # Дуальні: ребро має одну грань з обох боків, тож e1 і e3 утворюють кільце
    e1.next, e3.next = e3, e1
    e0.org, e2.org = org, dest
    return e0

def splice(a: Edge, b: Edge) -> None:
    """Основна операція Guibas–Stolfi.
    Якщо кільця onext(a) і onext(b) різні, зливає їх в одне;
    якщо це одне кільце, розрізає його на два. Одночасно так само
    змінюються відповідні кільця дуальних ребер. Операція є інволюцією:
    повторний splice(a, b) повертає все як було.
    """
    alpha = a.next.rot
    beta = b.next.rot
    a.next, b.next = b.next, a.next
    alpha.next, beta.next = beta.next, alpha.next

def connect(a: Edge, b: Edge) -> Edge:
    """Додає нове ребро від a.dest до b.org.
    Нове ребро e має ту саму ліву грань, що й a та b:
    після виклику e.lprev = a, e.lnext = b.
    """
    e = make_edge(a.dest, b.org)
    splice(e, a.lnext)
    splice(e.sym, b)
    return e

def delete_edge(e: Edge) -> None:
    """Від'єднує ребро e від решти структури."""
    splice(e, e.oprev)
    splice(e.sym, e.sym.oprev)

# ---------- обхід усієї структури ----------

def iter_edges(start: Edge) -> Iterator[Edge]:
    """Перебирає всі ребра компоненти зв'язності, що містить start.
    Кожне неорієнтоване ребро видається рівно один раз, як одне з двох
    первинних чверть-ребер. Обхід у глибину по переходах onext і sym,
    час O(E), де E — кількість ребер.
    """
    seen: set[int] = set()
    stack = [start]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        # позначаємо обидва напрямки, щоб ребро не видалося вдруге
        seen.add(id(e))
        seen.add(id(e.sym))
        yield e
        for nxt in (e.onext, e.sym.onext):
            if id(nxt) not in seen:
                stack.append(nxt)