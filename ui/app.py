"""Графічний інтерфейс лабораторної «Усі найближчі сусіди» (PySide6 / Qt 6).

Запуск з кореня репозиторію:
    python ui/app.py

Можливості:
    • ручний ввід до 100 точок мишкою або координатами;
    • автогенерація до 20 000 точок (6 розподілів, зокрема найгірші);
    • шари: тріангуляція Делоне, діаграма Вороного, стрілки p → NN(p),
      взаємні пари, номери точок;
    • масштаб коліщатком, зсув перетягуванням, підказка при наведенні;
    • звірка з наївним O(N²), відкриття/збереження точок, експорт PNG.

Обчислення йдуть у фоновому потоці, тож вікно не зависає на 10⁴ точках.
"""

from __future__ import annotations
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (QAction, QColor, QFont, QKeySequence, QPainter, QPen,
                           QPolygonF, QTransform)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from brute_force import all_nearest_neighbors_brute
from generator import generate
from geometry import Point
from scene import Scene, compute_scene

WORLD = 10**6            # робоча область [0, WORLD]²
MANUAL_LIMIT = 100       # обмеження ручного режиму з вимог
GEN_MAX = 20_000
# ---------- палітра ----------
C = {
    "bg":        QColor("#0e1016"),
    "area":      QColor("#141822"),
    "area_edge": QColor("#2a3040"),
    "grid":      QColor(255, 255, 255, 10),
    "text":      QColor("#e7e9ef"),
    "muted":     QColor("#8a93a6"),
    "point":     QColor("#f4f6fb"),
    "delaunay":  QColor(110, 150, 210, 120),
    "voronoi":   QColor("#f2a541"),
    "nn":        QColor("#38d0c6"),
    "mutual":    QColor("#ff5c8a"),
    "hover":     QColor("#ffd166"),
}

DISTRIBUTIONS = [
    ("uniform",   "Рівномірний"),
    ("gauss",     "Нормальний (Гаусс)"),
    ("clusters",  "Кластери"),
    ("grid",      "Решітка · коциклічні"),
    ("collinear", "Колінеарні · вироджений"),
    ("circle",    "Коло · найгірший"),
]

LAYERS = [
    ("voronoi",  "Діаграма Вороного",   C["voronoi"]),
    ("delaunay", "Тріангуляція Делоне", QColor(110, 150, 210)),
    ("nn",       "Стрілки p → NN(p)",   C["nn"]),
    ("mutual",   "Взаємні пари",        C["mutual"]),
    ("points",   "Точки",               C["point"]),
    ("labels",   "Номери точок",        C["muted"]),
]

def fmt_int(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")

def fmt_f(v: float, digits: int = 1) -> str:
    return f"{v:,.{digits}f}".replace(",", " ")
# ====================================================================
#                           Фонові потоки
# ====================================================================
class ComputeThread(QThread):
    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, token: int, points: list[Point]) -> None:
        super().__init__()
        self.token, self.points = token, points

    def run(self) -> None:
        try:
            self.done.emit(self.token, compute_scene(self.points, WORLD))
        except Exception:
            self.failed.emit(self.token, traceback.format_exc())

class BruteThread(QThread):
    done = Signal(object, float)

    def __init__(self, points: list[Point]) -> None:
        super().__init__()
        self.points = points

    def run(self) -> None:
        import time
        t0 = time.perf_counter()
        res = all_nearest_neighbors_brute(self.points)
        self.done.emit(res, (time.perf_counter() - t0) * 1000)
# ====================================================================
#                              Полотно
# ====================================================================
class Canvas(QWidget):
    pointAdded = Signal(object)        # Point
    pointRemoved = Signal(int)
    hoverChanged = Signal(str)

    PICK_PX = 10

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.points: list[Point] = []
        self.scene: Optional[Scene] = None
        self.editable = True
        self.layers = {k: k not in ("labels",) for k, _, _ in LAYERS}

        self.scale = 1.0
        self.cx = self.cy = WORLD / 2
        self._fitted = False

        self._press: Optional[QPointF] = None
        self._panning = False
        self._hover: Optional[int] = None
        self._buckets: dict[tuple[int, int], list[int]] = {}
        self._cell = WORLD / 256

        # геометрія в світових координатах, готується один раз на сцену
        self._tri: list = []
        self._vor: list = []
        self._nn: list = []
        self._nn_lines: list = []
        self._mut: list = []

    # ---------- дані ----------

    def set_points(self, pts: list[Point]) -> None:
        self.points = pts
        self._buckets = {}
        for i, p in enumerate(pts):
            self._buckets.setdefault((int(p.x // self._cell), int(p.y // self._cell)), []).append(i)
        if self.scene is not None and self.scene.points != pts:
            self.scene = None            # стара сцена не відповідає точкам
            self._tri, self._vor, self._nn, self._nn_lines, self._mut = [], [], [], [], []
        self._hover = None
        self.hoverChanged.emit("")
        self.update()

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene
        self._tri = [QLineF(a.x, a.y, b.x, b.y) for a, b in scene.tri_edges]
        self._vor = [QLineF(a[0], a[1], b[0], b[1]) for a, b in scene.vor_segments]
        mutual = {i for pair in scene.mutual for i in pair}
        pts = scene.points
        self._nn = [(i, j) for i, j in enumerate(scene.nn) if j is not None and i not in mutual]
        self._mut = [QLineF(pts[i].x, pts[i].y, pts[j].x, pts[j].y) for i, j in scene.mutual]
        self._nn_lines = [QLineF(pts[i].x, pts[i].y, pts[j].x, pts[j].y) for i, j in self._nn]
        self.update()

    # ---------- перетворення координат ----------

    def world_tf(self) -> QTransform:
        t = QTransform()
        t.translate(self.width() / 2, self.height() / 2)
        t.scale(self.scale, -self.scale)
        t.translate(-self.cx, -self.cy)
        return t

    def to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(self.width() / 2 + (x - self.cx) * self.scale,
                       self.height() / 2 - (y - self.cy) * self.scale)

    def to_world(self, p: QPointF) -> tuple[float, float]:
        return (self.cx + (p.x() - self.width() / 2) / self.scale,
                self.cy - (p.y() - self.height() / 2) / self.scale)

    def fit(self) -> None:
        m = 36
        side = max(50, min(self.width(), self.height()) - 2 * m)
        self.scale = side / WORLD
        self.cx = self.cy = WORLD / 2
        self.update()

    def resizeEvent(self, e) -> None:
        if not self._fitted:
            self.fit()
            self._fitted = True
        super().resizeEvent(e)

    # ---------- пошук точки під курсором ----------

    def pick(self, pos: QPointF) -> Optional[int]:
        if not self.points:
            return None
        wx, wy = self.to_world(pos)
        r = self.PICK_PX / self.scale
        best, best_d = None, r * r
        c = self._cell
        x0, x1 = int((wx - r) // c), int((wx + r) // c)
        y0, y1 = int((wy - r) // c), int((wy + r) // c)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 4096:   # дуже віддалений масштаб
            cand = range(len(self.points))
        else:
            cand = (i for gx in range(x0, x1 + 1) for gy in range(y0, y1 + 1)
                    for i in self._buckets.get((gx, gy), ()))
        for i in cand:
            p = self.points[i]
            d = (p.x - wx) ** 2 + (p.y - wy) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    # ---------- миша та клавіатура ----------

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._press = e.position()
            self._panning = False
        elif e.button() == Qt.RightButton and self.editable:
            i = self.pick(e.position())
            if i is not None:
                self.pointRemoved.emit(i)

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        if self._press is not None and e.buttons() & Qt.LeftButton:
            d = pos - self._press
            if self._panning or abs(d.x()) + abs(d.y()) > 4:
                self._panning = True
                self.setCursor(Qt.ClosedHandCursor)
                self.cx -= d.x() / self.scale
                self.cy += d.y() / self.scale
                self._press = pos
                self.update()
                return
        self._update_hover(pos)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self._press is not None:
            if not self._panning and self.editable:
                wx, wy = self.to_world(e.position())
                if 0 <= wx <= WORLD and 0 <= wy <= WORLD:
                    self.pointAdded.emit(Point(round(wx), round(wy)))
            self._press = None
            self._panning = False
            self.unsetCursor()

    def wheelEvent(self, e) -> None:
        k = 1.15 ** (e.angleDelta().y() / 120)
        fit_scale = (min(self.width(), self.height()) - 72) / WORLD
        new = min(max(self.scale * k, fit_scale * 0.3), fit_scale * 5000)
        pos = e.position()
        wx, wy = self.to_world(pos)
        self.scale = new
        # точка під курсором лишається на місці
        self.cx = wx - (pos.x() - self.width() / 2) / self.scale
        self.cy = wy + (pos.y() - self.height() / 2) / self.scale
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover = None
        self.hoverChanged.emit("")
        self.update()

    def _update_hover(self, pos: QPointF) -> None:
        wx, wy = self.to_world(pos)
        i = self.pick(pos)
        if i != self._hover:
            self._hover = i
            self.update()
        msg = f"x = {fmt_int(wx)}   y = {fmt_int(wy)}"
        if i is not None:
            p = self.points[i]
            msg += f"     │  точка #{i} ({fmt_int(p.x)}, {fmt_int(p.y)})"
            sc = self.scene
            if sc is not None and i < len(sc.nn) and sc.nn[i] is not None:
                j = sc.nn[i]
                q = self.points[j]
                tag = "  ⇄ взаємні" if sc.nn[j] == i else ""
                msg += f"  →  NN = #{j} ({fmt_int(q.x)}, {fmt_int(q.y)}),  d = {fmt_f(sc.nn_dist[i])}{tag}"
        self.hoverChanged.emit(msg)

    # ---------- малювання ----------

    def _pen(self, color: QColor, width: float) -> QPen:
        pen = QPen(color, width)
        pen.setCosmetic(True)                       # товщина в пікселях за будь-якого масштабу
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), C["bg"])

        n = len(self.points)
        big = n > 2000

        # робоча область і сітка
        tl, br = self.to_screen(0, WORLD), self.to_screen(WORLD, 0)
        area = QRectF(tl, br)
        p.setPen(Qt.NoPen)
        p.setBrush(C["area"])
        p.drawRoundedRect(area, 6, 6)
        p.setPen(QPen(C["grid"], 1))
        for k in range(1, 10):
            x = tl.x() + area.width() * k / 10
            y = tl.y() + area.height() * k / 10
            p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        p.setPen(QPen(C["area_edge"], 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(area, 6, 6)

        sc = self.scene
        p.save()
        p.setTransform(self.world_tf())
        if big:
            p.setRenderHint(QPainter.Antialiasing, False)
        if sc is not None:
            if self.layers["voronoi"] and self._vor:
                col = QColor(C["voronoi"])
                if big:
                    col.setAlpha(120)
                p.setPen(self._pen(col, 1.0 if big else 1.4))
                p.drawLines(self._vor)
            if self.layers["delaunay"] and self._tri:
                col = QColor(C["delaunay"])
                if big:
                    col.setAlpha(55)
                p.setPen(self._pen(col, 1.0))
                p.drawLines(self._tri)
            if self.layers["nn"] and self._nn_lines:
                p.setPen(self._pen(C["nn"], 1.0 if big else 1.6))
                p.drawLines(self._nn_lines)
            if self.layers["mutual"] and self._mut:
                p.setPen(self._pen(C["mutual"], 1.6 if big else 3.0))
                p.drawLines(self._mut)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.layers["points"] and n and n > MANUAL_LIMIT:
            p.setPen(self._pen(C["point"], 2.0 if n > 5000 else 3.0))
            p.drawPoints([QPointF(q.x, q.y) for q in self.points])
        p.restore()

        # стрілки і точки малого набору — в екранних координатах
        if sc is not None and self.layers["nn"]:
            self._draw_arrowheads(p, sc)
        if self.layers["points"] and 0 < n <= MANUAL_LIMIT:
            p.setPen(QPen(C["bg"], 1.5))
            p.setBrush(C["point"])
            for q in self.points:
                p.drawEllipse(self.to_screen(q.x, q.y), 4.5, 4.5)
        if self.layers["labels"] and 0 < n <= 300:
            p.setPen(C["muted"])
            p.setFont(QFont(self.font().family(), 8))
            for i, q in enumerate(self.points):
                s = self.to_screen(q.x, q.y)
                p.drawText(s + QPointF(6, -6), str(i))

        self._draw_hover(p)
        self._draw_legend(p)

        if n == 0:
            p.setPen(C["muted"])
            p.setFont(QFont(self.font().family(), 12))
            msg = ("Клацніть у робочій області, щоб додати точку,\nабо згенеруйте набір на панелі ліворуч"
                   if self.editable else "Згенеруйте набір точок на панелі ліворуч")
            p.drawText(area, Qt.AlignCenter, msg)
        p.end()

    def _draw_arrowheads(self, p: QPainter, sc: Scene, limit: int = 600) -> None:
        """Наконечники стрілок лише для видимих ребер і лише коли їх небагато
        (на 10⁴ точках без масштабу вони злилися б у шум)."""
        view = QRectF(self.rect()).adjusted(-20, -20, 20, 20)
        visible = []
        for i, j in self._nn:
            b = self.to_screen(sc.points[j].x, sc.points[j].y)
            if view.contains(b):
                visible.append((self.to_screen(sc.points[i].x, sc.points[i].y), b))
                if len(visible) > limit:
                    return
        p.setPen(Qt.NoPen)
        p.setBrush(C["nn"])
        for a, b in visible:
            dx, dy = b.x() - a.x(), b.y() - a.y()
            L = math.hypot(dx, dy)
            if L < 14:
                continue
            ux, uy = dx / L, dy / L
            tip = QPointF(b.x() - ux * 6.5, b.y() - uy * 6.5)
            base = QPointF(tip.x() - ux * 9, tip.y() - uy * 9)
            nx, ny = -uy * 4.2, ux * 4.2
            p.drawPolygon(QPolygonF([tip, QPointF(base.x() + nx, base.y() + ny),
                                     QPointF(base.x() - nx, base.y() - ny)]))

    def _draw_hover(self, p: QPainter) -> None:
        i = self._hover
        if i is None or i >= len(self.points):
            return
        a = self.to_screen(self.points[i].x, self.points[i].y)
        sc = self.scene
        if sc is not None and i < len(sc.nn) and sc.nn[i] is not None:
            j = sc.nn[i]
            b = self.to_screen(self.points[j].x, self.points[j].y)
            r = math.hypot(b.x() - a.x(), b.y() - a.y())
            # порожній круг навколо p радіуса |p, NN(p)|
            p.setPen(QPen(QColor(255, 209, 102, 150), 1, Qt.DashLine))
            p.setBrush(QColor(255, 209, 102, 18))
            p.drawEllipse(a, r, r)
            p.setPen(QPen(C["hover"], 2.4))
            p.drawLine(a, b)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(b, 7, 7)
        p.setPen(QPen(C["hover"], 2.4))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(a, 8, 8)

    def _draw_legend(self, p: QPainter) -> None:
        items = [(name, col) for key, name, col in LAYERS[:4] if self.layers[key]]
        if not items or self.scene is None:
            return
        f = QFont(self.font().family(), 9)
        p.setFont(f)
        fm = p.fontMetrics()
        w = 34 + max(fm.horizontalAdvance(n) for n, _ in items) + 14
        h = 12 + 20 * len(items)
        box = QRectF(14, 14, w, h)
        p.setPen(QPen(C["area_edge"], 1))
        p.setBrush(QColor(14, 16, 22, 215))
        p.drawRoundedRect(box, 8, 8)
        for k, (name, col) in enumerate(items):
            y = box.top() + 16 + 20 * k
            p.setPen(QPen(col, 3, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(box.left() + 12, y), QPointF(box.left() + 28, y))
            p.setPen(C["text"])
            p.drawText(QPointF(box.left() + 36, y + fm.ascent() / 2 - 1), name)

# ====================================================================
#                           Бічна панель
# ====================================================================

def card(title: str) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(8)
    t = QLabel(title.upper())
    t.setObjectName("cardTitle")
    lay.addWidget(t)
    return f, lay

class Swatch(QWidget):
    def __init__(self, color: QColor) -> None:
        super().__init__()
        self.color = color
        self.setFixedSize(18, 10)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(self.color, 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(2, 5), QPointF(16, 5))

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Усі найближчі сусіди · Divide & Conquer (Guibas–Stolfi)")
        self.resize(1360, 860)

        self.points: list[Point] = []
        self.history: list[list[Point]] = []
        self.token = 0
        self.threads: set[QThread] = set()
        self.scene: Optional[Scene] = None

        self.canvas = Canvas()
        self.canvas.pointAdded.connect(self.on_point_added)
        self.canvas.pointRemoved.connect(self.on_point_removed)

        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        h.addWidget(self.canvas, 1)
        self.setCentralWidget(root)

        sb = self.statusBar()
        self.hover_label = QLabel()
        self.state_label = QLabel()
        self.state_label.setObjectName("state")
        sb.addWidget(self.hover_label, 1)
        sb.addPermanentWidget(self.state_label)
        self.canvas.hoverChanged.connect(self.hover_label.setText)

        self._shortcut("Ctrl+Z", self.undo)
        self._shortcut("Ctrl+G", self.on_generate)
        self._shortcut("F", self.canvas.fit)

        self.set_mode("manual")
        self.refresh_stats()

    def _shortcut(self, keys: str, fn) -> None:
        a = QAction(self)
        a.setShortcut(QKeySequence(keys))
        a.triggered.connect(fn)
        self.addAction(a)

    # ---------- побудова панелі ----------

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(312)
        v = QVBoxLayout(side)
        v.setContentsMargins(16, 18, 16, 18)
        v.setSpacing(12)

        title = QLabel("Усі найближчі сусіди")
        title.setObjectName("appTitle")
        sub = QLabel("АБ1 · №46 · Divide & Conquer · O(N log N)")
        sub.setObjectName("appSub")
        v.addWidget(title)
        v.addWidget(sub)

        # перемикач режиму
        seg = QFrame()
        seg.setObjectName("segment")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(3)
        self.mode_group = QButtonGroup(self)
        self.btn_manual = QPushButton("Ручний ввід")
        self.btn_gen = QPushButton("Автогенерація")
        for b, m in ((self.btn_manual, "manual"), (self.btn_gen, "gen")):
            b.setCheckable(True)
            b.setObjectName("segBtn")
            b.clicked.connect(lambda _=False, m=m: self.set_mode(m))
            self.mode_group.addButton(b)
            sl.addWidget(b)
        v.addWidget(seg)

        # --- ручний ввід ---
        self.manual_card, ml = card("Ручний ввід")
        hint = QLabel("ЛКМ — додати · ПКМ — видалити\nколесо — масштаб · тягнути — зсув")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        ml.addWidget(hint)
        self.counter = QLabel()
        self.counter.setObjectName("counter")
        ml.addWidget(self.counter)
        row = QHBoxLayout()
        self.in_x, self.in_y = QSpinBox(), QSpinBox()
        for s, pre in ((self.in_x, "x "), (self.in_y, "y ")):
            s.setRange(0, WORLD)
            s.setSingleStep(10_000)
            s.setPrefix(pre)
            s.setGroupSeparatorShown(True)
            s.setMinimumWidth(60)
            row.addWidget(s, 1)
        add = QPushButton("+")
        add.setObjectName("square")
        add.setToolTip("Додати точку з координатами")
        add.clicked.connect(lambda: self.on_point_added(Point(self.in_x.value(), self.in_y.value())))
        row.addWidget(add)
        ml.addLayout(row)
        row = QHBoxLayout()
        undo = QPushButton("Скасувати")
        undo.setToolTip("Ctrl+Z")
        undo.clicked.connect(self.undo)
        clear = QPushButton("Очистити")
        clear.clicked.connect(self.clear)
        row.addWidget(undo)
        row.addWidget(clear)
        ml.addLayout(row)
        v.addWidget(self.manual_card)

        # --- генерація ---
        self.gen_card, gl = card("Автогенерація")
        self.dist = QComboBox()
        for key, name in DISTRIBUTIONS:
            self.dist.addItem(name, key)
        gl.addWidget(self.dist)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.addWidget(QLabel("N"), 0, 0)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(2, GEN_MAX)
        self.n_spin.setValue(10_000)
        self.n_spin.setSingleStep(1000)
        self.n_spin.setGroupSeparatorShown(True)
        grid.addWidget(self.n_spin, 0, 1, 1, 2)
        grid.addWidget(QLabel("seed"), 1, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(46)
        grid.addWidget(self.seed_spin, 1, 1)
        dice = QPushButton("⚄")
        dice.setObjectName("square")
        dice.setToolTip("Випадковий seed")
        dice.clicked.connect(lambda: self.seed_spin.setValue(random.randint(0, 999_999)))
        grid.addWidget(dice, 1, 2)
        grid.setColumnStretch(1, 1)
        gl.addLayout(grid)
        gen = QPushButton("Згенерувати")
        gen.setObjectName("primary")
        gen.setToolTip("Ctrl+G")
        gen.clicked.connect(self.on_generate)
        gl.addWidget(gen)
        v.addWidget(self.gen_card)

        # --- шари ---
        lc, ll = card("Шари")
        self.layer_boxes = {}
        for key, name, col in LAYERS:
            row = QHBoxLayout()
            row.setSpacing(8)
            cb = QCheckBox(name)
            cb.setChecked(self.canvas.layers[key])
            cb.toggled.connect(lambda on, k=key: self.toggle_layer(k, on))
            self.layer_boxes[key] = cb
            row.addWidget(Swatch(col))
            row.addWidget(cb, 1)
            ll.addLayout(row)
        v.addWidget(lc)

        # --- результат ---
        rc, rl = card("Результат")
        self.stats = QGridLayout()
        self.stats.setVerticalSpacing(5)
        self.stat_labels = {}
        rows = [("n", "Точок N"), ("u", "Різних точок"), ("e", "Ребер Делоне"),
                ("t", "Трикутників"), ("m", "Взаємних пар"), ("dmin", "Мін. |p, NN(p)|"),
                ("dmax", "Макс. |p, NN(p)|"), ("time", "Час ANN")]
        for r, (k, name) in enumerate(rows):
            a = QLabel(name)
            a.setObjectName("statKey")
            b = QLabel("—")
            b.setObjectName("statVal")
            b.setAlignment(Qt.AlignRight)
            self.stats.addWidget(a, r, 0)
            self.stats.addWidget(b, r, 1)
            self.stat_labels[k] = b
        rl.addLayout(self.stats)
        self.verify_btn = QPushButton("Звірити з наївним O(N²)")
        self.verify_btn.clicked.connect(self.on_verify)
        rl.addWidget(self.verify_btn)
        self.verify_label = QLabel()
        self.verify_label.setObjectName("verify")
        rl.addWidget(self.verify_label)
        v.addWidget(rc)

        # --- файли ---
        fc, fl = card("Файли")
        row = QHBoxLayout()
        for text, fn in (("Відкрити", self.on_open), ("Зберегти", self.on_save)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        fl.addLayout(row)
        row = QHBoxLayout()
        png = QPushButton("Експорт PNG")
        png.clicked.connect(self.on_png)
        fitb = QPushButton("Вписати (F)")
        fitb.clicked.connect(self.canvas.fit)
        row.addWidget(png)
        row.addWidget(fitb)
        fl.addLayout(row)
        v.addWidget(fc)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("sideScroll")
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(322)
        return scroll

    # ---------- режими та шари ----------

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        manual = mode == "manual"
        (self.btn_manual if manual else self.btn_gen).setChecked(True)
        self.manual_card.setVisible(manual)
        self.gen_card.setVisible(not manual)
        self.canvas.editable = manual
        self.canvas.update()
        self.update_counter()

    def toggle_layer(self, key: str, on: bool) -> None:
        self.canvas.layers[key] = on
        self.canvas.update()

    def update_counter(self) -> None:
        n = len(self.points)
        over = n >= MANUAL_LIMIT
        self.counter.setText(f"{n} / {MANUAL_LIMIT} точок" + ("  · ліміт досягнуто" if over else ""))
        self.counter.setProperty("full", over)
        self.counter.style().unpolish(self.counter)
        self.counter.style().polish(self.counter)

    # ---------- зміна набору ----------

    def set_points(self, pts: list[Point], remember: bool = True) -> None:
        if remember:
            self.history.append(self.points)
            del self.history[:-50]
        self.points = pts
        self.scene = None
        self.canvas.set_points(pts)
        self.verify_label.setText("")
        self.update_counter()
        self.refresh_stats()
        self.request_compute()

    def on_point_added(self, p: Point) -> None:
        if self.mode != "manual":
            return
        if len(self.points) >= MANUAL_LIMIT:
            self.statusBar().showMessage(f"Ручний режим: не більше {MANUAL_LIMIT} точок", 3000)
            return
        self.set_points(self.points + [p])

    def on_point_removed(self, i: int) -> None:
        self.set_points(self.points[:i] + self.points[i + 1:])

    def undo(self) -> None:
        if self.history:
            self.set_points(self.history.pop(), remember=False)

    def clear(self) -> None:
        if self.points:
            self.set_points([])

    def on_generate(self) -> None:
        if self.mode != "gen":
            self.set_mode("gen")
        kind = self.dist.currentData()
        pts = generate(kind, self.n_spin.value(), seed=self.seed_spin.value(), size=WORLD, unique=True)
        self.set_points(pts)
        self.canvas.fit()

    # ---------- обчислення ----------

    def request_compute(self) -> None:
        self.token += 1
        if len(self.points) == 0:
            self.state_label.setText("")
            return
        self.state_label.setText("⏳ обчислення…")
        th = ComputeThread(self.token, list(self.points))
        th.done.connect(self.on_computed)
        th.failed.connect(self.on_failed)
        th.finished.connect(lambda th=th: self.threads.discard(th))
        self.threads.add(th)
        th.start()

    def on_computed(self, token: int, scene: Scene) -> None:
        if token != self.token:
            return                              # результат для застарілого набору
        self.scene = scene
        self.canvas.set_scene(scene)
        self.state_label.setText(f"✓ готово за {fmt_f(scene.ann_ms)} мс")
        self.refresh_stats()

    def on_failed(self, token: int, tb: str) -> None:
        if token == self.token:
            self.state_label.setText("✗ помилка")
            QMessageBox.critical(self, "Помилка обчислення", tb)

    def refresh_stats(self) -> None:
        s = self.scene
        L = self.stat_labels
        L["n"].setText(fmt_int(len(self.points)))
        if s is None:
            for k in ("u", "e", "t", "m", "dmin", "dmax", "time"):
                L[k].setText("—")
            return
        d = [x for x in s.nn_dist if not math.isnan(x)]
        L["u"].setText(fmt_int(s.unique))
        L["e"].setText(fmt_int(len(s.tri_edges)))
        L["t"].setText(fmt_int(s.n_triangles))
        L["m"].setText(fmt_int(len(s.mutual)))
        L["dmin"].setText(fmt_f(min(d)) if d else "—")
        L["dmax"].setText(fmt_f(max(d)) if d else "—")
        L["time"].setText(f"{fmt_f(s.ann_ms)} мс")

    def on_verify(self) -> None:
        if self.scene is None or len(self.points) < 2:
            return
        n = len(self.points)
        if n > 3000:
            est = 45 * (n / 16000) ** 2
            r = QMessageBox.question(
                self, "Наївний алгоритм",
                f"O(N²) на {fmt_int(n)} точках триватиме приблизно {est:.0f} с. Запустити?")
            if r != QMessageBox.Yes:
                return
        self.verify_btn.setEnabled(False)
        self.verify_label.setProperty("ok", None)
        self.verify_label.setText("⏳ рахую наївним перебором…")
        scene, token = self.scene, self.token
        th = BruteThread(list(self.points))

        def done(res, ms):
            self.verify_btn.setEnabled(True)
            if token != self.token:
                self.verify_label.setText("")
                return
            ok = res == scene.nn
            self.verify_label.setProperty("ok", ok)
            self.verify_label.style().unpolish(self.verify_label)
            self.verify_label.style().polish(self.verify_label)
            speed = ms / scene.ann_ms if scene.ann_ms > 0 else float("inf")
            self.verify_label.setText(
                (f"✓ Збігається для всіх {fmt_int(n)} точок" if ok
                 else "✗ Розбіжність з наївним алгоритмом!")
                + f"\nO(N²): {ms:,.0f} мс · D&C: {scene.ann_ms:,.0f} мс · ×{speed:,.1f}".replace(",", " "))

        th.done.connect(done)
        th.finished.connect(lambda th=th: self.threads.discard(th))
        self.threads.add(th)
        th.start()

    # ---------- файли ----------

    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Відкрити точки", "", "Текст (*.txt);;Усі файли (*)")
        if not path:
            return
        pts = []
        try:
            for k, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
                line = line.split("#")[0].replace(",", " ").split()
                if not line:
                    continue
                if len(line) != 2:
                    raise ValueError(f"рядок {k}: очікується «x y»")
                x, y = int(line[0]), int(line[1])
                if not (0 <= x <= WORLD and 0 <= y <= WORLD):
                    raise ValueError(f"рядок {k}: координати мають бути в [0, {fmt_int(WORLD)}]")
                pts.append(Point(x, y))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Не вдалося відкрити", str(e))
            return
        if len(pts) > MANUAL_LIMIT:
            self.set_mode("gen")
        self.set_points(pts)
        self.canvas.fit()

    def on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти точки", "points.txt", "Текст (*.txt)")
        if path:
            Path(path).write_text("\n".join(f"{p.x} {p.y}" for p in self.points) + "\n", encoding="utf-8")

    def on_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Експорт зображення", "ann.png", "PNG (*.png)")
        if path:
            self.canvas.grab().save(path, "PNG")


# ====================================================================
#                               Стилі
# ====================================================================

QSS = """
* { color: #e7e9ef; font-size: 10pt; }
QMainWindow, #sidebar { background: #0b0d12; }
#sideScroll { border: none; border-right: 1px solid #1f2430; background: #0b0d12; }
#appTitle { font-size: 16pt; font-weight: 600; }
#appSub { color: #8a93a6; font-size: 9pt; margin-bottom: 4px; }
#card { background: #12151d; border: 1px solid #1f2430; border-radius: 10px; }
#cardTitle { color: #8a93a6; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }
#hint { color: #8a93a6; font-size: 9pt; }
#counter { font-weight: 600; }
#counter[full="true"] { color: #ff5c8a; }
#segment { background: #12151d; border: 1px solid #1f2430; border-radius: 9px; }
#segBtn { background: transparent; border: none; border-radius: 7px; padding: 7px 0; color: #8a93a6; }
#segBtn:checked { background: #262c3b; color: #ffffff; font-weight: 600; }
#segBtn:hover:!checked { color: #e7e9ef; }
QPushButton { background: #1b202b; border: 1px solid #2a3040; border-radius: 7px; padding: 7px 10px; }
QPushButton:hover { background: #232937; border-color: #3a4256; }
QPushButton:pressed { background: #161a23; }
QPushButton:disabled { color: #5b6375; }
#primary { background: #2f6fed; border: 1px solid #2f6fed; font-weight: 600; }
#primary:hover { background: #3f7cf0; }
#square { min-width: 34px; max-width: 34px; padding: 6px 0; font-size: 12pt; }
QSpinBox, QComboBox { background: #0e1016; border: 1px solid #2a3040; border-radius: 7px; padding: 6px 8px; }
QSpinBox:focus, QComboBox:focus { border-color: #2f6fed; }
QSpinBox::up-button, QSpinBox::down-button { width: 0; border: none; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #12151d; border: 1px solid #2a3040; selection-background-color: #262c3b; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #3a4256; background: #0e1016; }
QCheckBox::indicator:checked { background: #2f6fed; border-color: #2f6fed; }
#statKey { color: #8a93a6; }
#statVal { font-family: Consolas, 'JetBrains Mono', monospace; }
#verify { font-size: 9pt; color: #8a93a6; }
#verify[ok="true"] { color: #38d0c6; }
#verify[ok="false"] { color: #ff5c8a; }
QStatusBar { background: #0b0d12; border-top: 1px solid #1f2430; color: #8a93a6; }
QStatusBar QLabel { color: #8a93a6; font-family: Consolas, 'JetBrains Mono', monospace; font-size: 9pt; padding: 2px 8px; }
#state { color: #38d0c6; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: #2a3040; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QToolTip { background: #1b202b; color: #e7e9ef; border: 1px solid #2a3040; padding: 4px; }
QMessageBox { background: #12151d; }
"""

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("All Nearest Neighbors")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()