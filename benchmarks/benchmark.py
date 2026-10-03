"""Бенчмарк: емпірична перевірка складності O(N log N).

Запуск з кореня проєкту:
    python benchmarks/benchmark.py            # повний прогін (~2 хв)
    python benchmarks/benchmark.py --quick    # швидкий прогін (~15 с)

Результати (у benchmarks/results/):
    results.csv           — сирі заміри;
    dc_vs_brute.png       — D&C проти наївного O(N²), log-log;
    normalized.png        — T / (N log₂N) для всіх розподілів;
    у консоль — таблиці у форматі Markdown для звіту.
"""

from __future__ import annotations
import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brute_force import all_nearest_neighbors_brute  # noqa: E402
from generator import GENERATORS, generate  # noqa: E402
from nearest import all_nearest_neighbors  # noqa: E402

# ---------- палітра (категоріальні слоти 1–6 у фіксованому порядку) ----------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]     # друге кодування: для друку в ч/б

KIND_LABELS = {
    "uniform": "рівномірний",
    "gauss": "нормальний",
    "clusters": "кластери",
    "grid": "решітка",
    "collinear": "на прямій",
    "circle": "біля кола",
}

def measure(func, pts, repeats: int) -> float:
    """Мінімальний час із кількох запусків (найменш зашумлена оцінка)."""
    best = math.inf
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            func(pts)
            best = min(best, time.perf_counter() - t0)
        finally:
            gc.enable()
    return best

def slope(ns, ts) -> float:
    """Нахил прямої в координатах log N – log T (метод найменших квадратів).
    Для O(N^k) нахил ≈ k: для O(N²) ≈ 2, для O(N log N) трохи більше 1.
    """
    xs = [math.log(n) for n in ns]
    ys = [math.log(t) for t in ts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

def run(sizes, brute_max, repeats, kinds):
    rows = []
    for kind in kinds:
        for n in sizes:
            pts = generate(kind, n, seed=n)
            t = measure(all_nearest_neighbors, pts, repeats)
            rows.append({"kind": kind, "n": n, "algorithm": "dc", "seconds": t})
            msg = f"{kind:10s} N={n:6d}  D&C {t:8.4f} s"
            if kind == "uniform" and n <= brute_max:
                tb = measure(all_nearest_neighbors_brute, pts, 1 if n > 2000 else repeats)
                rows.append({"kind": kind, "n": n, "algorithm": "brute", "seconds": tb})
                msg += f"   brute {tb:8.4f} s"
            print(msg, flush=True)
    for r in rows:
        r["us_per_nlogn"] = r["seconds"] / (r["n"] * math.log2(r["n"])) * 1e6
    return rows

def save_csv(rows, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "n", "algorithm", "seconds", "us_per_nlogn"])
        w.writeheader()
        w.writerows(rows)

def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, which="both")

def plot(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fmt_n = FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))

    # ---- графік 1: D&C проти наївного, log-log ----
    dc = sorted((r["n"], r["seconds"]) for r in rows if r["kind"] == "uniform" and r["algorithm"] == "dc")
    bf = sorted((r["n"], r["seconds"]) for r in rows if r["algorithm"] == "brute")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    for (data, name, color, marker) in (
        (dc, "Розділяй і володарюй, O(N log N)", SERIES[0], MARKERS[0]),
        (bf, "Наївний перебір, O(N²)", SERIES[1], MARKERS[1]),
    ):
        if len(data) < 2:
            continue
        ns, ts = zip(*data)
        k = slope(ns, ts)
        ax.plot(ns, ts, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{name}: нахил {k:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(fmt_n)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    style_axes(ax, "Час роботи, рівномірний розподіл (логарифмічні шкали)",
               "Кількість точок N", "Час, с")
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "dc_vs_brute.png", facecolor=SURFACE)
    plt.close(fig)

    # ---- графік 2: нормований час T / (N log₂N) для всіх розподілів ----
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150, facecolor=SURFACE)
    kinds = [k for k in GENERATORS if any(r["kind"] == k for r in rows)]
    for i, kind in enumerate(kinds):
        data = sorted((r["n"], r["us_per_nlogn"]) for r in rows if r["kind"] == kind and r["algorithm"] == "dc")
        ns, vs = zip(*data)
        ax.plot(ns, vs, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=KIND_LABELS[kind])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(fmt_n)
    ax.set_ylim(bottom=0)
    style_axes(ax, "Нормований час T(N) / (N·log₂N): стала величина означає O(N log N)",
               "Кількість точок N", "мкс на одиницю N·log₂N")
    ax.legend(frameon=False, labelcolor=TEXT, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), handlelength=1.5, columnspacing=1.2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "normalized.png", facecolor=SURFACE)
    plt.close(fig)

def print_tables(rows, sizes):
    print("\n### D&C проти наївного (рівномірний розподіл)\n")
    print("| N | D&C, с | Наївний, с | Прискорення |")
    print("|---|---|---|---|")
    for n in sizes:
        d = next(r["seconds"] for r in rows if r["kind"] == "uniform" and r["n"] == n and r["algorithm"] == "dc")
        b = next((r["seconds"] for r in rows if r["n"] == n and r["algorithm"] == "brute"), None)
        print(f"| {n} | {d:.4f} | {'—' if b is None else f'{b:.4f}'} | {'—' if b is None else f'×{b / d:.1f}'} |")

    kinds = [k for k in GENERATORS if any(r["kind"] == k for r in rows)]
    print("\n### Час D&C (с) за розподілами\n")
    print("| N | " + " | ".join(KIND_LABELS[k] for k in kinds) + " |")
    print("|---|" + "---|" * len(kinds))
    for n in sizes:
        cells = [next(r["seconds"] for r in rows if r["kind"] == k and r["n"] == n and r["algorithm"] == "dc")
                 for k in kinds]
        print(f"| {n} | " + " | ".join(f"{c:.4f}" for c in cells) + " |")

    dc = sorted((r["n"], r["seconds"]) for r in rows if r["kind"] == "uniform" and r["algorithm"] == "dc")
    bf = sorted((r["n"], r["seconds"]) for r in rows if r["algorithm"] == "brute")
    print(f"\nНахил log T від log N: D&C = {slope(*zip(*dc)):.2f}", end="")
    print(f", наївний = {slope(*zip(*bf)):.2f}" if len(bf) >= 2 else "")

def main():
    ap = argparse.ArgumentParser(description="Бенчмарк алгоритму «Усі найближчі сусіди»")
    ap.add_argument("--quick", action="store_true", help="менші N і один повтор")
    ap.add_argument("--sizes", type=int, nargs="+", help="свої значення N")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--brute-max", type=int, default=4000, help="найбільше N для наївного алгоритму")
    ap.add_argument("--kinds", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results")
    args = ap.parse_args()

    sizes = args.sizes or ([250, 500, 1000, 2000, 4000] if args.quick
                           else [500, 1000, 2000, 4000, 8000, 16000])
    repeats = 1 if args.quick else args.repeats
    brute_max = min(args.brute_max, 2000) if args.quick else args.brute_max

    args.out.mkdir(parents=True, exist_ok=True)
    rows = run(sizes, brute_max, repeats, args.kinds)
    save_csv(rows, args.out / "results.csv")
    plot(rows, args.out)
    print_tables(rows, sizes)
    print(f"\nЗбережено в {args.out}")


if __name__ == "__main__":
    main()