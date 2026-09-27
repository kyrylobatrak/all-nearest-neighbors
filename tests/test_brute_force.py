"""Тести наївного алгоритму"""

import random
import pytest
from brute_force import all_nearest_neighbors_brute, nn_distances2
from geometry import Point, dist2


def P(*coords):
    """Короткий запис списку точок: P(0,0, 3,4) -> [Point(0,0), Point(3,4)]."""
    return [Point(coords[k], coords[k + 1]) for k in range(0, len(coords), 2)]


# ---------- граничні випадки ----------

def test_empty():
    assert all_nearest_neighbors_brute([]) == []


def test_single_point_has_no_neighbor():
    assert all_nearest_neighbors_brute(P(5, 5)) == [None]


def test_two_points_are_mutual_neighbors():
    assert all_nearest_neighbors_brute(P(0, 0, 3, 4)) == [1, 0]


# ---------- відомі відповіді ----------

def test_small_known_example():
    #   0:(0,0)  1:(1,0)  2:(10,0)  3:(10,2)
    pts = P(0, 0, 1, 0, 10, 0, 10, 2)
    assert all_nearest_neighbors_brute(pts) == [1, 0, 3, 2]


def test_relation_is_not_symmetric():
    # 0:(0,0)  1:(2,0)  2:(5,0): сусід 2 — це 1, але сусід 1 — це 0
    pts = P(0, 0, 2, 0, 5, 0)
    assert all_nearest_neighbors_brute(pts) == [1, 0, 1]


def test_one_point_is_neighbor_of_many():
    # центр і 4 точки навколо на різних відстанях: для всіх зовнішніх сусід — центр
    pts = P(0, 0, 10, 0, 0, 11, -12, 0, 0, -13)
    nn = all_nearest_neighbors_brute(pts)
    assert nn[1:] == [0, 0, 0, 0]
    assert nn[0] == 1


# ---------- вироджені випадки ----------

def test_duplicates_are_neighbors_with_zero_distance():
    pts = P(0, 0, 7, 7, 0, 0, 100, 100)
    nn = all_nearest_neighbors_brute(pts)
    assert nn[0] == 2 and nn[2] == 0
    assert nn_distances2(pts, nn)[0] == 0


def test_collinear_points():
    pts = P(0, 0, 1, 0, 3, 0, 6, 0, 10, 0)
    assert all_nearest_neighbors_brute(pts) == [1, 0, 1, 2, 3]


def test_ties_resolved_by_smallest_index():
    # центр рівновіддалений від 4 точок: береться найменший індекс
    pts = P(5, 5, 5, 6, 6, 5, 5, 4, 4, 5)
    assert all_nearest_neighbors_brute(pts)[0] == 1


def test_negative_coordinates():
    pts = P(-100, -100, -99, -100, 50, 50)
    assert all_nearest_neighbors_brute(pts) == [1, 0, 1]


# ---------- властивості на випадкових даних ----------

@pytest.mark.parametrize("n, seed", [(2, 0), (5, 1), (30, 2), (200, 3)])
def test_result_is_minimal_over_all_other_points(n, seed):
    rnd = random.Random(seed)
    pts = [Point(rnd.randint(-50, 50), rnd.randint(-50, 50)) for _ in range(n)]
    nn = all_nearest_neighbors_brute(pts)
    for i, j in enumerate(nn):
        assert j is not None and j != i
        d = dist2(pts[i], pts[j])
        assert all(d <= dist2(pts[i], pts[k]) for k in range(n) if k != i)


def test_distances_do_not_depend_on_input_order():
    rnd = random.Random(7)
    pts = [Point(rnd.randint(0, 1000), rnd.randint(0, 1000)) for _ in range(150)]
    d1 = dict(zip(pts, nn_distances2(pts, all_nearest_neighbors_brute(pts))))
    shuffled = pts[:]
    rnd.shuffle(shuffled)
    d2 = dict(zip(shuffled, nn_distances2(shuffled, all_nearest_neighbors_brute(shuffled))))
    assert d1 == d2
