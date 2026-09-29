"""Тести структури Quad-Edge"""

from geometry import Point
from quadedge import connect, delete_edge, iter_edges, make_edge, splice

A, B, C, D = Point(0, 0), Point(4, 0), Point(0, 3), Point(4, 3)

def ring(e, step):
    """Орбіта ребра e за переходом step (наприклад, lnext) до повернення в e."""
    out, cur = [e], step(e)
    while cur is not e:
        out.append(cur)
        cur = step(cur)
        assert len(out) < 100, "орбіта не замкнулася"
    return out

def build_triangle():
    """Трикутник A→B→C, обхід проти год. стрілки. Повертає ребра ab, bc, ca."""
    ab = make_edge(A, B)
    bc = make_edge(B, C)
    splice(ab.sym, bc)          # з'єднати ab і bc у вершині B
    ca = connect(bc, ab)        # замкнути трикутник ребром C → A
    return ab, bc, ca

# ---------- алгебраїчні тотожності ----------

def test_make_edge_endpoints():
    e = make_edge(A, B)
    assert e.org == A and e.dest == B
    assert e.sym.org == B and e.sym.dest == A

def test_rot_has_order_four_and_sym_is_involution():
    e = make_edge(A, B)
    assert e.rot.rot.rot.rot is e
    assert e.sym.sym is e
    assert e.rot.inv_rot is e and e.inv_rot.rot is e
    assert len({id(e), id(e.rot), id(e.sym), id(e.inv_rot)}) == 4

def test_isolated_edge_navigation():
    e = make_edge(A, B)
    assert e.onext is e and e.oprev is e       # у вершині A лише одне ребро
    assert e.lnext is e.sym                    # обхід грані: A→B, потім B→A
    assert e.lprev is e.sym
    assert e.rnext is e.sym and e.rprev is e.sym
    assert e.dnext is e and e.dprev is e

def test_navigation_identities_on_triangle():
    for e0 in build_triangle():
        for e in (e0, e0.sym):
            assert e.onext.oprev is e and e.oprev.onext is e
            assert e.lnext.lprev is e and e.lprev.lnext is e
            assert e.rnext.rprev is e and e.rprev.rnext is e
            assert e.dnext.dprev is e and e.dprev.dnext is e
            assert e.lnext.org == e.dest           # ліва грань: кінець → початок наступного

# ---------- splice / connect ----------

def test_splice_is_involution():
    ab = make_edge(A, B)
    ac = make_edge(A, C)
    splice(ab, ac)
    assert ab.onext is ac and ac.onext is ab    # спільне кільце навколо A
    splice(ab, ac)
    assert ab.onext is ab and ac.onext is ac    # знову два окремі ребра

def test_triangle_faces():
    ab, bc, ca = build_triangle()
    assert ca.org == C and ca.dest == A
    inner = ring(ab, lambda e: e.lnext)
    assert inner == [ab, bc, ca]                        # внутрішня грань — 3 ребра
    outer = ring(ab.sym, lambda e: e.lnext)
    assert {id(x) for x in outer} == {id(ab.sym), id(bc.sym), id(ca.sym)}

def test_vertex_degrees_in_triangle():
    ab, bc, ca = build_triangle()
    for e in (ab, bc, ca):
        assert len(ring(e, lambda x: x.onext)) == 2     # кожна вершина степеня 2

def test_connect_sets_left_face():
    ab = make_edge(A, B)
    bc = make_edge(B, C)
    splice(ab.sym, bc)
    e = connect(bc, ab)
    assert e.lprev is bc and e.lnext is ab

def test_splice_builds_ring_in_given_order():
    # вставляємо ребра з A в порядку проти год. стрілки: схід, північний схід, північ
    ab = make_edge(A, B)            # напрям 0°
    ad = make_edge(A, D)            # ~37°
    ac = make_edge(A, C)            # 90°
    splice(ab, ad)
    splice(ad, ac)
    assert [e.dest for e in ring(ab, lambda e: e.onext)] == [B, D, C]

# ---------- delete_edge ----------

def test_delete_edge_opens_triangle():
    ab, bc, ca = build_triangle()
    delete_edge(ca)
    assert ca.onext is ca and ca.sym.onext is ca.sym    # ребро стало ізольованим
    assert ab.onext is ab                               # у A лишилося одне ребро
    assert ab.lnext is bc                               # ланцюжок A→B→C цілий
    assert len(list(iter_edges(ab))) == 2

# ---------- iter_edges ----------

def test_iter_edges_counts_each_edge_once():
    assert len(list(iter_edges(make_edge(A, B)))) == 1
    ab, bc, ca = build_triangle()
    edges = list(iter_edges(ab))
    assert len(edges) == 3
    assert {frozenset((e.org, e.dest)) for e in edges} == {
        frozenset((A, B)), frozenset((B, C)), frozenset((C, A))
    }

def test_iter_edges_from_any_start():
    ab, bc, ca = build_triangle()
    for start in (ab, bc.sym, ca):
        assert len(list(iter_edges(start))) == 3