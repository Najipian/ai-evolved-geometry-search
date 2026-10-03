"""Symmetry-restricted simulated annealing.

Every 2n solution has exactly two points in every row and every column. The
moves below keep that true at all times, and also keep the configuration
symmetric, so the search only has to remove collinear triples.

Symmetries (c = n - 1):
  rot2  half turn (x, y) -> (c-x, c-y). Rows y and c-y mirror each other; for odd n
        the middle row holds a mirrored pair (a, c-a). Move: swap the columns of two
        points and apply the same swap to their mirror images.
  dia1  mirror in the main diagonal (x, y) -> (y, x), no points on the diagonal.
        The configuration is a 2-regular graph on the n rows: edge {a, b} is the
        two points (a, b) and (b, a). Move: rewire two edges (a 2-switch).
  none  no symmetry: swap the columns of two points in different rows.

The symmetric searches have far fewer degrees of freedom, so they converge much
faster, at the price of only finding symmetric solutions.

Contract: construct(n, time_limit, seed) -> list[tuple[int, int]]
"""
import math
import random
import time
from math import gcd


def _dir(dx, dy):
    g = gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        return -dx, -dy
    return dx, dy


_TABLE = {}


def _table(n):
    """Direction id for every offset (dx, dy), indexed by (dx + n) * (2n + 1) + (dy + n)."""
    if n not in _TABLE:
        w = 2 * n + 1
        ids, table = {}, [0] * (w * w)
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                if dx or dy:
                    table[(dx + n) * w + dy + n] = ids.setdefault(_dir(dx, dy), len(ids))
        _TABLE[n] = table
    return _TABLE[n]


_N = [0]  # grid size currently being searched


def _pairs(p, pts, skip):
    """Collinear pairs through p among pts, ignoring indices in skip."""
    n = _N[0]
    table, w = _table(n), 2 * n + 1
    base = (n - p[0]) * w + n - p[1]
    seen = {}
    pairs = 0
    for k, (qx, qy) in enumerate(pts):
        if k in skip:
            continue
        d = table[base + qx * w + qy]
        c = seen.get(d, 0)
        pairs += c
        seen[d] = c + 1
    return pairs


def _conflicted(pts):
    """Indices of points that lie on a line with at least two other points."""
    n = _N[0]
    table, w = _table(n), 2 * n + 1
    bad = []
    for i, (px, py) in enumerate(pts):
        base = (n - px) * w + n - py
        seen = set()
        for k, (qx, qy) in enumerate(pts):
            if k != i:
                d = table[base + qx * w + qy]
                if d in seen:
                    bad.append(i)
                    break
                seen.add(d)
    return bad


def _total(pts):
    return sum(_pairs(p, pts[i + 1:], ()) for i, p in enumerate(pts))


def _move_delta(pts, idxs, new):
    """Change in collinear triples if pts[idxs] are replaced by new. Leaves pts updated."""
    removed = set()
    before = 0
    for i in idxs:
        removed.add(i)
        before += _pairs(pts[i], pts, removed)
    after = 0
    for i, p in zip(idxs, new):
        pts[i] = p
        after += _pairs(p, pts, removed)
        removed.discard(i)
    return after - before


# ---------- rot2: half-turn symmetry ----------

def _rot2_init(n, rng):
    c, h = n - 1, n // 2
    for _ in range(1000):
        entries = []
        mid_a = None
        if n % 2:
            mid_a = rng.choice([x for x in range(n) if x != h])
            entries.append(h)
        for x in range(h):
            need = 1 if mid_a in (x, c - x) else 2
            entries += [rng.choice((x, c - x)) for _ in range(need)]
        rng.shuffle(entries)
        if all(entries[2 * y] != entries[2 * y + 1] for y in range(h)):
            cols = [0] * (2 * n)
            for k, col in enumerate(entries):
                cols[k] = col
                cols[_rot2_mirror(k, n)] = c - col
            if n % 2:
                cols[2 * h], cols[2 * h + 1] = mid_a, c - mid_a
            return cols
    return None


def _rot2_mirror(k, n):
    y, s = divmod(k, 2)
    c = n - 1
    return 2 * y + (1 - s) if y == c - y else 2 * (c - y) + s


def _rot2_search(n, rng, deadline):
    _N[0] = n
    c = n - 1
    cols = _rot2_init(n, rng)
    if cols is None:
        return None, []
    pts = [(cols[k], k // 2) for k in range(2 * n)]
    cost = _total(pts)

    def propose(focus):
        i, j = (focus if focus is not None else rng.randrange(2 * n)), rng.randrange(2 * n)
        mi, mj = _rot2_mirror(i, n), _rot2_mirror(j, n)
        if len({i, j, mi, mj}) < 4 or i // 2 == j // 2:
            return None
        ci, cj = pts[i][0], pts[j][0]
        if ci == cj:
            return None
        change = {i: cj, j: ci, mi: c - cj, mj: c - ci}
        for k in change:  # each touched row must keep two distinct columns
            other = k ^ 1
            if change[k] == change.get(other, pts[other][0]):
                return None
        idxs = list(change)
        return idxs, [(change[k], k // 2) for k in idxs]

    return _anneal(pts, cost, propose, rng, deadline, 400 * n * n)


# ---------- dia1: diagonal mirror ----------

def _dia1_search(n, rng, deadline):
    _N[0] = n
    order = list(range(n))
    rng.shuffle(order)
    edges = [(order[i], order[(i + 1) % n]) for i in range(n)]  # a Hamiltonian cycle
    present = {frozenset(e) for e in edges}
    pts = []
    for a, b in edges:
        pts += [(a, b), (b, a)]
    cost = _total(pts)

    def propose(focus):
        e, f = (focus // 2 if focus is not None else rng.randrange(n)), rng.randrange(n)
        if e == f:
            return None
        a, b = pts[2 * e]
        u, v = pts[2 * f]
        if len({a, b, u, v}) < 4:
            return None
        if rng.random() < 0.5:
            u, v = v, u
        ne, nf = (a, u), (b, v)
        if frozenset(ne) in present or frozenset(nf) in present:
            return None

        def commit():
            present.discard(frozenset((a, b)))
            present.discard(frozenset((u, v)))
            present.add(frozenset(ne))
            present.add(frozenset(nf))

        idxs = [2 * e, 2 * e + 1, 2 * f, 2 * f + 1]
        return idxs, [ne, ne[::-1], nf, nf[::-1]], commit

    return _anneal(pts, cost, propose, rng, deadline, 400 * n * n)


# ---------- none: plain two-per-row/column search ----------

def _plain_search(n, rng, deadline):
    _N[0] = n
    while True:
        cols = list(range(n)) * 2
        rng.shuffle(cols)
        if all(cols[2 * r] != cols[2 * r + 1] for r in range(n)):
            break
    pts = [(cols[k], k // 2) for k in range(2 * n)]
    cost = _total(pts)

    def propose(focus):
        i, j = (focus if focus is not None else rng.randrange(2 * n)), rng.randrange(2 * n)
        if i // 2 == j // 2:
            return None
        ci, cj = pts[i][0], pts[j][0]
        if ci == cj or pts[i ^ 1][0] == cj or pts[j ^ 1][0] == ci:
            return None
        return [i, j], [(cj, i // 2), (ci, j // 2)]

    return _anneal(pts, cost, propose, rng, deadline, 2000 * n * n)


# ---------- shared annealing loop ----------

def _anneal(pts, cost, propose, rng, deadline, max_steps):
    best_cost, best = cost, list(pts)
    temp = 2.0
    bad, bad_cost = _conflicted(pts), cost
    for step in range(max_steps):
        if cost == 0 or (step & 127 == 0 and time.time() > deadline):
            break
        if cost != bad_cost or step % 64 == 0:
            bad, bad_cost = _conflicted(pts), cost
        focus = rng.choice(bad) if bad and rng.random() < 0.8 else None
        move = propose(focus)
        if move is None:
            continue
        idxs, new = move[0], move[1]
        old = [pts[i] for i in idxs]
        delta = _move_delta(pts, idxs, new)
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            cost += delta
            if len(move) > 2:
                move[2]()
            if cost < best_cost:
                best_cost, best = cost, list(pts)
        else:
            for i, p in zip(idxs, old):
                pts[i] = p
        temp = max(0.3, temp * 0.9999)
    return best_cost, best


SEARCHES = {"rot2": _rot2_search, "dia1": _dia1_search, "none": _plain_search}


def construct(n, time_limit, seed):
    rng = random.Random(seed)
    deadline = time.time() + time_limit * 0.95
    names = ["rot2", "rot2", "dia1", "none"]
    best_cost, best_pts = None, []
    while time.time() < deadline:
        cost, pts = SEARCHES[rng.choice(names)](n, rng, deadline)
        if cost is not None and (best_cost is None or cost < best_cost):
            best_cost, best_pts = cost, pts
        if cost == 0:
            break
    return best_pts
