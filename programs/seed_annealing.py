"""Seed program: simulated annealing with exactly two points per row and column.

Program contract (every evolved program follows it):
    construct(n: int, time_limit: float, seed: int) -> list[tuple[int, int]]
Return points on the n x n grid; the evaluator verifies and scores them.
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


def _pairs(p, pts, skip_a, skip_b):
    """Collinear pairs (q, r) with p, among pts minus two skipped indices."""
    px, py = p
    groups = {}
    for k, (qx, qy) in enumerate(pts):
        if k == skip_a or k == skip_b:
            continue
        d = _dir(qx - px, qy - py)
        groups[d] = groups.get(d, 0) + 1
    return sum(c * (c - 1) // 2 for c in groups.values())


def _initial(n, rng):
    """Each column used exactly twice, two distinct columns per row."""
    while True:
        cols = list(range(n)) * 2
        rng.shuffle(cols)
        if all(cols[2 * r] != cols[2 * r + 1] for r in range(n)):
            return cols


def _total(pts):
    total = 0
    for i, (px, py) in enumerate(pts):
        groups = {}
        for qx, qy in pts[i + 1:]:
            d = _dir(qx - px, qy - py)
            groups[d] = groups.get(d, 0) + 1
        total += sum(c * (c - 1) // 2 for c in groups.values())
    return total


def _anneal(n, rng, deadline, steps):
    cols = _initial(n, rng)
    pts = [(i // 2, cols[i]) for i in range(2 * n)]
    cost = _total(pts)
    best_cost, best_pts = cost, list(pts)
    temp = 2.0
    for step in range(steps):
        if cost == 0 or (step & 255 == 0 and time.time() > deadline):
            break
        i = rng.randrange(2 * n)
        j = rng.randrange(2 * n)
        ri, rj = i // 2, j // 2
        if ri == rj:
            continue
        ci, cj = pts[i][1], pts[j][1]
        if ci == cj or pts[i ^ 1][1] == cj or pts[j ^ 1][1] == ci:
            continue
        # Swap the columns of points i and j (keeps every column count at two).
        before = _pairs(pts[i], pts, i, -1) + _pairs(pts[j], pts, i, j)
        new_i, new_j = (ri, cj), (rj, ci)
        old_i, old_j = pts[i], pts[j]
        pts[i] = new_i
        after = _pairs(new_i, pts, i, j)
        pts[j] = new_j
        after += _pairs(new_j, pts, j, -1)
        delta = after - before
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            cost += delta
            if cost < best_cost:
                best_cost, best_pts = cost, list(pts)
        else:
            pts[i], pts[j] = old_i, old_j
        temp = max(0.3, temp * 0.9999)
    return best_cost, best_pts


def construct(n, time_limit, seed):
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    best_cost, best_pts = None, []
    while time.time() < deadline:
        cost, pts = _anneal(n, rng, deadline, steps=2000 * n * n)
        if best_cost is None or cost < best_cost:
            best_cost, best_pts = cost, pts
        if cost == 0:
            break
    return best_pts
