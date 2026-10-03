"""Seed program for no-four-coplanar in the n x n x n grid (OEIS A280537).

Simulated annealing on a set of k points. The cost counts violations: for every
plane holding m >= 4 points, C(m, 4) bad quadruples, plus a penalty for every
collinear triple. When the cost reaches zero the set is valid, it is saved, and
a new random point is added (k + 1) to push further.

Contract: construct(n, time_limit, seed) -> list[tuple[int, int, int]]
"""
import math
import random
import time
from math import gcd

COLLINEAR_PENALTY = 3


def _plane(q, a, b):
    ux, uy, uz = a[0] - q[0], a[1] - q[1], a[2] - q[2]
    vx, vy, vz = b[0] - q[0], b[1] - q[1], b[2] - q[2]
    x, y, z = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    if x == y == z == 0:
        return None
    g = gcd(gcd(x, y), z)
    x, y, z = x // g, y // g, z // g
    if x < 0 or (x == 0 and (y < 0 or (y == 0 and z < 0))):
        x, y, z = -x, -y, -z
    return x, y, z, x * q[0] + y * q[1] + z * q[2]


def _added_cost(q, pts, skip):
    """Violations created by adding point q to pts (ignoring index skip)."""
    others = [p for k, p in enumerate(pts) if k != skip]
    planes = {}
    collinear = 0
    for i in range(len(others)):
        a = others[i]
        for j in range(i + 1, len(others)):
            key = _plane(q, a, others[j])
            if key is None:
                collinear += 1
            else:
                s = planes.get(key)
                if s is None:
                    planes[key] = {i, j}
                else:
                    s.add(i)
                    s.add(j)
    cost = COLLINEAR_PENALTY * collinear
    for s in planes.values():
        m = len(s)
        if m >= 3:
            cost += m * (m - 1) * (m - 2) // 6
    return cost


def _total_cost(pts):
    cost, cur = 0, []
    for p in pts:
        cost += _added_cost(p, cur, -1)
        cur.append(p)
    return cost


def _random_free(n, taken, rng):
    while True:
        p = (rng.randrange(n), rng.randrange(n), rng.randrange(n))
        if p not in taken:
            return p


def construct(n, time_limit, seed):
    rng = random.Random(seed)
    deadline = time.time() + time_limit * 0.95
    k = max(4, 3 * n // 2)  # start below the known values (about 2.5n) so a valid set is found early
    pts = []
    taken = set()
    while len(pts) < k:
        p = _random_free(n, taken, rng)
        pts.append(p)
        taken.add(p)
    cost = _total_cost(pts)
    best = []
    temp = 1.5
    step = 0
    while time.time() < deadline:
        step += 1
        if cost == 0:
            best = list(pts)
            p = _random_free(n, taken, rng)  # grow by one point and keep going
            cost += _added_cost(p, pts, -1)
            pts.append(p)
            taken.add(p)
            temp = 1.5
            continue
        # move the worst of three sampled points to a random free cell
        cands = [rng.randrange(len(pts)) for _ in range(3)]
        costs = [_added_cost(pts[i], pts, i) for i in cands]
        i = cands[costs.index(max(costs))]
        old_cost = max(costs)
        new = _random_free(n, taken, rng)
        new_cost = _added_cost(new, pts, i)
        delta = new_cost - old_cost
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            taken.discard(pts[i])
            pts[i] = new
            taken.add(new)
            cost += delta
        temp = max(0.2, temp * 0.9995)
    # No valid set reached yet: return the current one; the evaluator repairs it by removing points.
    return best if best else pts
