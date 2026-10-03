"""Baseline: add random grid points while no three become collinear; keep the best attempt."""
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


def _fits(p, pts):
    seen = set()
    for q in pts:
        d = _dir(q[0] - p[0], q[1] - p[1])
        if d in seen:
            return False
        seen.add(d)
    return True


def construct(n, time_limit, seed):
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    cells = [(x, y) for x in range(n) for y in range(n)]
    best = []
    while time.time() < deadline and len(best) < 2 * n:
        rng.shuffle(cells)
        pts = []
        for p in cells:
            if _fits(p, pts):
                pts.append(p)
        if len(pts) > len(best):
            best = pts
    return best
