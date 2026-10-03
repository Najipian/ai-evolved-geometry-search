"""Core math for the no-three-in-line problem.

A configuration is a list of (x, y) integer points on an n x n grid
(0 <= x, y < n). It is valid when no three points lie on a common line.
The maximum possible is 2n points (at most two per row).
"""
from collections import defaultdict
from math import gcd


def direction(dx, dy):
    """Canonical direction of the vector (dx, dy), identical for opposite vectors."""
    g = gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return dx, dy


def count_triples(points):
    """Number of collinear triples. Each triple is counted once, by its first point."""
    pts = list(points)
    total = 0
    for i, (px, py) in enumerate(pts):
        groups = defaultdict(int)
        for qx, qy in pts[i + 1:]:
            groups[direction(qx - px, qy - py)] += 1
        total += sum(c * (c - 1) // 2 for c in groups.values())
    return total


def involvement(points):
    """For each point, how many collinear triples it belongs to."""
    pts = list(points)
    result = []
    for i, (px, py) in enumerate(pts):
        groups = defaultdict(int)
        for j, (qx, qy) in enumerate(pts):
            if i != j:
                groups[direction(qx - px, qy - py)] += 1
        result.append(sum(c * (c - 1) // 2 for c in groups.values()))
    return result


def clean(points, n):
    """Drop out-of-grid, malformed and duplicate points."""
    seen = set()
    out = []
    for p in points:
        try:
            x, y = int(p[0]), int(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= x < n and 0 <= y < n and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    return out


def repair(points):
    """Greedily remove the most conflicted point until no three are collinear."""
    pts = list(points)
    while True:
        inv = involvement(pts)
        worst = max(range(len(pts)), key=inv.__getitem__, default=None)
        if worst is None or inv[worst] == 0:
            return pts
        pts.pop(worst)


def is_valid(points, n):
    pts = clean(points, n)
    return len(pts) == len(points) and count_triples(pts) == 0


def score(points, n):
    """Score any submission: the size of its valid (repaired) subset relative to 2n."""
    pts = clean(points, n)
    triples = count_triples(pts)
    valid = pts if triples == 0 else repair(pts)
    return {
        "n": n,
        "submitted": len(points),
        "triples": triples,
        "valid_points": len(valid),
        "target": 2 * n,
        "ratio": len(valid) / (2 * n),
        "solved": len(valid) == 2 * n,
        "points": valid,
    }
