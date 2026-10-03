"""Exact verifier for the no-four-coplanar problem in the n x n x n grid (OEIS A280537).

A configuration is a list of (x, y, z) integer points with 0 <= x, y, z < n. It is
valid when no four points lie on a common plane. Three collinear points plus any
fourth point are always coplanar, so a valid set (of 4+ points) also has no three
points on a line.
"""
from collections import defaultdict
from itertools import combinations
from math import gcd

# Best known values: exact for n <= 6, numerical evidence for 7-8, lower bounds above
# (OEIS A280537, read 2026-10-02; a(12) >= 31 by A. Kudriashov, 2026-08-23).
RECORDS = {2: 5, 3: 8, 4: 10, 5: 13, 6: 16, 7: 18, 8: 20, 9: 23, 10: 26, 11: 28,
           12: 31, 13: 32, 14: 35, 15: 36, 16: 38, 17: 42}


def _sub(p, q):
    return p[0] - q[0], p[1] - q[1], p[2] - q[2]


def _cross(u, v):
    return u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]


def plane_key(p, q, r):
    """Canonical (a, b, c, d) of the plane ax + by + cz = d through p, q, r; None if collinear."""
    a, b, c = _cross(_sub(q, p), _sub(r, p))
    if a == b == c == 0:
        return None
    g = gcd(gcd(a, b), c)
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
        a, b, c = -a, -b, -c
    return a, b, c, a * p[0] + b * p[1] + c * p[2]


def violations(points):
    """Bad planes (4+ points) and collinear triples, each as a set of point indices."""
    planes = defaultdict(set)
    bad = []
    for i, j, k in combinations(range(len(points)), 3):
        key = plane_key(points[i], points[j], points[k])
        if key is None:
            bad.append({i, j, k})
        else:
            planes[key].update((i, j, k))
    bad += [s for s in planes.values() if len(s) >= 4]
    return bad


def clean(points, n):
    seen, out = set(), []
    for p in points:
        try:
            x, y, z = int(p[0]), int(p[1]), int(p[2])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= x < n and 0 <= y < n and 0 <= z < n and (x, y, z) not in seen:
            seen.add((x, y, z))
            out.append((x, y, z))
    return out


def repair(points):
    """Greedily drop the point in the most violations until none remain."""
    pts = list(points)
    while len(pts) >= 4:
        bad = violations(pts)
        if not bad:
            break
        count = defaultdict(int)
        for s in bad:
            for i in s:
                count[i] += 1
        pts.pop(max(count, key=count.get))
    return pts


def is_valid(points, n):
    pts = clean(points, n)
    return len(pts) == len(points) and (len(pts) < 4 or not violations(pts))


def score(points, n):
    """valid_points = size of the repaired set; target = best known value; solved = a new record."""
    pts = clean(points, n)
    bad = violations(pts) if len(pts) >= 4 else []
    valid = repair(pts) if bad else pts
    record = RECORDS.get(n, 2 * n)
    return {
        "n": n,
        "submitted": len(points),
        "triples": len(bad),
        "valid_points": len(valid),
        "target": record,
        "ratio": len(valid) / record,
        "solved": len(valid) > record,
        "points": valid,
    }
