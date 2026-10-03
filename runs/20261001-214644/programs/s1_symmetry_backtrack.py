"""Randomised backtracking over symmetric configurations.

Rows are filled from the outside in. With half-turn symmetry (rot2), choosing
two columns for row y also fixes row n-1-y, so only about n/2 decisions are
made. Every time a point is placed, all grid cells on a line through it and an
earlier point are marked as forbidden, so a later point can never complete a
collinear triple. A row whose remaining free cells drop below two prunes the
branch immediately. Each attempt has a node budget and restarts with a new
random order, which avoids getting stuck in one bad part of the tree.

Symmetries: rot2 (half turn; works for odd and even n), none (plain search).
Contract: construct(n, time_limit, seed) -> list[tuple[int, int]]
"""
import random
import time
from math import gcd


class _Out(Exception):
    """Raised to abandon an attempt (node budget or deadline reached)."""


class _Search:
    def __init__(self, n, rng, symmetric, budget, deadline):
        self.n, self.rng, self.sym = n, rng, symmetric
        self.budget, self.deadline = budget, deadline
        self.forbid = [0] * (n * n)
        self.colcount = [0] * n
        self.points = []
        self.deepest = []  # largest valid partial configuration seen
        self.nodes = 0

    def _line_cells(self, p, q):
        n = self.n
        dx, dy = q[0] - p[0], q[1] - p[1]
        g = gcd(dx, dy)
        dx, dy = dx // g, dy // g
        x, y = p
        while 0 <= x - dx < n and 0 <= y - dy < n:
            x, y = x - dx, y - dy
        cells = []
        while 0 <= x < n and 0 <= y < n:
            cells.append(x * n + y)
            x, y = x + dx, y + dy
        return cells

    def place(self, p):
        """Add p if allowed; return the undo record, or None if p is blocked."""
        n = self.n
        if self.forbid[p[0] * n + p[1]] or self.colcount[p[0]] >= 2:
            return None
        marked = []
        for q in self.points:
            for cell in self._line_cells(p, q):
                self.forbid[cell] += 1
                marked.append(cell)
        self.points.append(p)
        self.colcount[p[0]] += 1
        if len(self.points) > len(self.deepest):
            self.deepest = list(self.points)
        return marked

    def undo(self, p, marked):
        for cell in marked:
            self.forbid[cell] -= 1
        self.points.pop()
        self.colcount[p[0]] -= 1

    def free(self, y):
        n = self.n
        return [x for x in range(n) if not self.forbid[x * n + y] and self.colcount[x] < 2]

    def _actual_rows(self, rows):
        c = self.n - 1
        return {r for y in rows for r in ((y, c - y) if self.sym else (y,))}

    def pick_row(self, rows):
        """Most constrained remaining row, or None if some row or column can no longer be completed."""
        n = self.n
        best, best_free = None, None
        col_free = [0] * n
        for y in self._actual_rows(rows):
            free = self.free(y)
            if len(free) < 2:
                return None
            for x in free:
                col_free[x] += 1
            if y in rows and (best_free is None or len(free) < best_free):
                best, best_free = y, len(free)
        if any(self.colcount[x] + col_free[x] < 2 for x in range(n)):
            return None
        return best

    def solve(self, rows):
        """rows: remaining rows to fill (for rot2, one row of each mirrored pair)."""
        if not rows:
            return True
        self.nodes += 1
        if self.nodes > self.budget or (self.nodes & 255 == 0 and time.time() > self.deadline):
            raise _Out
        n, c = self.n, self.n - 1
        y = self.pick_row(rows)
        if y is None:
            return False
        rest = [r for r in rows if r != y]
        cand = self.free(y)
        pairs = [(a, b) for i, a in enumerate(cand) for b in cand[i + 1:]]
        self.rng.shuffle(pairs)
        for a, b in pairs:
            if self.sym and y == c - y:  # middle row of odd n: must be a mirrored pair
                if b != c - a:
                    continue
                new = [(a, y), (b, y)]
            elif self.sym:
                new = [(a, y), (b, y), (c - a, c - y), (c - b, c - y)]
            else:
                new = [(a, y), (b, y)]
            undo = []
            for p in new:
                marked = self.place(p)
                if marked is None:
                    break
                undo.append((p, marked))
            if len(undo) == len(new):
                if self.solve(rest):
                    return True
            for p, marked in reversed(undo):
                self.undo(p, marked)
        return False


def _row_order(n, symmetric):
    if symmetric:
        return list(range((n + 1) // 2))  # rows 0..h-1 (+ middle row for odd n); mirrors are implied
    order = []
    for i in range((n + 1) // 2):
        order += [i, n - 1 - i] if i != n - 1 - i else [i]
    return order


def construct(n, time_limit, seed):
    rng = random.Random(seed)
    deadline = time.time() + time_limit * 0.95
    best = []
    budget = 2000
    attempt = 0
    while time.time() < deadline:
        symmetric = attempt % 4 != 3  # mostly half-turn symmetric, sometimes plain
        attempt += 1
        search = _Search(n, rng, symmetric, budget, deadline)
        try:
            if search.solve(_row_order(n, symmetric)):
                return list(search.points)
        except _Out:
            pass
        if len(search.deepest) > len(best):
            best = search.deepest
        budget = int(budget * 1.1) + 100  # slowly allow deeper attempts
    return best
