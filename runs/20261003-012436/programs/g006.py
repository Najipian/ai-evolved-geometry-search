# g006 (parent s0): ILS on valid sets with numpy per-cell and per-point plane-cover counts: force cells in, evict conflicting points, regrow by min-kill greedy, and try (1,2)-swaps.
import math
import random
import time
from math import gcd

import numpy as np


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


def _valid(pts):
    pts = list(pts)
    if len(set(pts)) != len(pts):
        return False
    planes = set()
    k = len(pts)
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                key = _plane(pts[i], pts[j], pts[l])
                if key is None or key in planes:
                    return False
                planes.add(key)
    return True


def construct(n, time_limit, seed, start=None):
    t0 = time.time()
    deadline = t0 + time_limit * 0.9 - 0.3
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    N = n ** 3
    coords = np.array([(x, y, z) for x in range(n) for y in range(n) for z in range(n)],
                      dtype=np.float32)
    cap = 3 * n + 8

    def to_id(p):
        return (p[0] * n + p[1]) * n + p[2]

    def to_pt(c):
        c = int(c)
        return (c // (n * n), (c // n) % n, c % n)

    tri_cache = {}

    def triu(m):
        t = tri_cache.get(m)
        if t is None:
            t = np.triu_indices(m, 1)
            tri_cache[m] = t
        return t

    def contrib(c, others):
        m = len(others)
        U = coords[others] - coords[c]
        X = np.cross(U[:, None, :], U[None, :, :]).reshape(m * m, 3)
        T = X @ (coords - coords[c]).T
        Mb = (T == 0).reshape(m, m, N)
        Do = Mb.sum(1, dtype=np.int32) - 1
        cnt = Do.sum(0) // 2
        return Do, cnt

    class State:
        __slots__ = ("pts", "C", "D", "occ")

        def __init__(self):
            self.pts = []
            self.C = np.zeros(N, np.int32)
            self.D = np.zeros((cap, N), np.int32)
            self.occ = np.zeros(N, bool)

        def copy(self):
            s = State.__new__(State)
            s.pts = list(self.pts)
            s.C = self.C.copy()
            s.D = self.D.copy()
            s.occ = self.occ.copy()
            return s

        def add(self, c):
            c = int(c)
            k = len(self.pts)
            if k >= 2:
                Do, cnt = contrib(c, self.pts)
                self.C += cnt
                self.D[:k] += Do
                self.D[k] = cnt
            else:
                self.D[k] = 0
            self.pts.append(c)
            self.occ[c] = True

        def remove(self, i):
            c = self.pts[i]
            k = len(self.pts)
            others = self.pts[:i] + self.pts[i + 1:]
            if len(others) >= 2:
                Do, cnt = contrib(c, others)
                self.C -= cnt
                if i > 0:
                    self.D[:i] -= Do[:i]
                if i + 1 < k:
                    self.D[i + 1:k] -= Do[i:]
            last = k - 1
            if i != last:
                self.D[i] = self.D[last]
                self.pts[i] = self.pts[last]
            self.pts.pop()
            self.occ[c] = False

    KC = 40

    def choose(st, free):
        if len(free) == 1:
            return int(free[0])
        cand = free if len(free) <= KC else nprng.choice(free, KC, replace=False)
        if rng.random() < 0.05:
            return int(cand[rng.randrange(len(cand))])
        P = st.pts
        m = len(P)
        iu, ju = triu(m)
        S = coords[P]
        F = coords[free]
        bestc, bests = None, 1e18
        for c in cand:
            U = S - coords[c]
            X = np.cross(U[iu], U[ju])
            T = X @ (F - coords[c]).T
            kills = int((T == 0).any(0).sum())
            s = kills + rng.random() * 0.9
            if s < bests:
                bests, bestc = s, int(c)
        return bestc

    def grow(st, tabu):
        while time.time() < deadline:
            if len(st.pts) >= cap - 1:
                return
            free = np.flatnonzero((st.C == 0) & ~st.occ)
            if tabu and len(free):
                free = np.array([f for f in free if int(f) not in tabu], dtype=np.int64)
            if len(free) == 0:
                return
            st.add(choose(st, free))

    def try_12(st):
        k = len(st.pts)
        if k < 4:
            return None
        Z = (st.C[None, :] - st.D[:k]) == 0
        Z &= ~st.occ[None, :]
        cnts = Z.sum(1)
        order = [i for i in range(k) if cnts[i] >= 2]
        rng.shuffle(order)
        for i in order:
            R = np.flatnonzero(Z[i])
            if len(R) > 60:
                R = nprng.choice(R, 60, replace=False)
            others = st.pts[:i] + st.pts[i + 1:]
            S = coords[others]
            iu, ju = triu(len(others))
            RC = coords[R]
            idxs = list(range(len(R)))
            rng.shuffle(idxs)
            for a in idxs:
                c1 = R[a]
                U = S - coords[c1]
                X = np.cross(U[iu], U[ju])
                T = X @ (RC - coords[c1]).T
                bad = (T == 0).any(0)
                ok = np.flatnonzero(~bad)
                if len(ok):
                    c2 = R[ok[rng.randrange(len(ok))]]
                    return i, int(c1), int(c2)
        return None

    def improve(st):
        while time.time() < deadline:
            r = try_12(st)
            if r is None:
                return
            i, c1, c2 = r
            st.remove(i)
            st.add(c1)
            st.add(c2)
            grow(st, set())

    def force(st, c):
        removed = []
        while st.C[c] > 0 and len(st.pts) > 0:
            k = len(st.pts)
            col = st.D[:k, c]
            mx = col.max()
            ch = np.flatnonzero(col == mx)
            i = int(ch[rng.randrange(len(ch))])
            removed.append(st.pts[i])
            st.remove(i)
        st.add(c)
        return removed

    tabu_until = {}

    def pick_force(st, it):
        for _ in range(10):
            cells = nprng.integers(0, N, size=48)
            cells = [int(c) for c in cells if not st.occ[c] and tabu_until.get(int(c), -1) < it]
            if cells:
                break
        else:
            return None
        if rng.random() < 0.2:
            return cells[rng.randrange(len(cells))]
        vals = [st.C[c] + rng.random() * 0.8 for c in cells]
        return cells[int(np.argmin(vals))]

    # ---- initial state ----
    start_pts = [tuple(int(v) for v in p) for p in start] if start else []
    if start_pts and not _valid(start_pts):
        start_pts = []
    st = State()
    if len(start_pts) >= 4:
        for p in start_pts:
            st.add(to_id(p))
    else:
        while True:
            cand = [(rng.randrange(n), rng.randrange(n), rng.randrange(n)) for _ in range(4)]
            if _valid(cand):
                break
        for p in cand:
            st.add(to_id(p))
    grow(st, set())
    improve(st)

    best_pts = [to_pt(c) for c in st.pts]
    if start_pts and len(start_pts) > len(best_pts):
        best_pts = list(start_pts)
    best_state = st.copy()
    best_len = len(st.pts)

    it = 0
    last_imp = 0
    while time.time() < deadline:
        it += 1
        saved = st.copy()
        cur = len(st.pts)
        if it - last_imp > 400 and rng.random() < 0.01:
            st = best_state.copy()
            cur = len(st.pts)
            nforce = 3
        else:
            nforce = 1 if rng.random() < 0.8 else 2
        removed = []
        for _ in range(nforce):
            c = pick_force(st, it)
            if c is None:
                break
            removed += force(st, c)
        tenure = 3 + rng.randrange(max(2, cur // 3))
        for r_ in removed:
            tabu_until[r_] = it + tenure
        grow(st, set(removed))
        improve(st)
        L = len(st.pts)
        if L > best_len:
            best_len = L
            best_state = st.copy()
            last_imp = it
            cand_pts = [to_pt(c) for c in st.pts]
            if len(cand_pts) > len(best_pts) and _valid(cand_pts):
                best_pts = cand_pts
        if L >= cur:
            pass
        else:
            p = 0.08 if cur - L == 1 else 0.005
            if rng.random() >= p:
                st = saved

    if not _valid(best_pts):
        if start_pts:
            return start_pts
        return best_pts[:3]
    if start_pts and len(start_pts) > len(best_pts):
        return start_pts
    return best_pts
