# g005 (parent g004): Replace backtracking with a plateau/tabu local search over symmetry orbits (rot4, dia2, rot2), resumed round-robin; then grow the best asymmetrically.
import time
import random
from math import gcd
import numpy as np


def _bits(m):
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


def _group_maps(n, name):
    m = n - 1
    idm = lambda x, y: (x, y)
    r90 = lambda x, y: (y, m - x)
    r180 = lambda x, y: (m - x, m - y)
    r270 = lambda x, y: (m - y, x)
    d1 = lambda x, y: (y, x)
    d2 = lambda x, y: (m - y, m - x)
    return {
        'rot4': [idm, r90, r180, r270],
        'dia2': [idm, r180, d1, d2],
        'rot2': [idm, r180],
        'dia1': [idm, d1],
        'iden': [idm],
    }[name]


class _Model(object):
    def __init__(self, n, name):
        self.n = n
        self.lcache = {}
        maps = _group_maps(n, name)
        orb_of = [-1] * (n * n)
        seen = [False] * (n * n)
        orbits = []
        for x in range(n):
            for y in range(n):
                c = x * n + y
                if seen[c]:
                    continue
                cells = sorted(set(f(x, y) for f in maps))
                for (a, b) in cells:
                    seen[a * n + b] = True
                if self._valid(cells):
                    oid = len(orbits)
                    orbits.append(cells)
                    for (a, b) in cells:
                        orb_of[a * n + b] = oid
        self.orb_of = orb_of
        self.orbits = orbits
        N = len(orbits)
        self.N = N
        self.size = [len(c) for c in orbits]
        internal = [0] * N
        for oid, cells in enumerate(orbits):
            p = cells[0]
            msk = 0
            for q in cells[1:]:
                msk |= self.line(p, q)
            internal[oid] = msk & ~(1 << oid)
        hits = [0] * N
        for a in range(N):
            for b in _bits(internal[a]):
                hits[b] |= 1 << a
        self.internal, self.hits = internal, hits
        self.cache = [None] * (N * N)

    def _valid(self, cells):
        rc, cc = {}, {}
        for (a, b) in cells:
            rc[a] = rc.get(a, 0) + 1
            cc[b] = cc.get(b, 0) + 1
            if rc[a] > 2 or cc[b] > 2:
                return False
        L = len(cells)
        for i in range(L):
            ax, ay = cells[i]
            for j in range(i + 1, L):
                bx, by = cells[j]
                for k in range(j + 1, L):
                    cx, cy = cells[k]
                    if (bx - ax) * (cy - ay) == (by - ay) * (cx - ax):
                        return False
        return True

    def line(self, p, q):
        n = self.n
        px, py = p
        dx, dy = q[0] - px, q[1] - py
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        x, y = px, py
        while 0 <= x - dx < n and 0 <= y - dy < n:
            x -= dx
            y -= dy
        key = (dx, dy, x, y)
        lc = self.lcache
        v = lc.get(key)
        if v is not None:
            return v
        orb_of = self.orb_of
        msk = 0
        while 0 <= x < n and 0 <= y < n:
            o = orb_of[x * n + y]
            if o >= 0:
                msk |= 1 << o
            x += dx
            y += dy
        lc[key] = msk
        return msk

    def compat(self, a, b):
        N = self.N
        v = self.cache[a * N + b]
        if v is not None:
            return v
        p = self.orbits[a][0]
        v = 0
        for q in self.orbits[b]:
            v |= self.line(p, q)
        self.cache[a * N + b] = v
        self.cache[b * N + a] = v
        return v


class _OLS(object):
    """Plateau / tabu local search over symmetry orbits."""

    def __init__(self, model, rng):
        self.m = model
        self.rng = rng
        N = model.N
        self.N = N
        self.target = 2 * model.n
        self.sz = np.array(model.size, dtype=np.int64)
        self.intA = [np.array(_bits(model.internal[a]), dtype=np.intp) for a in range(N)]
        self.hitA = [np.array(_bits(model.hits[a]), dtype=np.intp) for a in range(N)]
        self.ih = [model.internal[a] | model.hits[a] for a in range(N)]
        self.pidx = {}
        self.tabu = np.zeros(N, np.int64)
        self.it = 0
        self.since = 0
        self.nper = 0
        self.maxsz = max(model.size) if N else 1
        avg = float(sum(model.size)) / max(1, N)
        self.plateau = max(30, int(3 * self.target / avg))
        self.R = 40
        self.best = []
        self.bestpts = 0
        self.reset()

    def reset(self):
        self.conf = np.zeros(self.N, np.int64)
        self.inP = np.zeros(self.N, bool)
        self.P = []
        self.pts = 0

    def parr(self, a, b):
        N = self.N
        key = a * N + b if a < b else b * N + a
        arr = self.pidx.get(key)
        if arr is None:
            arr = np.array(_bits(self.m.compat(a, b)), dtype=np.intp)
            self.pidx[key] = arr
        return arr

    def add(self, a):
        conf = self.conf
        for b in self.P:
            conf[self.parr(a, b)] += 1
        conf[self.intA[a]] += 1
        conf[self.hitA[a]] += 1
        self.P.append(a)
        self.inP[a] = True
        self.pts += self.m.size[a]

    def remove(self, a):
        self.P.remove(a)
        self.inP[a] = False
        conf = self.conf
        for b in self.P:
            conf[self.parr(a, b)] -= 1
        conf[self.intA[a]] -= 1
        conf[self.hitA[a]] -= 1
        self.pts -= self.m.size[a]

    def culprits(self, c):
        bit = 1 << c
        P = self.P
        sing = [a for a in P if self.ih[a] & bit]
        pairs = []
        comp = self.m.compat
        L = len(P)
        for i in range(L):
            a = P[i]
            for j in range(i + 1, L):
                b = P[j]
                if (comp(a, b) >> c) & 1:
                    pairs.append((a, b))
        return sing, pairs

    def upd(self):
        if self.pts > self.bestpts:
            self.bestpts = self.pts
            self.best = list(self.P)
            self.nper = 0

    def points(self):
        out = []
        for o in self.best:
            out.extend(self.m.orbits[o])
        return [(int(x), int(y)) for (x, y) in out]

    def run(self, deadline):
        rng = self.rng
        tm = time.time
        sz = self.sz
        tabu = self.tabu
        target = self.target
        if self.bestpts >= target:
            return True
        while True:
            self.it += 1
            it = self.it
            if (it & 3) == 0 and tm() > deadline:
                return False
            conf = self.conf
            free = ~self.inP
            z = np.flatnonzero(free & (conf == 0))
            if z.size:
                s = sz[z]
                z = z[s == s.max()]
                self.add(int(z[rng.randrange(z.size)]))
                self.since = 0
                self.upd()
                if self.bestpts >= target:
                    return True
                continue
            if self.since < self.plateau:
                o = np.flatnonzero(free & (conf == 1) & (tabu <= it))
                if o.size:
                    c = int(o[rng.randrange(o.size)])
                    sing, pairs = self.culprits(c)
                    if sing:
                        q = sing[0]
                    elif pairs:
                        q = pairs[0][rng.randrange(2)]
                    else:
                        q = None
                    if q is not None:
                        self.remove(q)
                        tabu[q] = it + 5 + rng.randrange(12)
                        if self.conf[c] == 0:
                            self.add(c)
                        self.since += 1
                        self.upd()
                        if self.bestpts >= target:
                            return True
                        continue
            # perturbation
            self.nper += 1
            self.since = 0
            if self.nper > self.R or self.pts < self.bestpts - 3 * self.maxsz:
                self.reset()
                for p in self.best:
                    self.add(p)
                self.nper = 0
                conf = self.conf
                free = ~self.inP
            lim = 2 if rng.random() < 0.7 else 4
            cand = np.flatnonzero(free & (conf > 0) & (conf <= lim) & (tabu <= it))
            if not cand.size:
                cand = np.flatnonzero(free & (tabu <= it))
                if not cand.size:
                    cand = np.flatnonzero(free)
                    if not cand.size:
                        return False
            c = int(cand[rng.randrange(cand.size)])
            sing, pairs = self.culprits(c)
            rem = set(sing)
            for (a, b) in pairs:
                if a not in rem and b not in rem:
                    rem.add(a if rng.random() < 0.5 else b)
            for q in rem:
                self.remove(q)
                tabu[q] = it + 5 + rng.randrange(12)
            if self.conf[c] == 0 and not self.inP[c]:
                self.add(c)
            self.upd()
            if self.bestpts >= target:
                return True


class _LS(object):
    def __init__(self, n):
        self.n = n
        N2 = n * n
        self.N2 = N2
        self.xs = [i // n for i in range(N2)]
        self.ys = [i % n for i in range(N2)]
        D = 2 * n - 1
        self.D = D
        tA = [0] * (D * D)
        tB = [0] * (D * D)
        tI = [0] * (D * D)
        dirs = {}
        for dx in range(-(n - 1), n):
            for dy in range(-(n - 1), n):
                if dx == 0 and dy == 0:
                    continue
                g = gcd(abs(dx), abs(dy))
                a = dx // g
                b = dy // g
                if a < 0 or (a == 0 and b < 0):
                    a, b = -a, -b
                d = dirs.get((a, b))
                if d is None:
                    d = len(dirs)
                    dirs[(a, b)] = d
                t = (dx + n - 1) * D + (dy + n - 1)
                tA[t] = a
                tB[t] = b
                tI[t] = d
        self.tA, self.tB, self.tI = tA, tB, tI
        self.M = 4 * n * n + 1
        self.off = 2 * n * n
        self.lc = {}
        self.reset()

    def reset(self):
        self.conf = np.zeros(self.N2, np.int64)
        self.inS = np.zeros(self.N2, bool)
        self.S = []

    def _cells(self, p, a, b):
        n = self.n
        x, y = self.xs[p], self.ys[p]
        while 0 <= x - a < n and 0 <= y - b < n:
            x -= a
            y -= b
        out = []
        while 0 <= x < n and 0 <= y < n:
            out.append(x * n + y)
            x += a
            y += b
        return np.array(out, dtype=np.intp)

    def _group(self, p):
        xs, ys = self.xs, self.ys
        px, py = xs[p], ys[p]
        n1 = self.n - 1
        D = self.D
        tA, tB, tI = self.tA, self.tB, self.tI
        M = self.M
        off = self.off
        lc = self.lc
        if len(lc) > 250000:
            lc.clear()
        g = {}
        for s in self.S:
            t = (xs[s] - px + n1) * D + ys[s] - py + n1
            a = tA[t]
            b = tB[t]
            k = tI[t] * M + b * px - a * py + off
            v = g.get(k)
            if v is None:
                g[k] = 1
                if k not in lc:
                    lc[k] = self._cells(p, a, b)
            else:
                g[k] = v + 1
        return g

    def add(self, p):
        g = self._group(p)
        conf = self.conf
        lc = self.lc
        for k, v in g.items():
            conf[lc[k]] += v
        self.S.append(p)
        self.inS[p] = True

    def remove(self, p):
        self.S.remove(p)
        self.inS[p] = False
        g = self._group(p)
        conf = self.conf
        lc = self.lc
        for k, v in g.items():
            conf[lc[k]] -= v

    def conflicts(self, c):
        xs, ys = self.xs, self.ys
        px, py = xs[c], ys[c]
        n1 = self.n - 1
        D = self.D
        tA, tB, tI = self.tA, self.tB, self.tI
        M = self.M
        g = {}
        for s in self.S:
            t = (xs[s] - px + n1) * D + ys[s] - py + n1
            k = tI[t] * M + tB[t] * px - tA[t] * py
            lst = g.get(k)
            if lst is None:
                g[k] = [s]
            else:
                lst.append(s)
        return [v for v in g.values() if len(v) >= 2]


def _local_search(n, init_pts, deadline, rng):
    ls = _LS(n)
    for (x, y) in init_pts:
        p = x * n + y
        if 0 <= x < n and 0 <= y < n and not ls.inS[p] and ls.conf[p] == 0:
            ls.add(p)
    best = list(ls.S)
    target = 2 * n
    N2 = n * n
    tabu = np.zeros(N2, np.int64)
    it = 0
    since = 0
    nper = 0
    plateau = 3 * n
    R = 40
    tm = time.time
    while len(ls.S) < target:
        it += 1
        if (it & 3) == 0 and tm() > deadline:
            break
        conf = ls.conf
        free = ~ls.inS
        z = np.flatnonzero(free & (conf == 0))
        if z.size:
            ls.add(int(z[rng.randrange(z.size)]))
            since = 0
            if len(ls.S) > len(best):
                best = list(ls.S)
                nper = 0
            continue
        if since < plateau:
            o = np.flatnonzero(free & (conf == 1) & (tabu <= it))
            if o.size:
                c = int(o[rng.randrange(o.size)])
                grp = ls.conflicts(c)
                if grp and len(grp[0]) == 2:
                    q = grp[0][rng.randrange(2)]
                    ls.remove(q)
                    tabu[q] = it + 5 + rng.randrange(12)
                    if ls.conf[c] == 0:
                        ls.add(c)
                    since += 1
                    continue
        nper += 1
        since = 0
        if nper > R or len(ls.S) < len(best) - 3:
            ls.reset()
            for p in best:
                ls.add(p)
            nper = 0
            conf = ls.conf
            free = ~ls.inS
        lim = 2 if rng.random() < 0.7 else 4
        cand = np.flatnonzero(free & (conf > 0) & (conf <= lim) & (tabu <= it))
        if not cand.size:
            cand = np.flatnonzero(free & (tabu <= it))
            if not cand.size:
                cand = np.flatnonzero(free)
                if not cand.size:
                    break
        c = int(cand[rng.randrange(cand.size)])
        groups = ls.conflicts(c)
        for g in groups:
            rng.shuffle(g)
            for s in g[1:]:
                ls.remove(s)
                tabu[s] = it + 5 + rng.randrange(12)
        if ls.conf[c] == 0 and not ls.inS[c]:
            ls.add(c)
        if len(ls.S) > len(best):
            best = list(ls.S)
            nper = 0
    if len(ls.S) > len(best):
        best = list(ls.S)
    return [(p // n, p % n) for p in best]


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit - 0.5
    rng = random.Random(seed)
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if n == 3:
        return [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
    if n % 2 == 0:
        names = ['rot4', 'dia2', 'rot4', 'rot2']
    else:
        names = ['rot2', 'dia2', 'rot2', 'dia1']
    if n <= 8:
        names.append('iden')
    sym_deadline = t0 + 0.72 * time_limit
    slice_len = max(0.5, 0.72 * time_limit / 6.0)
    searchers = {}
    best_pts = []
    gi = 0
    while time.time() < sym_deadline - 0.05:
        nm = names[gi % len(names)]
        gi += 1
        try:
            if nm not in searchers:
                model = _Model(n, nm)
                if model.N == 0:
                    searchers[nm] = None
                else:
                    searchers[nm] = _OLS(model, rng)
            s = searchers[nm]
            if s is None:
                continue
            dl = min(sym_deadline, time.time() + slice_len)
            done = s.run(dl)
            if s.bestpts > len(best_pts):
                best_pts = s.points()
            if done or len(best_pts) >= 2 * n:
                return best_pts
        except Exception:
            continue
    init = list(best_pts)
    try:
        pts = _local_search(n, init, deadline, rng)
    except Exception:
        pts = init
    if len(pts) < len(init):
        pts = init
    return [(int(x), int(y)) for (x, y) in pts]
