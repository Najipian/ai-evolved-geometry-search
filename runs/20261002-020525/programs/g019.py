# g019 (parent g005): Spend ~85% of the time on a bitset DFS over row/column-class demands for rot4 and dia2 orbits, with MRV, forward checking, LCV, sibling exclusion and Luby restarts.
import time
import random
from math import gcd
import numpy as np

try:
    _pc = int.bit_count
    _pc(5)
except Exception:
    def _pc(v):
        return bin(v).count('1')


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
        self.name = name
        self.lcache = {}
        maps = _group_maps(n, name)
        self.maps = maps
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
        k = 1 << 30
        if dx > 0:
            k = px // dx
        if dy > 0:
            k = min(k, py // dy)
        elif dy < 0:
            k = min(k, (n - 1 - py) // (-dy))
        x = px - k * dx
        y = py - k * dy
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


class _Budget(Exception):
    pass


class _DFS(object):
    """Exact restart DFS over orbits with line-class demands (bitsets)."""

    def __init__(self, model, n):
        self.m = model
        self.n = n
        N = model.N
        self.N = N
        self.ALL = (1 << N) - 1
        maps = model.maps
        par = list(range(2 * n))

        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a

        def img(L, f):
            if L < n:
                a0, b0 = f(L, 0)
                a1, b1 = f(L, 1)
            else:
                a0, b0 = f(0, L - n)
                a1, b1 = f(1, L - n)
            if a0 == a1:
                return a0
            return n + b0

        for f in maps:
            for L in range(2 * n):
                a = find(L)
                b = find(img(L, f))
                if a != b:
                    par[a] = b
        roots = sorted(set(find(L) for L in range(2 * n)))
        C = len(roots)
        self.C = C
        cm = [0] * C
        cm2 = [0] * C
        contrib = [[] for _ in range(N)]
        for o, cells in enumerate(model.orbits):
            for ci, L in enumerate(roots):
                if L < n:
                    k = sum(1 for (x, y) in cells if x == L)
                else:
                    k = sum(1 for (x, y) in cells if y == L - n)
                if k > 0:
                    contrib[o].append((ci, k))
                    cm[ci] |= 1 << o
                    if k >= 2:
                        cm2[ci] |= 1 << o
        self.cm, self.cm2, self.contrib = cm, cm2, contrib
        self.has2 = [v != 0 for v in cm2]
        self.base = [model.internal[o] | model.hits[o] | (1 << o) for o in range(N)]
        self.size = model.size
        self.best = []
        self.bestpts = 0
        self.bdl = 0
        self.basebudget = max(40, 8 * C)
        self.k = 0

    def points(self, S):
        out = []
        for o in S:
            out.extend(self.m.orbits[o])
        return [(int(x), int(y)) for (x, y) in out]

    def run(self, budget, deadline, rng):
        N = self.N
        ALL = self.ALL
        cm, cm2, has2 = self.cm, self.cm2, self.has2
        C = self.C
        base = self.base
        contrib = self.contrib
        cache = self.m.cache
        compat = self.m.compat
        size = self.size
        pc = _pc
        rnd = rng.random
        tm = time.time
        noise = rng.choice((0.0, 0.7, 2.0, 4.0, 8.0))
        mnoise = rng.choice((0.99, 0.99, 1.5))
        cnt = [0]
        me = self

        def rec(F, S, r):
            cnt[0] += 1
            c0 = cnt[0]
            if c0 > budget:
                raise _Budget()
            if (c0 & 127) == 0 and tm() > deadline:
                raise _Budget()
            ls = len(S)
            if ls >= me.bdl:
                p = 0
                for o in S:
                    p += size[o]
                if p > me.bestpts:
                    me.bestpts = p
                    me.best = list(S)
                    me.bdl = ls
            avail = ALL & ~F
            bk = -1
            bs = 1e18
            for c in range(C):
                rc = r[c]
                if rc == 0:
                    continue
                s = pc(avail & cm[c])
                if has2[c]:
                    s += pc(avail & cm2[c])
                if s < rc:
                    return None
                key = (s - rc) + rnd() * mnoise
                if key < bs:
                    bs = key
                    bk = c
            if bk < 0:
                return list(S)
            v = avail & cm[bk]
            children = []
            while v:
                low = v & -v
                o = low.bit_length() - 1
                v ^= low
                nf = F | base[o]
                ob = o * N
                for b in S:
                    w = cache[ob + b]
                    if w is None:
                        w = compat(o, b)
                    nf |= w
                nr = list(r)
                ok = True
                for (c, a) in contrib[o]:
                    t = nr[c] - a
                    if t < 0:
                        ok = False
                        break
                    nr[c] = t
                    if t == 0:
                        nf |= cm[c]
                    elif t == 1 and has2[c]:
                        nf |= cm2[c]
                if not ok:
                    continue
                sc = pc(ALL & ~nf) + noise * rnd()
                children.append((sc, o, nf, nr))
            children.sort(key=lambda t: -t[0])
            excl = 0
            for (_, o, nf, nr) in children:
                S.append(o)
                res = rec(nf | excl, S, nr)
                if res is not None:
                    return res
                S.pop()
                excl |= 1 << o
            return None

        try:
            return rec(0, [], [2] * C)
        except _Budget:
            return None


def _luby(i):
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    if i == (1 << k) - 1:
        return 1 << (k - 1)
    return _luby(i - (1 << (k - 1)) + 1)


def _ok(pts, n):
    if len(set(pts)) != len(pts):
        return False
    for (x, y) in pts:
        if not (0 <= x < n and 0 <= y < n):
            return False
    L = len(pts)
    for i in range(L):
        px, py = pts[i]
        seen = set()
        for j in range(L):
            if j == i:
                continue
            dx = pts[j][0] - px
            dy = pts[j][1] - py
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            if (dx, dy) in seen:
                return False
            seen.add((dx, dy))
    return True


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit - 0.6
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
        groups = ['rot4', 'dia2']
        w0 = 0.6
    else:
        groups = ['rot2', 'dia1']
        w0 = 0.6
    models = {}

    def get_model(nm):
        if nm not in models:
            models[nm] = _Model(n, nm)
        return models[nm]

    best_pts = []
    # Phase 1: short orbit ILS for a solid fallback.
    try:
        mdl = get_model(groups[0])
        if mdl.N > 0:
            ils = _OLS(mdl, rng)
            dl = min(deadline, t0 + min(3.0, 0.15 * time_limit))
            done = ils.run(dl)
            best_pts = ils.points()
            if done and _ok(best_pts, n):
                return best_pts
    except Exception:
        pass
    # Phase 2: restarting exact DFS.
    dfss = {}
    broken = set()
    while time.time() < deadline - 0.05:
        nm = groups[0] if rng.random() < w0 else groups[1]
        if nm in broken:
            nm = groups[1] if nm == groups[0] else groups[0]
            if nm in broken:
                break
        try:
            if nm not in dfss:
                mdl = get_model(nm)
                if mdl.N == 0:
                    broken.add(nm)
                    continue
                dfss[nm] = _DFS(mdl, n)
            d = dfss[nm]
            d.k += 1
            budget = _luby(d.k) * d.basebudget
            sol = d.run(budget, deadline, rng)
            if sol is not None:
                pts = d.points(sol)
                if len(pts) >= 2 * n and _ok(pts, n):
                    return pts
        except Exception:
            broken.add(nm)
            continue
    for d in dfss.values():
        if d.bestpts > len(best_pts):
            cand = d.points(d.best)
            if _ok(cand, n):
                best_pts = cand
    return [(int(x), int(y)) for (x, y) in best_pts]
