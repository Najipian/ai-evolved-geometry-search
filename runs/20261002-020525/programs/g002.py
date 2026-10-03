# g002 (parent g001): Replace rot4 annealing with vectorized numpy tabu min-conflicts: first under 180° symmetry, then without symmetry starting from the best symmetric state.
# Vectorized min-conflicts tabu search (numpy): every row always holds exactly two points.
# Phase 1 searches half-turn (rot2) symmetric configurations. Phase 2 continues without symmetry
# from the best symmetric state, so each remaining defect costs only one point.
# Exact triple deltas for all n columns of a row are computed at once with a direction table.
# The backtracker is kept for small n and as an extra option for odd n.
import time
import random
import math
from math import gcd
import numpy as np

try:
    _pc = int.bit_count
except AttributeError:
    def _pc(x):
        return bin(x).count('1')


class _Abort(Exception):
    pass


def _bits(m):
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


def _luby(i):
    while True:
        k = 1
        while (1 << k) - 1 < i:
            k += 1
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i -= (1 << (k - 1)) - 1


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
        self.ALL = (1 << N) - 1
        self.size = [len(c) for c in orbits]
        lm = [0] * (2 * n)
        lm2 = [0] * (2 * n)
        contrib = []
        for oid, cells in enumerate(orbits):
            cnt = {}
            for (a, b) in cells:
                cnt[a] = cnt.get(a, 0) + 1
                cnt[n + b] = cnt.get(n + b, 0) + 1
            bit = 1 << oid
            lst = []
            for idx, k in cnt.items():
                lm[idx] |= bit
                if k >= 2:
                    lm2[idx] |= bit
                lst.append((idx, k))
            contrib.append(lst)
        self.lm, self.lm2, self.contrib = lm, lm2, contrib
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
        self.base = [(1 << a) | internal[a] | hits[a] for a in range(N)]
        self.cache = [None] * (N * N)
        parent = list(range(2 * n))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        for f in maps:
            for r in range(n):
                (x1, y1), (x2, y2) = f(r, 0), f(r, 1)
                img = x1 if x1 == x2 else n + y1
                parent[find(r)] = find(img)
                (x1, y1), (x2, y2) = f(0, r), f(1, r)
                img = x1 if x1 == x2 else n + y1
                parent[find(n + r)] = find(img)
        self.reps = sorted(set(find(i) for i in range(2 * n)))

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

    def compat_slow(self, a, b):
        p = self.orbits[a][0]
        v = 0
        for q in self.orbits[b]:
            v |= self.line(p, q)
        N = self.N
        self.cache[a * N + b] = v
        self.cache[b * N + a] = v
        return v


def _run(model, rng, budget, deadline, best, noise):
    n = model.n
    ALL = model.ALL
    lm, lm2 = model.lm, model.lm2
    contrib = model.contrib
    base = model.base
    size = model.size
    cache = model.cache
    N = model.N
    cslow = model.compat_slow
    target = 2 * n
    reps = list(model.reps)
    rng.shuffle(reps)
    cnt = [0]
    pc = _pc
    tm = time.time
    rnd = rng.random
    keyf = lambda t: t[0]

    def rec(dead, counts, placed, npts):
        if npts == target:
            return list(placed)
        c0 = cnt[0] + 1
        cnt[0] = c0
        if c0 > budget or ((c0 & 127) == 0 and tm() > deadline):
            raise _Abort()
        if npts > best[0]:
            best[0] = npts
            best[1] = [model.orbits[o] for o in placed]
        avail = ALL & ~dead
        bestc = 1 << 30
        bm = 0
        bi = -1
        for i in reps:
            ci = counts[i]
            if ci < 2:
                m = avail & lm[i]
                if not m:
                    return None
                c = pc(m)
                if ci == 0 and c == 1 and not (m & lm2[i]):
                    return None
                if c < bestc:
                    bestc = c
                    bm = m
                    bi = i
        if bm == 0:
            return None
        lmi = lm[bi]
        lm2i = lm2[bi]
        need2 = counts[bi] == 0
        children = []
        for a in _bits(bm):
            nd = dead | base[a]
            row = a * N
            for b in placed:
                v = cache[row + b]
                if v is None:
                    v = cslow(a, b)
                nd |= v
            nc = counts[:]
            for idx, k in contrib[a]:
                v = nc[idx] + k
                nc[idx] = v
                if v >= 2:
                    nd |= lm[idx]
                else:
                    nd |= lm2[idx]
            sc = pc(ALL & ~nd) + rnd() * noise
            children.append((sc, a, nd, nc))
        children.sort(key=keyf, reverse=True)
        failed = 0
        for sc, a, nd, nc in children:
            ba = 1 << a
            placed.append(a)
            r = rec(nd | failed, nc, placed, npts + size[a])
            if r:
                return r
            placed.pop()
            failed |= ba
            avail &= ~ba
            rem = avail & lmi
            if rem == 0:
                break
            if need2 and not (rem & lm2i) and pc(rem) < 2:
                break
        return None

    try:
        res = rec(0, [0] * (2 * n), [], 0)
    except _Abort:
        return None
    if res:
        return [model.orbits[o] for o in res]
    return None


def _bt(n, deadline, rng):
    if n % 2 == 0:
        names = ['rot4', 'dia2', 'rot4', 'dia2', 'rot2']
    else:
        names = ['rot2', 'dia2', 'rot2', 'dia1']
    if n <= 8:
        names.append('iden')
    models = {}
    counters = dict((nm, 0) for nm in names)
    best = [0, []]
    unit = max(40, 4 * n)
    gi = 0
    while time.time() < deadline:
        nm = names[gi % len(names)]
        gi += 1
        if nm not in models:
            models[nm] = _Model(n, nm)
        model = models[nm]
        if model.N == 0:
            continue
        counters[nm] += 1
        k = counters[nm]
        budget = unit * _luby(k)
        noise = 3.0 if (k % 4) else 12.0
        res = _run(model, rng, budget, deadline, best, noise)
        if res:
            pts = []
            for cells in res:
                pts.extend(cells)
            return [(int(x), int(y)) for (x, y) in pts], None
    pts = []
    for cells in best[1]:
        pts.extend(cells)
    return None, [(int(x), int(y)) for (x, y) in pts]


def _rowpairs(A):
    s = np.sort(A, axis=1)
    eq = s[:, 1:] == s[:, :-1]
    c = np.cumsum(eq, axis=1)
    br = np.where(eq, 0, c)
    base = np.maximum.accumulate(br, axis=1)
    return (c - base).sum(axis=1)


def _mc(n, deadline, rng, mode, Yinit=None):
    m = n - 1
    W = 2 * n - 1
    r_ = np.arange(-m, m + 1)
    dx = np.repeat(r_, W)
    dy = np.tile(r_, W)
    g = np.gcd(np.abs(dx), np.abs(dy))
    g[g == 0] = 1
    a = dx // g
    b = dy // g
    neg = (a < 0) | ((a == 0) & (b < 0))
    a = np.where(neg, -a, a)
    b = np.where(neg, -b, b)
    D = ((a + m) * W + (b + m)).astype(np.int64)
    D[m * W + m] = -1
    rot2 = (mode == 'rot2')
    N2 = 2 * n
    if rot2:
        h = n // 2
        bs = np.repeat(np.arange(h), 2)
        X = np.concatenate([bs, m - bs]).astype(np.int64)
        nv = n
    else:
        X = np.repeat(np.arange(n), 2).astype(np.int64)
        nv = N2
    allidx = np.arange(N2)
    O = []
    XO = []
    for p in range(nv):
        if rot2:
            o = allidx[(allidx != p) & (allidx != p + n)]
        else:
            o = allidx[allidx != p]
        O.append(o)
        XO.append(((X[o] - X[p] + m) * W + m)[None, :])
    CX = (X[None, :] - X[:nv, None] + m) * W + m
    cs = np.arange(n)
    csc = cs[:, None]
    dpp = None
    if rot2:
        dpp = [D[(2 * m - 2 * r) * W + 2 * m - 2 * cs][:, None] for r in range(n // 2)]
    if Yinit is not None:
        Y = np.array(Yinit, dtype=np.int64)
    else:
        Y = np.zeros(N2, dtype=np.int64)
        if rot2:
            for r in range(n // 2):
                c1, c2 = rng.sample(range(n), 2)
                Y[2 * r] = c1
                Y[2 * r + 1] = c2
        else:
            cols = list(range(n)) * 2
            while True:
                rng.shuffle(cols)
                ok = all(cols[2 * r] != cols[2 * r + 1] for r in range(n))
                if ok:
                    break
            Y[:] = cols
    if rot2:
        Y[n:] = m - Y[:n]
    nrs = np.random.RandomState(rng.randrange(1 << 30))
    tabu = np.zeros((nv, n), dtype=np.int64)
    best = 1 << 30
    bestY = Y.copy()
    it = 0
    lastimp = 0
    stag = 30 * n
    rnd = rng.random
    rint = rng.randrange
    tm = time.time
    INF = float('inf')
    tmax = max(2, n // 4)
    while True:
        it += 1
        if tm() > deadline:
            break
        idx = CX + (Y[None, :] - Y[:nv, None])
        t = _rowpairs(D[idx])
        s = int(t.sum())
        total = (2 * s) // 3 if rot2 else s // 3
        if total < best:
            best = total
            bestY = Y.copy()
            lastimp = it
            if total == 0:
                break
        if it - lastimp > stag:
            Y = bestY.copy()
            for _ in range(2 + rint(3)):
                p = rint(nv)
                q = p ^ 1
                c = rint(n)
                if c == Y[q]:
                    continue
                Y[p] = c
                if rot2:
                    Y[p + n] = m - c
            lastimp = it
            continue
        conf = np.flatnonzero(t)
        if len(conf) == 0:
            break
        p = int(conf[rint(len(conf))])
        q = p ^ 1
        r = int(X[p])
        cur = int(Y[p])
        yq = int(Y[q])
        if rnd() < 0.03:
            c = rint(n)
            if c == cur or c == yq:
                continue
        else:
            Oi = O[p]
            dirs = D[XO[p] + (Y[Oi][None, :] - csc)]
            pr = _rowpairs(dirs)
            if rot2:
                cost = 2 * pr + (dirs == dpp[r]).sum(axis=1)
            else:
                cost = pr
            ccur = cost[cur]
            val = cost.astype(np.float64) + nrs.random_sample(n) * 0.5
            val[yq] = INF
            val[cur] = INF
            tb = tabu[p] > it
            if tb.any():
                asp = (total + cost - ccur) < best
                val[tb & ~asp] = INF
            c = int(np.argmin(val))
            if val[c] == INF:
                c = rint(n)
                if c == cur or c == yq:
                    continue
        Y[p] = c
        if rot2:
            Y[p + n] = m - c
        tabu[p, cur] = it + 3 + rint(tmax)
    pts = list(zip(X.tolist(), bestY.tolist()))
    return [(int(x), int(y)) for (x, y) in pts], best


def _to_full(n, pts):
    rows = {}
    for (x, y) in pts:
        rows.setdefault(x, []).append(y)
    Y = []
    for r in range(n):
        lst = rows.get(r, [])
        lst = list(lst[:2])
        while len(lst) < 2:
            c = random.randrange(n)
            if c not in lst:
                lst.append(c)
        Y.extend(lst)
    return Y


def _repair(n, pts, rng, deadline):
    pts = list(set((int(x), int(y)) for (x, y) in pts))
    while True:
        lines = {}
        L = len(pts)
        for i in range(L):
            x1, y1 = pts[i]
            for j in range(i + 1, L):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                key = (dx, dy, dy * x1 - dx * y1)
                s = lines.get(key)
                if s is None:
                    lines[key] = set((i, j))
                else:
                    s.add(i)
                    s.add(j)
        bad = [s for s in lines.values() if len(s) >= 3]
        if not bad:
            break
        score = [0] * L
        for s in bad:
            k = len(s)
            w = (k - 1) * (k - 2) // 2
            for i in s:
                score[i] += w
        worst = max(range(L), key=lambda i: (score[i], rng.random()))
        pts.pop(worst)
    occ = set(pts)
    cells = [(x, y) for x in range(n) for y in range(n) if (x, y) not in occ]
    rng.shuffle(cells)
    for (x, y) in cells:
        if len(pts) >= 2 * n or time.time() > deadline:
            break
        seen = set()
        okc = True
        for (qx, qy) in pts:
            dx = qx - x
            dy = qy - y
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            kk = (dx, dy)
            if kk in seen:
                okc = False
                break
            seen.add(kk)
        if okc:
            pts.append((x, y))
    return pts


def construct(n, time_limit, seed):
    t0 = time.time()
    rng = random.Random(seed)
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if n == 3:
        return [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
    final_deadline = t0 + time_limit - 0.3
    deadline = t0 + time_limit - 1.0
    partial = []
    if n <= 12:
        sol, partial = _bt(n, t0 + 0.5 * time_limit, rng)
        if sol:
            return sol
    elif n % 2 == 1:
        sol, partial = _bt(n, t0 + 0.25 * time_limit, rng)
        if sol:
            return sol
    init = None
    if n % 2 == 0 and n >= 8:
        now = time.time()
        d1 = now + 0.45 * max(0.0, deadline - now)
        pts1, c1 = _mc(n, d1, rng, 'rot2')
        if c1 == 0:
            return pts1
        init = _to_full(n, pts1)
    pts2, c2 = _mc(n, deadline, rng, 'full', init)
    if c2 == 0:
        return pts2
    res = _repair(n, pts2, rng, final_deadline)
    if partial and len(partial) > len(res):
        res2 = _repair(n, partial, rng, final_deadline)
        if len(res2) > len(res):
            res = res2
    return res
