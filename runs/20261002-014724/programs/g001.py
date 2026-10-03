# g001 (parent s0): Keep the symmetric orbit backtracker for 45% of the time, then run a numpy tabu min-conflicts search seeded from its best partial to add valid points.
# Orbit backtracker (quarter-turn / two-diagonal / half-turn) with forward checking and Luby restarts,
# followed by a numpy line-id-table min-conflicts / tabu local search that grows the best valid set.
import time
import random
from math import gcd

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


# ---------------- local search phase ----------------

def _build_lid(n):
    import numpy as np
    N2 = n * n
    idx = np.arange(N2, dtype=np.int32)
    X = idx // n
    Y = idx % n
    DX = X[None, :] - X[:, None]
    DY = Y[None, :] - Y[:, None]
    G = np.gcd(DX, DY)
    G[G == 0] = 1
    A = DX // G
    B = DY // G
    del DX, DY, G
    neg = (A < 0) | ((A == 0) & (B < 0))
    A = np.where(neg, -A, A)
    B = np.where(neg, -B, B)
    del neg
    C = B * X[:, None] - A * Y[:, None]
    key = (A.astype(np.int64) * (2 * n + 1) + (B + n)) * (4 * n * n + 1) + (C + 2 * n * n)
    del A, B, C
    _, inv = np.unique(key.ravel(), return_inverse=True)
    del key
    NL = int(inv.max()) + 1
    LID = inv.reshape(N2, N2).astype(np.int32)
    return LID, NL


def _local_search(n, init, deadline, rng):
    import numpy as np
    LID, NL = _build_lid(n)
    N2 = n * n
    target = 2 * n
    cnt = np.zeros(NL, dtype=np.int32)
    S = []
    inS = np.zeros(N2, dtype=bool)
    rowc = np.zeros(n, dtype=np.int32)
    colc = np.zeros(n, dtype=np.int32)
    Xc = np.arange(N2) // n
    Yc = np.arange(N2) % n
    tabu = np.zeros(N2, dtype=np.int64)
    nrng = np.random.default_rng(rng.randrange(1 << 30))

    def add(c):
        if S:
            ids = np.unique(LID[c, S])
            v = cnt[ids]
            cnt[ids] = np.where(v == 0, 2, v + 1)
        S.append(c)
        inS[c] = True
        rowc[c // n] += 1
        colc[c % n] += 1

    def rem(i):
        c = S[i]
        S[i] = S[-1]
        S.pop()
        inS[c] = False
        rowc[c // n] -= 1
        colc[c % n] -= 1
        if S:
            ids = np.unique(LID[c, S])
            v = cnt[ids]
            cnt[ids] = np.where(v == 2, 0, v - 1)
        return c

    def evalc(cand, Sa):
        M = cnt[LID[np.ix_(cand, Sa)]]
        return np.maximum(M - 1, 0).sum(axis=1)

    seen = set()
    for c in init:
        if c not in seen:
            seen.add(c)
            add(c)
    best = []
    it = 0
    K = 3 * n + 20
    while time.time() < deadline:
        it += 1
        if S:
            Sa = np.array(S, dtype=np.int64)
            M = cnt[LID[np.ix_(Sa, Sa)]]
            T = (M - 2).sum(axis=1) + 2
            tmax = int(T.max())
        else:
            Sa = np.zeros(0, dtype=np.int64)
            T = None
            tmax = 0
        if tmax == 0:
            if len(S) > len(best):
                best = list(S)
            if len(S) >= target:
                break
            cand = np.nonzero(~inS)[0]
            if len(cand) == 0:
                break
            if len(S) == 0:
                add(int(cand[rng.randrange(len(cand))]))
                continue
            cr = evalc(cand, Sa)
            mn = cr.min()
            ch = np.nonzero(cr == mn)[0]
            add(int(cand[ch[rng.randrange(len(ch))]]))
            continue
        conf = np.nonzero(T > 0)[0]
        if rng.random() < 0.3:
            i = int(np.argmax(T))
        else:
            i = int(conf[rng.randrange(len(conf))])
        old = rem(i)
        tabu[old] = it + 8 + rng.randrange(n // 2 + 1)
        Sa = np.array(S, dtype=np.int64)
        free = (~inS) & (tabu < it)
        capok = (rowc[Xc] < 2) & (colc[Yc] < 2)
        okc = np.nonzero(free & capok)[0]
        others = np.nonzero(free & ~capok)[0]
        if len(others) > K:
            others = others[nrng.integers(0, len(others), K)]
        cand = np.concatenate((okc, others))
        if len(cand) == 0:
            add(old)
            continue
        if rng.random() < 0.03:
            add(int(cand[rng.randrange(len(cand))]))
            continue
        cr = evalc(cand, Sa)
        mn = cr.min()
        ch = np.nonzero(cr == mn)[0]
        add(int(cand[ch[rng.randrange(len(ch))]]))
    return best


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit - 0.4
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
        names = ['rot4', 'dia2', 'rot4', 'dia2', 'rot2']
    else:
        names = ['rot2', 'dia2', 'rot2', 'dia1']
    if n <= 8:
        names.append('iden')
    bt_deadline = t0 + time_limit * 0.45
    if n <= 12:
        bt_deadline = deadline
    models = {}
    counters = dict((nm, 0) for nm in names)
    best = [0, []]
    unit = max(40, 4 * n)
    gi = 0
    while time.time() < bt_deadline:
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
        res = _run(model, rng, budget, bt_deadline, best, noise)
        if res:
            pts = []
            for cells in res:
                pts.extend(cells)
            return [(int(x), int(y)) for (x, y) in pts]
    init_pts = []
    for cells in best[1]:
        init_pts.extend(cells)
    init_pts = [(int(x), int(y)) for (x, y) in init_pts]
    if time.time() < deadline - 0.5:
        try:
            res = _local_search(n, [x * n + y for (x, y) in init_pts], deadline, rng)
        except Exception:
            res = None
        if res and len(res) > len(init_pts):
            return [(int(c // n), int(c % n)) for c in res]
    return init_pts
