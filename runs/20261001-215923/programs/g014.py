# g014 (parent g006): Keep the proven orbit backtracker and its search order unchanged, and speed it up with a flat symmetric compatibility cache and an inlined lookup.
# Bitmask backtracking over symmetry orbits (quarter-turn / two-diagonal / half-turn),
# flat orbit-pair line caches, Luby restarts, need-aware forward checking.
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
        orb_of = self.orb_of
        px, py = p
        dx, dy = q[0] - px, q[1] - py
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        x, y = px, py
        while 0 <= x - dx < n and 0 <= y - dy < n:
            x -= dx
            y -= dy
        msk = 0
        while 0 <= x < n and 0 <= y < n:
            o = orb_of[x * n + y]
            if o >= 0:
                msk |= 1 << o
            x += dx
            y += dy
        return msk

    def compat_fill(self, a, b):
        p = self.orbits[a][0]
        v = 0
        for q in self.orbits[b]:
            v |= self.line(p, q)
        N = self.N
        self.cache[a * N + b] = v
        self.cache[b * N + a] = v
        return v


def _run(model, rng, budget, deadline, best):
    n = model.n
    ALL = model.ALL
    lm, lm2 = model.lm, model.lm2
    contrib = model.contrib
    internal, hits = model.internal, model.hits
    size = model.size
    cache = model.cache
    cfill = model.compat_fill
    NN = model.N
    target = 2 * n
    reps = list(model.reps)
    rng.shuffle(reps)
    cnt = [0]
    pc = _pc
    tm = time.time

    def rec(dead, counts, placed, npts):
        if npts == target:
            return list(placed)
        cnt[0] += 1
        c0 = cnt[0]
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
                if m == 0:
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
        cands = _bits(bm)
        rng.shuffle(cands)
        lmi = lm[bi]
        lm2i = lm2[bi]
        need2 = counts[bi] == 0
        for a in cands:
            ba = 1 << a
            if not (avail & ba):
                continue
            nd = dead | ba | internal[a] | hits[a]
            base = a * NN
            for b in placed:
                v = cache[base + b]
                if v is None:
                    v = cfill(a, b)
                nd |= v
            nc = counts[:]
            for idx, k in contrib[a]:
                v = nc[idx] + k
                nc[idx] = v
                if v >= 2:
                    nd |= lm[idx]
                else:
                    nd |= lm2[idx]
            placed.append(a)
            r = rec(nd, nc, placed, npts + size[a])
            if r:
                return r
            placed.pop()
            avail &= ~ba
            dead |= ba
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


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit - 0.35
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
        names = ['rot4', 'dia2', 'rot2']
    else:
        names = ['rot2', 'dia2', 'dia1']
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
        budget = unit * _luby(counters[nm])
        res = _run(model, rng, budget, deadline, best)
        if res:
            pts = []
            for cells in res:
                pts.extend(cells)
            return [(int(x), int(y)) for (x, y) in pts]
    pts = []
    for cells in best[1]:
        pts.extend(cells)
    return [(int(x), int(y)) for (x, y) in pts]
