# g001 (parent s0): Use simulated annealing over quarter-turn-symmetric configurations, modelled as 2-regular multigraphs, with fast per-orbit triple counting; then repair greedily and re-add free cells.
# Quarter-turn symmetric simulated annealing for even n: the configuration is a 2-regular multigraph
# on n/2 row/column classes (each edge = one rot4 orbit), so every row and column always holds exactly
# two points. Moves are edge swaps (2-opt) and orbit flips. Exact triple deltas come from one
# direction scan per orbit, using rot4 symmetry. The parent backtracker is kept for odd and small n.
# The final result is greedily repaired and then greedily extended.
import time
import random
import math
from math import gcd
from collections import Counter

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


def _sa_rot4(n, deadline, rng):
    m = n - 1
    h = n // 2
    W = 2 * n - 1
    off = m * W + m
    D = [-1] * (W * W)
    ids = {}
    for dx in range(-m, m + 1):
        for dy in range(-m, m + 1):
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            a = dx // g
            b = dy // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            k = (a, b)
            v = ids.get(k)
            if v is None:
                v = len(ids)
                ids[k] = v
            D[(dx + m) * W + dy + m] = v

    ok = {}
    oc = {}
    for u in range(h):
        for v in range(u, h):
            for t in (0, 1):
                if u == v and t == 1:
                    continue
                key = (u * h + v) * 2 + t
                x, y = (u, v) if t == 0 else (u, m - v)
                cs = [(x, y), (y, m - x), (m - x, m - y), (m - y, x)]
                oc[key] = cs
                ok[key] = [a * W + b for (a, b) in cs]

    def canon(u, v, t):
        if u > v:
            u, v = v, u
            t ^= 1
        if u == v:
            t = 0
        return (u * h + v) * 2 + t

    def TO(o, rest):
        b = off - o[0]
        ds = [D[k + b] for k in rest]
        d1 = D[o[1] + b]
        d2 = D[o[2] + b]
        d3 = D[o[3] + b]
        ds.append(d1)
        ds.append(d2)
        ds.append(d3)
        if len(set(ds)) == len(ds):
            return 0
        c = Counter(ds)
        t = 0
        for v in c.values():
            if v > 1:
                t += (v * (v - 1)) >> 1
        return 4 * t - 2 * (c[d1] + c[d2] + c[d3] - 3)

    def rand_state():
        while True:
            perm = list(range(h))
            rng.shuffle(perm)
            E = []
            for i in range(h):
                a = perm[i]
                b = perm[(i + 1) % h]
                E.append((a, b, canon(a, b, rng.randrange(2))))
            if len(set(e[2] for e in E)) == h:
                return E

    def full(E):
        K = []
        cost = 0
        for e in E:
            o = ok[e[2]]
            cost += TO(o, K)
            K = K + o
        return K, cost

    def pts_of(E):
        out = []
        for e in E:
            out.extend(oc[e[2]])
        return out

    E = rand_state()
    K, cost = full(E)
    keyset = set(e[2] for e in E)
    best = cost
    bestE = list(E)
    if cost == 0:
        return pts_of(E), 0
    rnd = rng.random
    rint = rng.randrange
    exp = math.exp
    Thi, Tlo = 4.0, 0.6
    L = 25000
    T = Thi
    it = 0
    stall = 0
    lastbest = best
    tm = time.time
    while True:
        it += 1
        if (it & 127) == 0:
            if tm() > deadline:
                break
            ph = (it % L) / float(L)
            T = Thi * (Tlo / Thi) ** ph
        if it % L == 0:
            if best < lastbest:
                lastbest = best
                stall = 0
            else:
                stall += 1
                if stall >= 3:
                    stall = 0
                    if rnd() < 0.5:
                        E = list(bestE)
                    else:
                        E = rand_state()
                    K, cost = full(E)
                    keyset = set(e[2] for e in E)
        if rnd() < 0.25:
            i = rint(h)
            a, b, k = E[i]
            if a == b:
                continue
            f = k ^ 1
            if f in keyset:
                continue
            rest0 = K[:4 * i] + K[4 * i + 4:]
            nf = ok[f]
            delta = TO(nf, rest0) - TO(ok[k], rest0)
            if delta <= 0 or rnd() < exp(-delta / T):
                E[i] = (a, b, f)
                K[4 * i:4 * i + 4] = nf
                keyset.discard(k)
                keyset.add(f)
                cost += delta
            else:
                continue
        else:
            i = rint(h)
            j = rint(h - 1)
            if j >= i:
                j += 1
            a, b, k1 = E[i]
            c, d, k2 = E[j]
            if rnd() < 0.5:
                a, b = b, a
            f1 = canon(a, c, rint(2))
            f2 = canon(b, d, rint(2))
            if f1 == f2:
                continue
            if f1 in keyset and f1 != k1 and f1 != k2:
                continue
            if f2 in keyset and f2 != k1 and f2 != k2:
                continue
            if (f1 == k1 and f2 == k2) or (f1 == k2 and f2 == k1):
                continue
            lo, hi = (i, j) if i < j else (j, i)
            rest1 = K[:4 * lo] + K[4 * lo + 4:4 * hi] + K[4 * hi + 4:]
            o1 = ok[k1]
            o2 = ok[k2]
            n1 = ok[f1]
            n2 = ok[f2]
            delta = (TO(n1, rest1) + TO(n2, rest1 + n1)
                     - TO(o1, rest1 + o2) - TO(o2, rest1))
            if delta <= 0 or rnd() < exp(-delta / T):
                E[i] = (a, c, f1)
                E[j] = (b, d, f2)
                K[4 * i:4 * i + 4] = n1
                K[4 * j:4 * j + 4] = n2
                keyset.discard(k1)
                keyset.discard(k2)
                keyset.add(f1)
                keyset.add(f2)
                cost += delta
            else:
                continue
        if cost < best:
            best = cost
            bestE = list(E)
            if cost <= 0:
                break
    return pts_of(bestE), best


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
    # greedy extension
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
    final_deadline = t0 + time_limit - 0.25
    deadline = t0 + time_limit - 0.9
    partial = []
    if n % 2 == 1 or n <= 6:
        sol, partial = _bt(n, deadline, rng)
        if sol:
            return sol
        return _repair(n, partial, rng, final_deadline)
    if n <= 16:
        sol, partial = _bt(n, t0 + 0.3 * time_limit, rng)
        if sol:
            return sol
    pts, cost = _sa_rot4(n, deadline, rng)
    pts = [(int(x), int(y)) for (x, y) in pts]
    if cost == 0:
        return pts
    res = _repair(n, pts, rng, final_deadline)
    if partial and len(partial) > len(res):
        res2 = _repair(n, partial, rng, final_deadline)
        if len(res2) > len(res):
            res = res2
    return res
