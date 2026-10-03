# g008 (parent g001): Symmetric orbit DFS that tries asymmetric exact completion at deep dead ends, then full-grid numpy large-neighbourhood search. LNS forces in blocked cells and exactly repairs.
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


class _Found(Exception):
    def __init__(self, pts):
        Exception.__init__(self)
        self.pts = pts


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
        self.bline = [(1 << a) | internal[a] for a in range(N)]
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


def _complete(model, placed, ld, counts, budget, rng):
    """Asymmetric exact completion over line-free cells in deficient rows/cols."""
    n = model.n
    orbits = model.orbits
    orb_of = model.orb_of
    pts = []
    for o in placed:
        pts.extend(orbits[o])
    rowc = list(counts[:n])
    colc = list(counts[n:2 * n])
    free = model.ALL & ~ld
    drows = [x for x in range(n) if rowc[x] < 2]
    dcols = [y for y in range(n) if colc[y] < 2]
    cands = []
    for x in drows:
        b0 = x * n
        for y in dcols:
            o = orb_of[b0 + y]
            if o >= 0 and (free >> o) & 1:
                cands.append((x, y))
    bestx = []
    extra = []
    nodes = [0]

    def rec(cands):
        nodes[0] += 1
        if len(extra) > len(bestx):
            bestx[:] = extra
        if not cands or nodes[0] > budget:
            return
        rc = {}
        cc = {}
        for (x, y) in cands:
            rc[x] = rc.get(x, 0) + 1
            cc[y] = cc.get(y, 0) + 1
        u1 = 0
        for x, v in rc.items():
            u1 += min(v, 2 - rowc[x])
        u2 = 0
        for y, v in cc.items():
            u2 += min(v, 2 - colc[y])
        if len(extra) + min(u1, u2) <= len(bestx):
            return
        r = min(rc, key=rc.get)
        opts = [c for c in cands if c[0] == r]
        q = opts[rng.randrange(len(opts))]
        qx, qy = q
        dirs = set()
        for (px, py) in pts:
            dx = px - qx
            dy = py - qy
            g = gcd(dx, dy)
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            dirs.add((dx, dy))
        rowc[qx] += 1
        colc[qy] += 1
        nc = []
        for d in cands:
            if d == q:
                continue
            ex, ey = d
            if rowc[ex] >= 2 or colc[ey] >= 2:
                continue
            dx = ex - qx
            dy = ey - qy
            g = gcd(dx, dy)
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            if (dx, dy) in dirs:
                continue
            nc.append(d)
        pts.append(q)
        extra.append(q)
        rec(nc)
        pts.pop()
        extra.pop()
        rowc[qx] -= 1
        colc[qy] -= 1
        if nodes[0] > budget:
            return
        rec([d for d in cands if d != q])

    rec(cands)
    return pts, list(bestx)


def _run(model, rng, budget, deadline, best, noise, bc, ctl):
    n = model.n
    ALL = model.ALL
    lm, lm2 = model.lm, model.lm2
    contrib = model.contrib
    base = model.base
    bline = model.bline
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
    THR = 8

    def on_dead(placed, ld, counts, npts):
        if npts < target - THR and npts < best[0]:
            return
        now = tm()
        if ctl['spent'] > 0.3 * (now - ctl['start']) + 0.05:
            return
        pts, ex = _complete(model, placed, ld, counts, 400, rng)
        ctl['spent'] += tm() - now
        tot = len(pts) + len(ex)
        if tot > bc[0]:
            bc[0] = tot
            bc[1] = list(pts) + list(ex)
        if tot >= target:
            raise _Found(list(pts) + list(ex))

    def rec(dead, ld, counts, placed, npts):
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
        deadend = False
        for i in reps:
            ci = counts[i]
            if ci < 2:
                m = avail & lm[i]
                if not m:
                    deadend = True
                    break
                c = pc(m)
                if ci == 0 and c == 1 and not (m & lm2[i]):
                    deadend = True
                    break
                if c < bestc:
                    bestc = c
                    bm = m
                    bi = i
        if deadend or bm == 0:
            on_dead(placed, ld, counts, npts)
            return None
        lmi = lm[bi]
        lm2i = lm2[bi]
        need2 = counts[bi] == 0
        children = []
        for a in _bits(bm):
            pm = 0
            row = a * N
            for b in placed:
                v = cache[row + b]
                if v is None:
                    v = cslow(a, b)
                pm |= v
            nl = ld | bline[a] | pm
            nd = dead | base[a] | pm
            nc = counts[:]
            for idx, k in contrib[a]:
                v = nc[idx] + k
                nc[idx] = v
                if v >= 2:
                    nd |= lm[idx]
                else:
                    nd |= lm2[idx]
            sc = pc(ALL & ~nd) + rnd() * noise
            children.append((sc, a, nd, nl, nc))
        children.sort(key=keyf, reverse=True)
        failed = 0
        for sc, a, nd, nl, nc in children:
            ba = 1 << a
            placed.append(a)
            r = rec(nd | failed, nl, nc, placed, npts + size[a])
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
        res = rec(0, 0, [0] * (2 * n), [], 0)
    except _Abort:
        return None
    if res:
        return [model.orbits[o] for o in res]
    return None


# ---------------- full-grid LNS phase ----------------

def _build_lid(n):
    import numpy as np
    N2 = n * n
    idx = np.arange(N2, dtype=np.int64)
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
    key = (A * (2 * n + 1) + (B + n)) * (4 * n * n + 1) + (C + 2 * n * n)
    del A, B, C
    _, inv = np.unique(key.ravel(), return_inverse=True)
    del key
    NL = int(inv.max()) + 1
    LID = inv.reshape(N2, N2).astype(np.int32)
    return LID, NL


def _lns(n, init, deadline, rng):
    import numpy as np
    LID, NL = _build_lid(n)
    N2 = n * n
    target = 2 * n
    cnt = np.zeros(NL, dtype=np.int32)
    X = np.arange(N2) // n
    Y = np.arange(N2) % n
    S = []
    inS = np.zeros(N2, dtype=bool)
    rowc = np.zeros(n, dtype=np.int64)
    colc = np.zeros(n, dtype=np.int64)
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

    def rem(c):
        if S and S[-1] == c:
            S.pop()
        else:
            S.remove(c)
        inS[c] = False
        rowc[c // n] -= 1
        colc[c % n] -= 1
        if S:
            ids = np.unique(LID[c, S])
            v = cnt[ids]
            cnt[ids] = np.where(v == 2, 0, v - 1)

    def is_free(c):
        if inS[c] or rowc[c // n] >= 2 or colc[c % n] >= 2:
            return False
        if not S:
            return True
        return not bool((cnt[LID[c, S]] >= 2).any())

    for c in init:
        if is_free(c):
            add(c)

    def free_cands():
        if not S:
            ok = ~inS
        else:
            Sa = np.array(S, dtype=np.int64)
            F = (cnt[LID[:, Sa]] >= 2).any(axis=1)
            ok = ~F & ~inS
        ok &= (rowc[X] < 2) & (colc[Y] < 2)
        return np.nonzero(ok)[0]

    def repair(cands, budget):
        bestx = []
        extra = []
        nodes = [0]

        def rec(cands):
            nodes[0] += 1
            if len(extra) > len(bestx):
                bestx[:] = extra
            if len(cands) == 0 or nodes[0] > budget:
                return
            r = cands // n
            cc = cands - r * n
            br = np.bincount(r, minlength=n)
            bcn = np.bincount(cc, minlength=n)
            u = min(int(np.minimum(br, 2 - rowc).sum()), int(np.minimum(bcn, 2 - colc).sum()))
            if len(extra) + u <= len(bestx):
                return
            brm = np.where(br > 0, br, 1 << 30)
            row = int(np.argmin(brm))
            opts = cands[r == row]
            q = int(opts[rng.randrange(len(opts))])
            add(q)
            extra.append(q)
            nc = cands[cands != q]
            if len(nc):
                nc = nc[(rowc[nc // n] < 2) & (colc[nc % n] < 2)]
            if len(nc):
                nc = nc[cnt[LID[q, nc]] < 2]
            rec(nc)
            extra.pop()
            rem(q)
            if nodes[0] > budget:
                return
            rec(cands[cands != q])

        rec(cands)
        return list(bestx)

    # initial fill
    ex = repair(free_cands(), 300)
    for c in ex:
        add(c)
    best = list(S)
    it = 0
    while len(S) < target and time.time() < deadline:
        it += 1
        s = len(S)
        Sa = np.array(S, dtype=np.int64)
        drows = np.nonzero(rowc < 2)[0]
        dcols = np.nonzero(colc < 2)[0]
        T = (drows[:, None] * n + dcols[None, :]).ravel()
        T = T[~inS[T]]
        T2 = T[tabu[T] < it]
        if len(T2):
            T = T2
        removed = []
        t = -1
        if len(T):
            k = min(len(T), 12)
            samp = T[nrng.choice(len(T), k, replace=False)]
            L = LID[samp][:, Sa]
            B = cnt[L] >= 2
            bsc = 1e9
            bj = -1
            for j in range(k):
                lst = L[j][B[j]]
                sc = len(lst) - len(np.unique(lst)) + rng.random() * 0.7
                if sc < bsc:
                    bsc = sc
                    bj = j
            if bsc <= 6:
                t = int(samp[bj])
                Lj = L[bj]
                lines = np.unique(Lj[B[bj]])
                for ln in lines:
                    onl = Sa[Lj == ln]
                    keep = rng.randrange(len(onl))
                    for q, p in enumerate(onl):
                        if q != keep:
                            removed.append(int(p))
        if t < 0:
            if len(S) >= 2:
                removed = [int(x) for x in rng.sample(S, 2)]
        elif rng.random() < 0.25 and len(S) > len(removed) + 1:
            p = S[rng.randrange(len(S))]
            if p not in removed:
                removed.append(p)
        for p in removed:
            rem(p)
            tabu[p] = it + 7 + rng.randrange(10)
        added = []
        if t >= 0 and is_free(t):
            add(t)
            added.append(t)
        ex = repair(free_cands(), 150)
        for c in ex:
            add(c)
            added.append(c)
        s2 = len(S)
        if s2 >= s or (s2 == s - 1 and rng.random() < 0.02):
            if s2 > len(best):
                best = list(S)
        else:
            for c in reversed(added):
                rem(c)
            for p in removed:
                add(p)
    if len(S) > len(best):
        best = list(S)
    return best


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
        names = ['rot4', 'rot4', 'dia2']
    else:
        names = ['rot2', 'dia2', 'rot2', 'dia1']
    if n <= 8:
        names.append('iden')
    bt_deadline = t0 + time_limit * 0.62
    if n <= 12:
        bt_deadline = deadline
    models = {}
    counters = dict((nm, 0) for nm in names)
    best = [0, []]
    bc = [0, []]
    ctl = {'spent': 0.0, 'start': time.time()}
    unit = max(40, 4 * n)
    gi = 0
    try:
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
            res = _run(model, rng, budget, bt_deadline, best, noise, bc, ctl)
            if res:
                pts = []
                for cells in res:
                    pts.extend(cells)
                return [(int(x), int(y)) for (x, y) in pts]
    except _Found as e:
        return [(int(x), int(y)) for (x, y) in e.pts]
    if bc[0] >= best[0] and bc[1]:
        init_pts = [(int(x), int(y)) for (x, y) in bc[1]]
    else:
        init_pts = []
        for cells in best[1]:
            init_pts.extend(cells)
        init_pts = [(int(x), int(y)) for (x, y) in init_pts]
    if time.time() < deadline - 1.5:
        try:
            res = _lns(n, [x * n + y for (x, y) in init_pts], deadline, rng)
        except Exception:
            res = None
        if res and len(res) > len(init_pts):
            return [(int(c // n), int(c % n)) for c in res]
    return init_pts
