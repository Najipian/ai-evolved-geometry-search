# g013 (parent g008): Build full rot4/dia2 orbit-conflict rows lazily so each child's line mask is one C-level OR. Interleave a rot2 DFS for exact solutions and 2n-2 partials.
import time
import random
from math import gcd
from functools import reduce
from operator import or_, itemgetter

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
        n2 = n * n
        maps = _group_maps(n, name)
        orb_of = [-1] * n2
        seen = [False] * n2
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
        self.cidx = [[a * n + b for (a, b) in c] for c in orbits]
        self.vcells = [(q, 1 << orb_of[q]) for q in range(n2) if orb_of[q] >= 0]
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
            px, py = cells[0]
            msk = 0
            for (qx, qy) in cells[1:]:
                dx, dy = qx - px, qy - py
                g = gcd(dx, dy)
                msk |= self._walk(px, py, dx // g, dy // g)
            internal[oid] = msk & ~(1 << oid)
        hits = [0] * N
        for a in range(N):
            for b in _bits(internal[a]):
                hits[b] |= 1 << a
        self.internal, self.hits = internal, hits
        self.base = [(1 << a) | internal[a] | hits[a] for a in range(N)]
        self.bline = [(1 << a) | internal[a] for a in range(N)]
        # direction table
        W = 2 * n - 1
        dirtab = [0] * (W * W)
        dmap = {}
        for dx in range(-(n - 1), n):
            for dy in range(-(n - 1), n):
                if dx == 0 and dy == 0:
                    continue
                g = gcd(dx, dy)
                a = dx // g
                b = dy // g
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                k = (a, b)
                v = dmap.get(k)
                if v is None:
                    v = len(dmap) + 1
                    dmap[k] = v
                dirtab[(dx + n - 1) * W + dy + n - 1] = v
        self.dirtab = dirtab
        self.ndir = len(dmap) + 1
        self.rows = [None] * N
        self.nrows = 0
        per = max(1, N * (28 + N // 8 + 8))
        self.rowcap = max(150, 70000000 // per)
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

    def _walk(self, px, py, dx, dy):
        n = self.n
        orb = self.orb_of
        m = 0
        x, y = px, py
        while 0 <= x < n and 0 <= y < n:
            o = orb[x * n + y]
            if o >= 0:
                m |= 1 << o
            x += dx
            y += dy
        x, y = px - dx, py - dy
        while 0 <= x < n and 0 <= y < n:
            o = orb[x * n + y]
            if o >= 0:
                m |= 1 << o
            x -= dx
            y -= dy
        return m

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

    def make_row(self, a):
        n = self.n
        W = 2 * n - 1
        px, py = self.orbits[a][0]
        dt = self.dirtab
        off = (n - 1 - px) * W + (n - 1 - py)
        D = []
        for x in range(n):
            s = off + x * W
            D += dt[s:s + n]
        dm = [0] * self.ndir
        for q, bv in self.vcells:
            dm[D[q]] |= bv
        row = [0] * self.N
        i = 0
        for cl in self.cidx:
            v = 0
            for c in cl:
                v |= dm[D[c]]
            row[i] = v
            i += 1
        rows = self.rows
        if self.nrows >= self.rowcap:
            for j in range(self.N):
                rows[j] = None
            self.nrows = 0
        rows[a] = row
        self.nrows += 1
        return row


def _complete(model, placed, ld, counts, budget, rng):
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
    rows = model.rows
    mkrow = model.make_row
    orbits = model.orbits
    target = 2 * n
    reps = list(model.reps)
    rng.shuffle(reps)
    cnt = [0]
    pc = _pc
    tm = time.time
    rnd = rng.random
    keyf = lambda t: t[0]
    THR = 6

    def on_dead(placed, ld, counts, npts):
        if npts < target - THR or npts + 4 <= best[0]:
            return
        now = tm()
        if ctl['spent'] > 0.12 * (now - ctl['start']) + 0.05:
            return
        pts, ex = _complete(model, placed, ld, counts, 300, rng)
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
        if c0 > budget or ((c0 & 15) == 0 and tm() > deadline):
            raise _Abort()
        if npts > best[0]:
            best[0] = npts
            best[1] = [orbits[o] for o in placed]
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
        np_ = len(placed)
        ig = itemgetter(*placed) if np_ >= 2 else None
        p0 = placed[0] if np_ == 1 else -1
        children = []
        for a in _bits(bm):
            r = rows[a]
            if r is None:
                r = mkrow(a)
            if ig is not None:
                pm = reduce(or_, ig(r))
            elif p0 >= 0:
                pm = r[p0]
            else:
                pm = 0
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
        return [orbits[o] for o in res]
    return None


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
    target = 2 * n
    if n % 2 == 0:
        cyc = ['rot4', 'rot4', 'dia2', 'rot2']
        late = ['rot2', 'rot4', 'rot2']
    else:
        cyc = ['rot2', 'dia2', 'rot2', 'dia1']
        late = ['rot2', 'rot2', 'dia2']
    if n <= 8:
        cyc = cyc + ['iden']
    models = {}
    counters = {}
    best = [0, []]
    bc = [0, []]
    ctl = {'spent': 0.0, 'start': t0}
    late_t = t0 + 0.7 * time_limit
    gi = 0
    try:
        while time.time() < deadline:
            if time.time() > late_t and best[0] < target - 2 and bc[0] < target - 2:
                names = late
            else:
                names = cyc
            nm = names[gi % len(names)]
            gi += 1
            model = models.get(nm)
            if model is None:
                model = _Model(n, nm)
                models[nm] = model
            if model.N == 0:
                continue
            k = counters.get(nm, 0) + 1
            counters[nm] = k
            avg = max(1.0, float(sum(model.size)) / model.N)
            unit = max(40, int(4 * n * 4.0 / avg))
            budget = unit * _luby(k)
            noise = 3.0 if (k % 4) else 12.0
            res = _run(model, rng, budget, deadline, best, noise, bc, ctl)
            if res:
                pts = []
                for cells in res:
                    pts.extend(cells)
                return [(int(x), int(y)) for (x, y) in pts]
    except _Found as e:
        return [(int(x), int(y)) for (x, y) in e.pts]
    if bc[0] >= best[0] and bc[1]:
        return [(int(x), int(y)) for (x, y) in bc[1]]
    out = []
    for cells in best[1]:
        out.extend(cells)
    return [(int(x), int(y)) for (x, y) in out]
