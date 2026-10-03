# g004 (parent s0): Replace backtracking with an ARW-style local search for maximum independent sets over symmetric orbits. It runs quarter-turn, then diagonal, then half-turn, then free-cell phases, each seeded hierarchically, with numpy blocker counts.
import time
import random
from math import gcd
import numpy as np


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


def _valid_cells(cells):
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


class _Model(object):
    def __init__(self, n, name):
        self.n = n
        self.name = name
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
                if len(cells) == 1 or _valid_cells(cells):
                    oid = len(orbits)
                    orbits.append(cells)
                    for (a, b) in cells:
                        orb_of[a * n + b] = oid
        self.orb_of = orb_of
        self.orbits = orbits
        N = len(orbits)
        self.N = N
        self.size = [len(c) for c in orbits]
        self.lcache = {}
        self.pc = {}
        internal = [set() for _ in range(N)]
        for oid, cells in enumerate(orbits):
            p = cells[0]
            s = set()
            for q in cells[1:]:
                s.update(self.line(p, q).tolist())
            s.discard(oid)
            internal[oid] = s
        hits = [set() for _ in range(N)]
        for a in range(N):
            for b in internal[a]:
                hits[b].add(a)
        self.B = [np.array(sorted(internal[a] | hits[a]), dtype=np.intp) for a in range(N)]

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
        v = self.lcache.get(key)
        if v is not None:
            return v
        orb_of = self.orb_of
        lst = []
        while 0 <= x < n and 0 <= y < n:
            o = orb_of[x * n + y]
            if o >= 0:
                lst.append(o)
            x += dx
            y += dy
        v = np.array(lst, dtype=np.intp)
        self.lcache[key] = v
        return v

    def R(self, a, b):
        N = self.N
        key = a * N + b if a < b else b * N + a
        v = self.pc.get(key)
        if v is None:
            p = self.orbits[a][0]
            parts = [self.line(p, q) for q in self.orbits[b]]
            arr = np.concatenate(parts) if len(parts) > 1 else parts[0]
            arr = arr[(arr != a) & (arr != b)]
            if len(self.pc) > 350000:
                self.pc.clear()
            self.pc[key] = arr
            v = arr
        return v


class _ILS(object):
    def __init__(self, model, rng):
        self.m = model
        N = model.N
        self.blk = np.zeros(N, np.int32)
        self.inS = np.zeros(N, dtype=bool)
        self.S = []
        self.rng = rng
        self.pts = 0

    def add(self, u):
        R = self.m.R
        blk = self.blk
        for b in self.S:
            r = R(u, b)
            if len(r):
                blk[r] += 1
        Bu = self.m.B[u]
        if len(Bu):
            blk[Bu] += 1
        self.S.append(u)
        self.inS[u] = True
        self.pts += self.m.size[u]

    def remove(self, u):
        S = self.S
        if S[-1] == u:
            S.pop()
        else:
            S.remove(u)
        self.inS[u] = False
        self.pts -= self.m.size[u]
        R = self.m.R
        blk = self.blk
        for b in S:
            r = R(u, b)
            if len(r):
                blk[r] -= 1
        Bu = self.m.B[u]
        if len(Bu):
            blk[Bu] -= 1

    def free(self):
        return np.flatnonzero((self.blk == 0) & ~self.inS)

    def fill(self, ncand):
        rng = self.rng
        added = []
        while True:
            fr = self.free()
            L = len(fr)
            if L == 0:
                break
            if L == 1:
                u = int(fr[0])
            else:
                if L > ncand:
                    cs = [int(fr[i]) for i in rng.sample(range(L), ncand)]
                else:
                    cs = [int(x) for x in fr]
                bestv = -1.0
                u = cs[0]
                for c in cs:
                    self.add(c)
                    v = len(self.free()) + rng.random()
                    self.remove(c)
                    if v > bestv:
                        bestv = v
                        u = c
            self.add(u)
            added.append(u)
        return added

    def force(self, u):
        inS = self.inS
        R = self.m.R
        rng = self.rng
        Bu = self.m.B[u]
        rem = set(int(x) for x in Bu[inS[Bu]]) if len(Bu) else set()
        pairs = set()
        for b in self.S:
            if b in rem:
                continue
            r = R(u, b)
            if len(r):
                ms = r[inS[r]]
                if len(ms):
                    for a in set(ms.tolist()):
                        if a != b and a not in rem:
                            pairs.add((a, b) if a < b else (b, a))
        pairs = list(pairs)
        while pairs:
            freq = {}
            for a, b in pairs:
                freq[a] = freq.get(a, 0) + 1
                freq[b] = freq.get(b, 0) + 1
            mx = max(freq.values())
            opts = [x for x, v in freq.items() if v == mx]
            x = opts[rng.randrange(len(opts))]
            rem.add(x)
            pairs = [p for p in pairs if p[0] != x and p[1] != x]
        for x in rem:
            self.remove(x)
        if self.blk[u] == 0 and not self.inS[u]:
            self.add(u)
        return rem


def _search(model, init, deadline, target, rng, tenure, ncand, restart_lim, fresh_prob):
    ils = _ILS(model, rng)
    for o in init:
        if not ils.inS[o] and ils.blk[o] == 0:
            ils.add(o)
    ils.fill(ncand)
    best = list(ils.S)
    bestp = ils.pts
    N = model.N
    last = np.full(N, -10 ** 9, dtype=np.int64)
    it = 0
    stag = 0
    tm = time.time
    while bestp < target:
        if (it & 7) == 0 and tm() > deadline:
            break
        it += 1
        cur = ils.pts
        ok = (~ils.inS) & ((it - last) > tenure)
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            cand = np.flatnonzero(~ils.inS)
            if len(cand) == 0:
                break
        if rng.random() < 0.9:
            vals = ils.blk[cand]
            mn = vals.min()
            pool = cand[vals == mn]
            u = int(pool[rng.randrange(len(pool))])
        else:
            u = int(cand[rng.randrange(len(cand))])
        rem = ils.force(u)
        for x in rem:
            last[x] = it
        added = ils.fill(ncand)
        if ils.pts < cur and rng.random() < 0.95:
            for a in added:
                ils.remove(a)
            if ils.inS[u]:
                ils.remove(u)
            for x in rem:
                ils.add(x)
        if ils.pts > bestp:
            bestp = ils.pts
            best = list(ils.S)
            stag = 0
        else:
            stag += 1
        if stag > restart_lim:
            stag = 0
            ils = _ILS(model, rng)
            last[:] = -10 ** 9
            if rng.random() < fresh_prob:
                ils.fill(ncand)
            else:
                for o in best:
                    ils.add(o)
                for _ in range(rng.randint(2, 4)):
                    nm = np.flatnonzero(~ils.inS)
                    if len(nm) == 0:
                        break
                    uu = int(nm[rng.randrange(len(nm))])
                    ils.force(uu)
                    ils.fill(ncand)
            if ils.pts > bestp:
                bestp = ils.pts
                best = list(ils.S)
    return best, bestp


def _clean(pts):
    pts = list(dict.fromkeys(pts))
    while True:
        lines = {}
        P = len(pts)
        for i in range(P):
            x1, y1 = pts[i]
            for j in range(i + 1, P):
                x2, y2 = pts[j]
                dx, dy = x2 - x1, y2 - y1
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                key = (dx, dy, dy * x1 - dx * y1)
                s = lines.get(key)
                if s is None:
                    lines[key] = {i, j}
                else:
                    s.add(i)
                    s.add(j)
        cnt = {}
        for s in lines.values():
            if len(s) >= 3:
                for i in s:
                    cnt[i] = cnt.get(i, 0) + 1
        if not cnt:
            return pts
        worst = max(cnt, key=lambda i: cnt[i])
        pts.pop(worst)


def construct(n, time_limit, seed):
    t0 = time.time()
    T = max(0.3, time_limit - 0.8)
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
        plan = [('rot4', 0.36), ('dia2', 0.56), ('rot2', 0.8), ('iden', 1.0)]
    else:
        plan = [('rot2', 0.45), ('dia2', 0.6), ('iden', 1.0)]
    bestpts = []
    for name, frac in plan:
        dl = t0 + T * frac
        if time.time() >= dl - 0.05:
            continue
        model = _Model(n, name)
        if model.N == 0:
            continue
        init = []
        if name in ('rot2', 'iden') and bestpts:
            ps = set(bestpts)
            os_ = set()
            for (x, y) in bestpts:
                o = model.orb_of[x * n + y]
                if o >= 0:
                    os_.add(o)
            for o in os_:
                if all(c in ps for c in model.orbits[o]):
                    init.append(o)
        if name == 'iden':
            tenure, ncand, rl, fp = 10, 2, 2500, 0.0
        elif name == 'rot2':
            tenure, ncand, rl, fp = 8, 3, 3000, 0.0
        else:
            tenure, ncand, rl, fp = 7, 4, 2500, 0.5
        best, bp = _search(model, init, dl, target, rng, tenure, ncand, rl, fp)
        pts = []
        for o in best:
            pts.extend(model.orbits[o])
        pts = [(int(x), int(y)) for (x, y) in pts]
        if len(pts) > len(bestpts):
            bestpts = pts
        if len(bestpts) >= target:
            break
    return _clean(bestpts)
