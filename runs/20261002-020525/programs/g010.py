# g010 (parent g007): After a rot4/dia2 orbit ILS phase, refine near-solutions in the finer common subgroup rot2 (pairs), seeded from the best symmetric states, with seed-based restarts.
import time
import random
from math import gcd
import numpy as np

try:
    _pc = int.bit_count
except AttributeError:
    def _pc(x):
        return bin(x).count('1')


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


class _ILS(object):
    def __init__(self, model, rng, target, seedS=None, lim=1500):
        self.m = model
        self.rng = rng
        self.target = target
        self.L = 2 * model.n
        self.reps = list(model.reps)
        self.S = []
        self.d = 0
        self.cnt = [0] * self.L
        self.val = 0
        self.best = []
        self.bestval = 0
        self.it = 0
        self.lastimp = 0
        self.tabu = [0] * model.N
        self.unit = max(model.size)
        self.started = False
        self.lim = lim
        self.end = 0.0
        self.seedS = seedS
        self.nrest = 0

    def _val(self, S):
        size = self.m.size
        return sum(size[a] for a in S)

    def dead_of(self, S):
        m = self.m
        N = m.N
        base = m.base
        cache = m.cache
        csl = m.compat_slow
        contrib = m.contrib
        lm = m.lm
        lm2 = m.lm2
        d = 0
        cnt = [0] * self.L
        for i in range(len(S)):
            a = S[i]
            d |= base[a]
            row = a * N
            for j in range(i):
                b = S[j]
                v = cache[row + b]
                if v is None:
                    v = csl(a, b)
                d |= v
            for idx, k in contrib[a]:
                cnt[idx] += k
        for idx in self.reps:
            c = cnt[idx]
            if c >= 2:
                d |= lm[idx]
            elif c == 1:
                d |= lm2[idx]
        return d, cnt

    def add_dead(self, d, cnt, S, a):
        m = self.m
        cache = m.cache
        csl = m.compat_slow
        d |= m.base[a]
        row = a * m.N
        for b in S:
            v = cache[row + b]
            if v is None:
                v = csl(a, b)
            d |= v
        lm = m.lm
        lm2 = m.lm2
        for idx, k in m.contrib[a]:
            if cnt[idx] + k >= 2:
                d |= lm[idx]
            else:
                d |= lm2[idx]
        return d

    def fill(self, noise):
        m = self.m
        ALL = m.ALL
        lm = m.lm
        contrib = m.contrib
        rng = self.rng
        rnd = rng.random
        S = self.S
        cnt = self.cnt
        d = self.d
        reps = self.reps
        while True:
            free = ALL & ~d
            if not free:
                break
            bc = 1 << 30
            bm = 0
            for i in reps:
                if cnt[i] < 2:
                    mm = free & lm[i]
                    if mm:
                        c = _pc(mm)
                        if c < bc:
                            bc = c
                            bm = mm
            if not bm:
                bm = free
            cands = _bits(bm)
            if len(cands) > 8:
                cands = rng.sample(cands, 8)
            bs = -1.0
            ba = -1
            bd = 0
            for a in cands:
                nd = self.add_dead(d, cnt, S, a)
                sc = _pc(ALL & ~nd) + rnd() * noise
                if sc > bs:
                    bs = sc
                    ba = a
                    bd = nd
            S.append(ba)
            for idx, k in contrib[ba]:
                cnt[idx] += k
            d = bd
        self.d = d

    def _apply(self, S):
        self.S = S
        self.d, self.cnt = self.dead_of(S)
        self.fill(1.0)
        self.val = self._val(self.S)

    def improve(self):
        m = self.m
        ALL = m.ALL
        size = m.size
        N = m.N
        cache = m.cache
        csl = m.compat_slow
        base = m.base
        lm = m.lm
        lm2 = m.lm2
        contrib = m.contrib
        reps = self.reps
        S = self.S
        L = len(S)
        if L == 0:
            return False
        cnt = self.cnt
        pres = [None] * L
        sufs = [None] * L
        A = [0] * L
        for i in range(L):
            a = S[i]
            row = a * N
            lst = [0] * i
            for j in range(i):
                b = S[j]
                v = cache[row + b]
                if v is None:
                    v = csl(a, b)
                lst[j] = v
            pre = [0] * (i + 1)
            acc = 0
            for j in range(i):
                pre[j] = acc
                acc |= lst[j]
            pre[i] = acc
            A[i] = acc
            suf = [0] * (i + 1)
            acc = 0
            for j in range(i - 1, -1, -1):
                suf[j] = acc
                acc |= lst[j]
            pres[i] = pre
            sufs[i] = suf
        PA = [0] * (L + 1)
        acc = 0
        for i in range(L):
            PA[i] = acc
            acc |= A[i]
        PA[L] = acc
        bpre = [0] * (L + 1)
        acc = 0
        for i in range(L):
            bpre[i] = acc
            acc |= base[S[i]]
        bpre[L] = acc
        bsuf = [0] * (L + 1)
        for i in range(L - 1, -1, -1):
            bsuf[i] = bsuf[i + 1] | base[S[i]]
        order = list(range(L))
        self.rng.shuffle(order)
        for si in order:
            s = S[si]
            D = PA[si] | bpre[si] | bsuf[si + 1]
            for i in range(si + 1, L):
                D |= pres[i][si] | sufs[i][si]
            c2 = cnt[:]
            for idx, k in contrib[s]:
                c2[idx] -= k
            for r in reps:
                c = c2[r]
                if c >= 2:
                    D |= lm[r]
                elif c == 1:
                    D |= lm2[r]
            F = ALL & ~D & ~(1 << s)
            if not F:
                continue
            Sp = S[:si] + S[si + 1:]
            sz = size[s]
            fl = _bits(F)
            self.rng.shuffle(fl)
            for a in fl:
                if size[a] > sz:
                    self._apply(Sp + [a])
                    return True
                da = self.add_dead(D, c2, Sp, a)
                F2 = F & ~da
                if F2:
                    for b in _bits(F2):
                        if size[a] + size[b] > sz:
                            self._apply(Sp + [a, b])
                            return True
        return False

    def removal(self, c):
        m = self.m
        S = self.S
        base = m.base
        cache = m.cache
        csl = m.compat_slow
        N = m.N
        lm = m.lm
        contrib = m.contrib
        rnd = self.rng.random
        R = set()
        bc = 1 << c
        for a in S:
            if base[a] & bc:
                R.add(a)
        for i in range(len(S)):
            a = S[i]
            if a in R:
                continue
            row = a * N
            for j in range(i):
                b = S[j]
                if b in R:
                    continue
                v = cache[row + b]
                if v is None:
                    v = csl(a, b)
                if v & bc:
                    if rnd() < 0.5:
                        R.add(a)
                        break
                    else:
                        R.add(b)
        cnt = self.cnt[:]
        for r in R:
            for idx, k in contrib[r]:
                cnt[idx] -= k
        for idx, k in contrib[c]:
            guard = 0
            while cnt[idx] + k > 2 and guard < 10:
                guard += 1
                cand = [a for a in S if a not in R and (lm[idx] >> a) & 1]
                if not cand:
                    break
                r = self.rng.choice(cand)
                R.add(r)
                for idx2, k2 in contrib[r]:
                    cnt[idx2] -= k2
        return R

    def perturb(self):
        m = self.m
        N = m.N
        size = m.size
        lm = m.lm
        rng = self.rng
        S = self.S
        it = self.it
        tabu = self.tabu
        Sb = 0
        for a in S:
            Sb |= 1 << a
        cand = []
        if rng.random() < 0.6:
            defm = 0
            for i in self.reps:
                if self.cnt[i] < 2:
                    defm |= lm[i]
            defm &= ~Sb
            if defm:
                bl = _bits(defm)
                cand = rng.sample(bl, min(6, len(bl)))
        if not cand:
            cand = [rng.randrange(N) for _ in range(6)]
        bestc = -1
        bestR = None
        bestcost = 1e18
        for c in cand:
            if (Sb >> c) & 1 or tabu[c] > it:
                continue
            R = self.removal(c)
            cost = sum(size[r] for r in R) - size[c] + rng.random() * 1.5
            if cost < bestcost:
                bestcost = cost
                bestc = c
                bestR = R
        if bestc < 0:
            for _ in range(50):
                c = rng.randrange(N)
                if not (Sb >> c) & 1:
                    bestc = c
                    bestR = self.removal(c)
                    break
            if bestc < 0:
                self.S = S[:]
                self.cnt = self.cnt[:]
                return
        newS = [a for a in S if a not in bestR] + [bestc]
        for r in bestR:
            tabu[r] = it + 3 + rng.randrange(8)
        self.S = newS
        self.d, self.cnt = self.dead_of(newS)

    def extend(self, Sp, d, cnt, need, ctr):
        if need <= 0:
            return []
        ctr[0] += 1
        if ctr[0] > 3000:
            return None
        ALL = self.m.ALL
        free = ALL & ~d
        if not free or _pc(free) * self.unit < need:
            return None
        size = self.m.size
        contrib = self.m.contrib
        for a in _bits(free):
            da = self.add_dead(d, cnt, Sp, a) | ((2 << a) - 1)
            c2 = cnt[:]
            for idx, k in contrib[a]:
                c2[idx] += k
            Sp.append(a)
            r = self.extend(Sp, da, c2, need - size[a], ctr)
            Sp.pop()
            if r is not None:
                return [a] + r
            if ctr[0] > 3000:
                return None
        return None

    def deep(self):
        S = self.S
        L = len(S)
        m = self.m
        ALL = m.ALL
        size = m.size
        pairs = [(i, j) for i in range(L) for j in range(i)]
        self.rng.shuffle(pairs)
        ctr = [0]
        tm = time.time
        cntp = 0
        for (i, j) in pairs:
            cntp += 1
            if (cntp & 7) == 0 and tm() > self.end:
                return False
            si, sj = S[i], S[j]
            Sp = [S[k] for k in range(L) if k != i and k != j]
            ds, cs_ = self.dead_of(Sp)
            ds |= (1 << si) | (1 << sj)
            F = ALL & ~ds
            need = size[si] + size[sj] + 1
            if _pc(F) * self.unit < need:
                continue
            ctr[0] = 0
            r = self.extend(Sp, ds, cs_, need, ctr)
            if r:
                self._apply(Sp + r)
                return True
        return False

    def restart(self):
        if self.seedS is not None:
            S = list(self.seedS)
            if self.nrest > 0 and len(S) > 2:
                k = self.rng.randint(1, max(2, len(S) // 4))
                self.rng.shuffle(S)
                S = S[k:]
            self.S = S
            self.d, self.cnt = self.dead_of(S)
            self.fill(1.0)
        else:
            self.S = []
            self.d = 0
            self.cnt = [0] * self.L
            self.fill(4.0)
        self.nrest += 1
        k = 0
        while k < 60 and time.time() < self.end and self.improve():
            k += 1
        self.val = self._val(self.S)
        if self.val > self.bestval:
            self.bestval = self.val
            self.best = list(self.S)
        self.lastimp = self.it

    def step(self):
        self.it += 1
        rng = self.rng
        pS, pd, pcn, pv = self.S, self.d, self.cnt, self.val
        self.perturb()
        self.fill(0.7)
        k = 0
        while k < 30 and self.improve():
            k += 1
        v = self._val(self.S)
        self.val = v
        if v < self.target and v >= self.target - self.unit and rng.random() < 0.12:
            if self.deep():
                v = self._val(self.S)
                self.val = v
        if v > self.bestval:
            self.bestval = v
            self.best = list(self.S)
            self.lastimp = self.it
        elif v < pv:
            dl = (pv - v) / float(self.unit)
            dst = (self.bestval - v) / float(self.unit)
            if rng.random() > 1.0 / (1.0 + dl * dst):
                self.S, self.d, self.cnt, self.val = pS, pd, pcn, pv
        if self.it - self.lastimp > self.lim:
            self.restart()

    def run(self, end):
        self.end = end
        if not self.started:
            self.started = True
            self.restart()
        while self.bestval < self.target and time.time() < end:
            self.step()

    def points(self):
        pts = []
        for o in self.best:
            pts.extend(self.m.orbits[o])
        return [(int(x), int(y)) for (x, y) in pts]


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


def _run_states(states, stop, slc, gstate):
    while time.time() < stop and states:
        for st in states:
            now = time.time()
            if now >= stop:
                break
            end = min(stop, now + slc)
            try:
                st.run(end)
            except Exception:
                pass
            if st.bestval > gstate[0]:
                gstate[0] = st.bestval
                gstate[1] = st.points()
            if gstate[0] >= gstate[2]:
                return True
    return gstate[0] >= gstate[2]


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
        names = ['rot4', 'dia2']
    else:
        names = ['rot2', 'dia2']
    if n <= 10:
        names.append('iden')
    ls_time = 0.07 * time_limit if n >= 12 else 0.0
    sym_deadline = deadline - ls_time
    phase1_end = t0 + 0.55 * time_limit if n % 2 == 0 else sym_deadline
    phase1_end = min(phase1_end, sym_deadline)
    states = []
    for nm in names:
        try:
            model = _Model(n, nm)
        except Exception:
            continue
        if model.N > 0:
            states.append(_ILS(model, rng, target))
    gstate = [0, [], target]
    slc = max(0.5, time_limit / 16.0)
    if _run_states(states, phase1_end, slc, gstate):
        return gstate[1]
    # phase 2: refine in rot2 (common subgroup), seeded from phase-1 bests
    if n % 2 == 0 and time.time() < sym_deadline - 0.5:
        try:
            m2 = _Model(n, 'rot2')
            st2 = []
            seeds = []
            for st in states:
                if st.best:
                    pts = st.points()
                    S = sorted(set(m2.orb_of[x * n + y] for (x, y) in pts))
                    if all(o >= 0 for o in S):
                        seeds.append((st.bestval, S))
            seeds.sort(key=lambda t: -t[0])
            for _, S in seeds:
                st2.append(_ILS(m2, rng, target, seedS=S, lim=600))
            if not st2:
                st2.append(_ILS(m2, rng, target, lim=1500))
            if _run_states(st2, sym_deadline, slc, gstate):
                return gstate[1]
        except Exception:
            pass
    pts = gstate[1]
    if len(pts) < target and time.time() < deadline - 0.2:
        try:
            p2 = _local_search(n, pts, deadline, rng)
            if len(p2) > len(pts):
                pts = p2
        except Exception:
            pass
    return [(int(x), int(y)) for (x, y) in pts]
