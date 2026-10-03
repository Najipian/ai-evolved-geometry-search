# g015 (parent g010): Interleave the orbit ILS with a fast restarting exact DFS over rot4/dia2 orbits. The DFS branches on the most constrained line class, using forward checking, sibling exclusion and Luby restarts.
import time
import random
from math import gcd

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
        self.slc = 1.0

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


class _Abort(Exception):
    pass


def _luby(i):
    while True:
        k = 1
        while (1 << k) - 1 < i:
            k += 1
        if (1 << k) - 1 == i:
            return 1 << (k - 1)
        i = i - (1 << (k - 1)) + 1


class _DFS(object):
    def __init__(self, model, rng, target, unit=150):
        self.m = model
        self.rng = rng
        self.target = target
        self.reps = list(model.reps)
        self.best = []
        self.bestval = 0
        self.li = 0
        self.unit = unit
        self.slc = 2.0

    def attempt(self, budget, end):
        m = self.m
        ALL = m.ALL
        lm = m.lm
        lm2 = m.lm2
        contrib = m.contrib
        base = m.base
        cache = m.cache
        csl = m.compat_slow
        N = m.N
        size = m.size
        reps = self.reps
        rnd = self.rng.random
        L = 2 * m.n
        S = []
        cnt = [0] * L
        nodes = [0]
        tm = time.time
        W = 3.0
        noise = 2.5
        me = self

        def rec(d, val):
            if val > me.bestval:
                me.bestval = val
                me.best = list(S)
            free = ALL & ~d
            bk = 1e18
            bmm = 0
            for r in reps:
                c = cnt[r]
                if c < 2:
                    mm = free & lm[r]
                    if not mm:
                        return False
                    k = _pc(mm)
                    if c == 0 and k == 1 and not (mm & lm2[r]):
                        return False
                    key = (k - 2 + c) * 8 + k * 0.01 + rnd()
                    if key < bk:
                        bk = key
                        bmm = mm
            if not bmm:
                return True
            ch = []
            for a in _bits(bmm):
                nd = d | base[a]
                row = a * N
                for b in S:
                    v = cache[row + b]
                    if v is None:
                        v = csl(a, b)
                    nd |= v
                ca = contrib[a]
                for idx, k in ca:
                    c2 = cnt[idx] + k
                    if c2 >= 2:
                        nd |= lm[idx]
                    else:
                        nd |= lm2[idx]
                    cnt[idx] = c2
                nf = ALL & ~nd
                ok = True
                mins = 99
                for r in reps:
                    c = cnt[r]
                    if c < 2:
                        mm = nf & lm[r]
                        if not mm:
                            ok = False
                            break
                        k = _pc(mm)
                        if c == 0 and k == 1 and not (mm & lm2[r]):
                            ok = False
                            break
                        s = k - 2 + c
                        if s < mins:
                            mins = s
                for idx, k in ca:
                    cnt[idx] -= k
                if ok:
                    ch.append((_pc(nf) + W * mins + rnd() * noise, a, nd))
            ch.sort(reverse=True)
            excl = 0
            for sc, a, nd in ch:
                nodes[0] += 1
                if nodes[0] > budget:
                    raise _Abort()
                if (nodes[0] & 63) == 0 and tm() > end:
                    raise _Abort()
                S.append(a)
                ca = contrib[a]
                for idx, k in ca:
                    cnt[idx] += k
                if rec(nd | excl, val + size[a]):
                    return True
                S.pop()
                for idx, k in ca:
                    cnt[idx] -= k
                excl |= 1 << a
            return False

        try:
            if rec(0, 0):
                v = sum(size[a] for a in S)
                if v >= self.bestval:
                    self.bestval = v
                    self.best = list(S)
                return True
        except _Abort:
            pass
        except RecursionError:
            pass
        return False

    def run(self, end):
        while self.bestval < self.target and time.time() < end:
            self.li += 1
            budget = self.unit * _luby(self.li)
            self.attempt(budget, end)

    def points(self):
        pts = []
        for o in self.best:
            pts.extend(self.m.orbits[o])
        return [(int(x), int(y)) for (x, y) in pts]


def _run_states(states, stop, gstate):
    while time.time() < stop and states:
        for st in states:
            now = time.time()
            if now >= stop:
                break
            end = min(stop, now + st.slc)
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
    states = []
    for nm in names:
        try:
            model = _Model(n, nm)
        except Exception:
            continue
        if model.N <= 0:
            continue
        ils = _ILS(model, rng, target)
        ils.slc = max(0.4, time_limit / 20.0)
        dfs = _DFS(model, rng, target)
        dfs.slc = max(0.6, time_limit / 10.0)
        states.append(ils)
        states.append(dfs)
    gstate = [0, [], target]
    _run_states(states, deadline, gstate)
    pts = gstate[1]
    return [(int(x), int(y)) for (x, y) in pts]
