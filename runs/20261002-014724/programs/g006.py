# g006 (parent g005): Short DFS seeding, then tabu search over full quarter-turn cycle covers. Moves are 2-opt edge swaps that minimise collinear triples, scored by bitmask deltas with numpy counts.
import time
import random
from math import gcd
import numpy as np

try:
    (0).bit_count()
    _pc = int.bit_count
except Exception:
    def _pc(x):
        return bin(x).count('1')


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


def _build_orbits(n, name):
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
    return maps, orb_of, orbits


class _DModel(object):
    def __init__(self, n, name):
        self.n = n
        self.name = name
        maps, orb_of, orbits = _build_orbits(n, name)
        self.orb_of = orb_of
        self.orbits = orbits
        N = len(orbits)
        self.N = N
        self.sz = [len(c) for c in orbits]
        self.lc = {}
        self.rc = {}

        def canon(t, i):
            best = None
            for f in maps:
                if t == 0:
                    a = f(i, 0); b = f(i, 1)
                else:
                    a = f(0, i); b = f(1, i)
                key = (0, a[0]) if a[0] == b[0] else (1, a[1])
                if best is None or key < best:
                    best = key
            return best
        ls = set()
        for i in range(n):
            ls.add(canon(0, i))
            ls.add(canon(1, i))
        ls = sorted(ls)
        L = len(ls)
        self.L = L
        lidx = {l: j for j, l in enumerate(ls)}
        contrib = [[] for _ in range(N)]
        mask1 = [0] * L
        mask2 = [0] * L
        for o, cells in enumerate(orbits):
            cnt = {}
            for (x, y) in cells:
                j = lidx.get((0, x))
                if j is not None:
                    cnt[j] = cnt.get(j, 0) + 1
                j = lidx.get((1, y))
                if j is not None:
                    cnt[j] = cnt.get(j, 0) + 1
            for j, c in cnt.items():
                contrib[o].append((j, c))
                mask1[j] |= 1 << o
                if c >= 2:
                    mask2[j] |= 1 << o
        self.contrib = contrib
        self.mask1 = mask1
        self.mask2 = mask2
        B = []
        for o, cells in enumerate(orbits):
            mk = 1 << o
            px, py = cells[0]
            for (qx, qy) in cells[1:]:
                mk |= self.lmask(px, py, qx, qy)
            B.append(mk)
        self.B = B

    def lmask(self, px, py, qx, qy):
        dx, dy = qx - px, qy - py
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        key = (dx, dy, dy * px - dx * py)
        v = self.lc.get(key)
        if v is not None:
            return v
        n = self.n
        x, y = px, py
        while 0 <= x - dx < n and 0 <= y - dy < n:
            x -= dx
            y -= dy
        orb_of = self.orb_of
        mk = 0
        while 0 <= x < n and 0 <= y < n:
            o = orb_of[x * n + y]
            if o >= 0:
                mk |= 1 << o
            x += dx
            y += dy
        self.lc[key] = mk
        return mk

    def R(self, a, b):
        key = a * self.N + b if a < b else b * self.N + a
        v = self.rc.get(key)
        if v is None:
            px, py = self.orbits[a][0]
            v = 0
            for (qx, qy) in self.orbits[b]:
                v |= self.lmask(px, py, qx, qy)
            if len(self.rc) > 400000:
                self.rc.clear()
            self.rc[key] = v
        return v


class _Stop(Exception):
    pass


def _dfs(M, budget, rng, noise, deadline, rb):
    L = M.L
    deg = [0] * L
    chosen = []
    mask1 = M.mask1
    mask2 = M.mask2
    contrib = M.contrib
    B = M.B
    R = M.R
    sz = M.sz
    order = list(range(L))
    rng.shuffle(order)
    st = [0, 0]
    rnd = rng.random
    tm = time.time

    def rec(F):
        st[0] += 1
        if st[0] > budget:
            raise _Stop()
        if (st[0] & 127) == 0 and tm() > deadline:
            raise _Stop()
        if st[1] > rb[0]:
            rb[0] = st[1]
            rb[1] = list(chosen)
        bl = -1
        bc = 1 << 30
        bav = 0
        nF = ~F
        for li in order:
            d = 2 - deg[li]
            if d == 0:
                continue
            av = mask1[li] & nF
            c = _pc(av)
            if c < d:
                if c == 0 or not (av & mask2[li]):
                    return False
            if c < bc:
                bc = c
                bl = li
                bav = av
        if bl < 0:
            return True
        kids = []
        while bav:
            low = bav & -bav
            o = low.bit_length() - 1
            bav ^= low
            nf = F | B[o]
            for b in chosen:
                nf |= R(o, b)
            for (li, c) in contrib[o]:
                nd = deg[li] + c
                if nd >= 2:
                    nf |= mask1[li]
                elif nd == 1:
                    nf |= mask2[li]
            kids.append((_pc(nf) + noise * rnd(), o, nf))
        kids.sort()
        acc = 0
        for _, o, nf in kids:
            if acc >> o & 1:
                continue
            for (li, c) in contrib[o]:
                deg[li] += c
            chosen.append(o)
            st[1] += sz[o]
            if rec(nf | acc):
                return True
            chosen.pop()
            st[1] -= sz[o]
            for (li, c) in contrib[o]:
                deg[li] -= c
            acc |= 1 << o
        return False

    try:
        if rec(0):
            return list(chosen)
    except _Stop:
        pass
    return None


def _luby(i):
    while True:
        k = 1
        while (1 << k) - 1 < i:
            k += 1
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i = i - (1 << (k - 1)) + 1


# ---------------------------------------------------------------- tabu LS on cycle covers
class _LS(object):
    def __init__(self, M, rng):
        self.M = M
        N = M.N
        self.N = N
        self.rng = rng
        self.nb = (N + 7) // 8
        self.k = M.L
        ends = []
        for o in range(N):
            cl = []
            for (li, c) in M.contrib[o]:
                cl.extend([li] * c)
            ends.append(tuple(sorted(cl)))
        self.ends = ends
        self.ok = N > 0 and all(len(e) == 2 for e in ends)
        E = {}
        for o, e in enumerate(ends):
            if len(e) == 2:
                E.setdefault(e, []).append(o)
        self.E = E
        Bx = [M.B[o] & ~(1 << o) for o in range(N)]
        Bin = [0] * N
        for o in range(N):
            m = Bx[o]
            while m:
                low = m & -m
                c = low.bit_length() - 1
                m ^= low
                Bin[c] |= 1 << o
        self.Bx = Bx
        self.Bin = Bin
        self.rc = {}

    def rp(self, a, b):
        key = a * self.N + b if a < b else b * self.N + a
        v = self.rc.get(key)
        if v is None:
            v = self.M.R(a, b) & ~((1 << a) | (1 << b))
            self.rc[key] = v
        return v

    def vec(self, masks):
        nb = self.nb
        buf = b''.join([m.to_bytes(nb, 'little') for m in masks])
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(masks), nb)
        return np.unpackbits(arr, axis=1, bitorder='little').sum(axis=0, dtype=np.int64)[:self.N]

    def rand_cover(self):
        k = self.k
        rng = self.rng
        E = self.E
        for _ in range(200):
            perm = list(range(k))
            rng.shuffle(perm)
            S = []
            ok = True
            for i in range(k):
                u = perm[i]
                v = perm[(i + 1) % k]
                key = (u, v) if u <= v else (v, u)
                L = [o for o in E.get(key, []) if o not in S]
                if not L:
                    ok = False
                    break
                S.append(L[rng.randrange(len(L))])
            if ok:
                return S
        return None

    def complete(self, part):
        k = self.k
        rng = self.rng
        deg = [0] * k
        for o in part:
            for li in self.ends[o]:
                deg[li] += 1
        if any(d > 2 for d in deg):
            return None
        D = []
        for li in range(k):
            D += [li] * (2 - deg[li])
        for _ in range(60):
            rng.shuffle(D)
            used = set(part)
            add = []
            ok = True
            for i in range(0, len(D), 2):
                u, v = D[i], D[i + 1]
                key = (u, v) if u <= v else (v, u)
                L = [o for o in self.E.get(key, []) if o not in used]
                if not L:
                    ok = False
                    break
                o = L[rng.randrange(len(L))]
                used.add(o)
                add.append(o)
            if ok:
                return list(part) + add
        return None

    def perturb(self, S, m):
        S = list(S)
        rng = self.rng
        E = self.E
        ends = self.ends
        for _ in range(m):
            for _t in range(30):
                i, j = rng.sample(range(len(S)), 2)
                a, b = ends[S[i]]
                c, d = ends[S[j]]
                ch = rng.randrange(3)
                if ch == 0:
                    q1, q2 = (a, b), (c, d)
                elif ch == 1:
                    q1 = (min(a, c), max(a, c)); q2 = (min(b, d), max(b, d))
                else:
                    q1 = (min(a, d), max(a, d)); q2 = (min(b, c), max(b, c))
                ss = set(S)
                L1 = [o for o in E.get(q1, []) if o not in ss]
                L2 = [o for o in E.get(q2, []) if o not in ss]
                if not L1 or not L2:
                    continue
                y1 = L1[rng.randrange(len(L1))]
                L2 = [o for o in L2 if o != y1]
                if not L2:
                    continue
                y2 = L2[rng.randrange(len(L2))]
                S[i] = y1
                S[j] = y2
                break
        return S

    def run(self, start, deadline, maxstag):
        N = self.N
        rng = self.rng
        rnd = rng.random
        Bx = self.Bx
        Bin = self.Bin
        E = self.E
        ends = self.ends
        rp = self.rp
        vec = self.vec
        pc = _pc
        C = np.zeros(N, dtype=np.int64)
        Sl = []
        Sb = [0]
        cost = 0

        def add(c):
            v = vec([Bx[c], Bin[c]] + [rp(c, b) for b in Sl])
            np.add(C, v, out=C)
            Sl.append(c)
            Sb[0] |= 1 << c

        def rem(c):
            Sl.remove(c)
            Sb[0] &= ~(1 << c)
            v = vec([Bx[c], Bin[c]] + [rp(c, b) for b in Sl])
            np.subtract(C, v, out=C)

        for c in start:
            if Sb[0] >> c & 1:
                continue
            cost += int(C[c])
            add(c)
        Cl = C.tolist()
        best = cost
        bestS = list(Sl)
        tabu = [0] * N
        it = 0
        last = 0
        tm = time.time
        while True:
            S = Sb[0]
            conf = [x for x in Sl if Cl[x] > 0]
            if not conf:
                return list(Sl), 0, True
            if (it & 7) == 0 and tm() > deadline:
                break
            if it - last > maxstag:
                break
            it += 1
            x1 = conf[rng.randrange(len(conf))]
            e1 = ends[x1]
            a, b = e1
            bk = 1e18
            bm = None
            Cx1 = Cl[x1]
            for y in E[e1]:
                if S >> y & 1:
                    continue
                dl = Cl[y] - ((Bx[y] >> x1) & 1) - ((Bx[x1] >> y) & 1) - pc(rp(y, x1) & S) - Cx1
                if tabu[y] > it and cost + dl >= best:
                    continue
                key = dl + rnd() * 0.9
                if key < bk:
                    bk = key
                    bm = (None, y, None, dl)
            for x2 in Sl:
                if x2 == x1:
                    continue
                c, d = ends[x2]
                S0 = S & ~((1 << x1) | (1 << x2))
                remv = Cx1 + Cl[x2] - ((Bx[x1] >> x2) & 1) - ((Bx[x2] >> x1) & 1) - pc(rp(x1, x2) & S0)
                prs = [((a, b), (c, d))]
                p2 = ((a, c) if a <= c else (c, a), (b, d) if b <= d else (d, b))
                p3 = ((a, d) if a <= d else (d, a), (b, c) if b <= c else (c, b))
                for p in (p2, p3):
                    sp = tuple(sorted(p))
                    dup = False
                    for q in prs:
                        if tuple(sorted(q)) == sp:
                            dup = True
                            break
                    if not dup:
                        prs.append(p)
                cache = {}
                for (q1, q2) in prs:
                    L1 = E.get(q1)
                    L2 = E.get(q2)
                    if not L1 or not L2:
                        continue
                    for y1 in L1:
                        if S >> y1 & 1:
                            continue
                        v1 = cache.get(y1)
                        if v1 is None:
                            r1 = rp(y1, x1)
                            v1 = (Cl[y1] - ((Bx[y1] >> x1) & 1) - ((Bx[x1] >> y1) & 1)
                                  - ((Bx[y1] >> x2) & 1) - ((Bx[x2] >> y1) & 1)
                                  - pc(r1 & S) - pc(rp(y1, x2) & S) + ((r1 >> x2) & 1))
                            cache[y1] = v1
                        t1 = tabu[y1] > it
                        for y2 in L2:
                            if y2 == y1 or S >> y2 & 1:
                                continue
                            v2 = cache.get(y2)
                            if v2 is None:
                                r2 = rp(y2, x1)
                                v2 = (Cl[y2] - ((Bx[y2] >> x1) & 1) - ((Bx[x1] >> y2) & 1)
                                      - ((Bx[y2] >> x2) & 1) - ((Bx[x2] >> y2) & 1)
                                      - pc(r2 & S) - pc(rp(y2, x2) & S) + ((r2 >> x2) & 1))
                                cache[y2] = v2
                            dl = (v1 + v2 + ((Bx[y1] >> y2) & 1) + ((Bx[y2] >> y1) & 1)
                                  + pc(rp(y1, y2) & S0) - remv)
                            if (t1 or tabu[y2] > it) and cost + dl >= best:
                                continue
                            key = dl + rnd() * 0.9
                            if key < bk:
                                bk = key
                                bm = (x2, y1, y2, dl)
            if bm is None:
                continue
            x2, y1, y2, dl = bm
            ten = 6 + rng.randrange(8)
            rem(x1)
            tabu[x1] = it + ten
            if x2 is not None:
                rem(x2)
                tabu[x2] = it + ten
            add(y1)
            if y2 is not None:
                add(y2)
            cost += dl
            Cl = C.tolist()
            if cost < best:
                best = cost
                bestS = list(Sl)
                last = it
        return bestS, best, False


# ---------------------------------------------------------------- ILS fallback (odd n)
class _Model(object):
    def __init__(self, n, name):
        self.n = n
        self.name = name
        maps, orb_of, orbits = _build_orbits(n, name)
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


def _orbpts(M, sol):
    pts = []
    for o in sol:
        pts.extend(M.orbits[o])
    return [(int(x), int(y)) for (x, y) in pts]


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
    even = (n % 2 == 0)
    groups = ['rot4', 'dia2'] if even else ['rot2', 'dia2']
    models = []
    for g in groups:
        M = _DModel(n, g)
        if M.N > 0:
            models.append(M)
    M4 = models[0] if (even and models and models[0].name == 'rot4') else None
    bests = [[0, []] for _ in models]
    pool = []
    dfs_end = t0 + T * (0.25 if M4 is not None else 0.7)
    unit = 1000
    i = 1
    while models and time.time() < dfs_end:
        mi = (i - 1) % len(models)
        j = (i - 1) // len(models) + 1
        M = models[mi]
        noise = rng.choice([0.5, 2.0, 4.0, 8.0, 20.0])
        rbl = [0, []]
        sol = _dfs(M, _luby(j) * unit, rng, noise, dfs_end, rbl)
        if sol is not None:
            pts = _orbpts(M, sol)
            if len(pts) >= target:
                cp = _clean(pts)
                if len(cp) >= target:
                    return cp
            rbl = [len(pts), list(sol)]
        if rbl[0] > bests[mi][0]:
            bests[mi] = [rbl[0], list(rbl[1])]
        if M is M4 and rbl[1]:
            pool.append((rbl[0], list(rbl[1])))
            if len(pool) > 200:
                pool.sort(key=lambda t: -t[0])
                pool = pool[:40]
        i += 1
    pool.sort(key=lambda t: -t[0])
    pool = pool[:40]

    bestpts = []
    bmi = -1
    for mi, M in enumerate(models):
        pts = _orbpts(M, bests[mi][1])
        if len(pts) > len(bestpts):
            bestpts = pts
            bmi = mi

    end_all = t0 + T * 0.97
    if M4 is not None:
        ls = _LS(M4, rng)
        if ls.ok:
            gb = None
            gbc = 1 << 30
            maxstag = 1200
            while time.time() < end_all - 0.05:
                r = rng.random()
                start = None
                if pool and r < 0.5:
                    part = pool[rng.randrange(min(len(pool), 12))][1]
                    start = ls.complete(part)
                elif gb is not None and r < 0.75:
                    start = ls.perturb(gb, rng.randint(2, 5))
                if start is None:
                    start = ls.rand_cover()
                if start is None:
                    break
                S, c, solved = ls.run(start, end_all, maxstag)
                if solved:
                    cp = _clean(_orbpts(M4, S))
                    if len(cp) >= target:
                        return cp
                    if len(cp) > len(bestpts):
                        bestpts = cp
                if c < gbc:
                    gbc = c
                    gb = list(S)
            if gb is not None:
                cp = _clean(_orbpts(M4, gb))
                if len(cp) > len(bestpts):
                    bestpts = cp
        return _clean(bestpts)

    name = models[bmi].name if bmi >= 0 else ('rot4' if even else 'rot2')
    dl = end_all
    if time.time() < dl - 0.2:
        model = _Model(n, name)
        if model.N > 0:
            ps = set(bestpts)
            init = []
            seen = set()
            for (x, y) in bestpts:
                o = model.orb_of[x * n + y]
                if o >= 0 and o not in seen:
                    seen.add(o)
                    if all(c in ps for c in model.orbits[o]):
                        init.append(o)
            best, bp = _search(model, init, dl, target, rng, 7, 4, 2500, 0.3)
            pts = []
            for o in best:
                pts.extend(model.orbits[o])
            pts = [(int(x), int(y)) for (x, y) in pts]
            if len(pts) > len(bestpts):
                bestpts = pts
    return _clean(bestpts)
