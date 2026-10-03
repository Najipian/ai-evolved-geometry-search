# g008 (parent g004): Run an ILS on rot4 and dia2 orbits, tracking exact blocking counts and involvement per orbit. It uses exhaustive (1,2)-swaps, plateau swaps, forced inserts and kicks, round-robin, then asymmetric polish.
import time
import random
from math import gcd
import numpy as np


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
        self.size = [len(c) for c in orbits]
        contrib = []
        for oid, cells in enumerate(orbits):
            cnt = {}
            for (a, b) in cells:
                cnt[a] = cnt.get(a, 0) + 1
                cnt[n + b] = cnt.get(n + b, 0) + 1
            contrib.append(list(cnt.items()))
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
        self.baseidx = [np.array(_bits(self.base[a] & ~(1 << a)), dtype=np.intp) for a in range(N)]
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
        reps = sorted(set(find(i) for i in range(2 * n)))
        repidx = dict((r, i) for i, r in enumerate(reps))
        self.L = len(reps)
        U = np.zeros((max(N, 1), self.L), np.int32)
        for oid in range(N):
            for idx, k in contrib[oid]:
                j = repidx.get(idx)
                if j is not None:
                    U[oid, j] = k
        self.U = U
        self.pm = {}
        self.pi = {}

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

    def pmask(self, a, b):
        if a > b:
            a, b = b, a
        key = a * self.N + b
        v = self.pm.get(key)
        if v is None:
            p = self.orbits[a][0]
            v = 0
            for q in self.orbits[b]:
                v |= self.line(p, q)
            v &= ~((1 << a) | (1 << b))
            self.pm[key] = v
        return v

    def pidx(self, a, b):
        if a > b:
            a, b = b, a
        key = a * self.N + b
        v = self.pi.get(key)
        if v is None:
            v = np.array(_bits(self.pmask(a, b)), dtype=np.intp)
            self.pi[key] = v
        return v


class _State(object):
    def __init__(self, M):
        self.M = M
        N = M.N
        self.cnt = np.zeros(N, np.int32)
        self.inv = np.zeros((N, N), np.int32)
        self.lc = np.zeros(M.L, np.int32)
        self.inS = np.zeros(N, bool)
        self.S = []
        self.pts = 0

    def clear(self):
        self.cnt[:] = 0
        self.inv[:] = 0
        self.lc[:] = 0
        self.inS[:] = False
        self.S = []
        self.pts = 0

    def load(self, S):
        self.clear()
        for a in S:
            self.add(a)

    def add(self, a):
        M = self.M
        cnt, inv = self.cnt, self.inv
        bi = M.baseidx[a]
        if bi.size:
            cnt[bi] += 1
            inv[bi, a] += 1
        for b in self.S:
            idx = M.pidx(a, b)
            if idx.size:
                cnt[idx] += 1
                inv[idx, a] += 1
                inv[idx, b] += 1
        self.lc += M.U[a]
        self.inS[a] = True
        self.S.append(a)
        self.pts += M.size[a]

    def remove(self, a):
        M = self.M
        cnt, inv = self.cnt, self.inv
        self.S.remove(a)
        self.inS[a] = False
        for b in self.S:
            idx = M.pidx(a, b)
            if idx.size:
                cnt[idx] -= 1
                inv[idx, a] -= 1
                inv[idx, b] -= 1
        bi = M.baseidx[a]
        if bi.size:
            cnt[bi] -= 1
            inv[bi, a] -= 1
        self.lc -= M.U[a]
        self.pts -= M.size[a]

    def free(self):
        ok = (~self.inS) & (self.cnt == 0)
        cand = np.flatnonzero(ok)
        if not cand.size:
            return cand
        cap = ((self.lc[None, :] + self.M.U[cand]) <= 2).all(1)
        return cand[cap]

    def fill(self, rng):
        while True:
            f = self.free()
            if not f.size:
                return
            self.add(int(f[rng.randrange(f.size)]))

    def improve(self, rng):
        S = self.S
        k = len(S)
        if k == 0:
            return False, None, None
        M = self.M
        Sarr = np.array(S, np.intp)
        cnt = self.cnt
        TF = self.inv[:, Sarr] == cnt[:, None]
        TF &= (~self.inS)[:, None]
        rows = np.flatnonzero(TF.any(1))
        if not rows.size:
            return False, None, None
        U = M.U
        rem = self.lc[None, :] - U[Sarr]
        CF = ((rem[None, :, :] + U[rows][:, None, :]) <= 2).all(2)
        OK = TF[rows] & CF
        colsum = OK.sum(0)
        order = list(range(k))
        rng.shuffle(order)
        size = M.size
        base = M.base
        pmask = M.pmask
        for j in order:
            if colsum[j] == 0:
                continue
            a = S[j]
            F = rows[np.flatnonzero(OK[:, j])].tolist()
            sa = size[a]
            for c in F:
                if size[c] > sa:
                    self.remove(a)
                    self.add(c)
                    return True, None, None
            if len(F) < 2:
                continue
            others = [x for x in S if x != a]
            rj = rem[j]
            rng.shuffle(F)
            for i in range(len(F) - 1):
                c = F[i]
                mc = base[c]
                for x in others:
                    mc |= pmask(c, x)
                rc = rj + U[c]
                sc = size[c]
                for d in F[i + 1:]:
                    if (mc >> d) & 1:
                        continue
                    if sc + size[d] <= sa:
                        continue
                    if ((rc + U[d]) <= 2).all():
                        self.remove(a)
                        self.add(c)
                        self.add(d)
                        return True, None, None
        return False, OK, (rows, Sarr)

    def force(self, c, rng):
        removed = []
        inv = self.inv
        while self.cnt[c] > 0:
            Sarr = np.array(self.S, np.intp)
            w = inv[c, Sarr]
            mx = w.max()
            idx = np.flatnonzero(w == mx)
            a = int(Sarr[idx[rng.randrange(idx.size)]])
            self.remove(a)
            removed.append(a)
        U = self.M.U
        while True:
            over = np.flatnonzero(self.lc + U[c] > 2)
            if not over.size:
                break
            l = int(over[0])
            occ = [x for x in self.S if U[x, l] > 0]
            if not occ:
                break
            a = occ[rng.randrange(len(occ))]
            self.remove(a)
            removed.append(a)
        self.add(c)
        return removed


def _new_group(n, name):
    M = _Model(n, name)
    g = {'M': M, 'st': _State(M) if M.N else None, 'bp': 0, 'bS': [],
         'it': 0, 'since': 0, 'tabu': np.zeros(max(M.N, 1), np.int64),
         'R': 500, 'thr': 4}
    return g


def _ils(g, deadline, rng, target):
    M = g['M']
    st = g['st']
    tabu = g['tabu']
    tm = time.time
    N = M.N
    while tm() < deadline:
        OK = None
        info = None
        while True:
            st.fill(rng)
            if st.pts >= g['bp']:
                if st.pts > g['bp']:
                    g['since'] = 0
                g['bp'] = st.pts
                g['bS'] = list(st.S)
                if st.pts >= target:
                    return True
            imp, OK, info = st.improve(rng)
            if not imp:
                break
        g['it'] += 1
        it = g['it']
        g['since'] += 1
        if g['since'] > g['R']:
            st.load(g['bS'])
            if st.S:
                frac = 0.15 + 0.2 * rng.random()
                kk = max(1, int(len(st.S) * frac))
                for a in rng.sample(list(st.S), kk):
                    st.remove(a)
                    tabu[a] = it + 10
            g['since'] = 0
            continue
        if st.pts < g['bp'] - g['thr']:
            st.load(g['bS'])
            continue
        ten = 4 + rng.randrange(8)
        if OK is not None and rng.random() < 0.5:
            rows, Sarr = info
            ci, ji = np.nonzero(OK)
            if ci.size:
                cs = rows[ci]
                good = tabu[cs] <= it
                if good.any():
                    cs = cs[good]
                    ji = ji[good]
                r = rng.randrange(cs.size)
                c = int(cs[r])
                a = int(Sarr[ji[r]])
                st.remove(a)
                st.add(c)
                tabu[a] = it + ten
                continue
        bestc = -1
        bv = None
        for _ in range(5):
            c = rng.randrange(N)
            if st.inS[c] or tabu[c] > it:
                continue
            v = st.cnt[c] + rng.random()
            if bv is None or v < bv:
                bv = v
                bestc = c
        if bestc < 0:
            continue
        removed = st.force(bestc, rng)
        for a in removed:
            tabu[a] = it + ten
    return False


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


def _to_pts(M, S):
    pts = []
    for o in S:
        pts.extend(M.orbits[o])
    return [(int(x), int(y)) for (x, y) in pts]


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit - 0.45
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
        names = ['rot4', 'dia2']
    else:
        names = ['rot2', 'dia1']
    if n <= 10:
        names.append('iden')
    total = max(0.5, deadline - t0)
    sym_end = t0 + 0.9 * total
    chunk = max(0.5, min(2.5, 0.12 * time_limit))
    target = 2 * n
    groups = {}
    gi = 0
    empty_streak = 0
    while time.time() < sym_end:
        nm = names[gi % len(names)]
        gi += 1
        if nm not in groups:
            try:
                groups[nm] = _new_group(n, nm)
            except Exception:
                groups[nm] = None
        g = groups[nm]
        if g is None or g['st'] is None:
            empty_streak += 1
            if empty_streak > len(names):
                break
            continue
        empty_streak = 0
        end = min(sym_end, time.time() + chunk)
        try:
            if _ils(g, end, rng, target):
                return _to_pts(g['M'], g['bS'])
        except Exception:
            pass
    bestg = None
    for g in groups.values():
        if g is None or g['st'] is None:
            continue
        if bestg is None or g['bp'] > bestg['bp']:
            bestg = g
    init = _to_pts(bestg['M'], bestg['bS']) if bestg is not None else []
    try:
        pts = _local_search(n, init, deadline, rng)
    except Exception:
        pts = init
    if len(pts) < len(init):
        pts = init
    return [(int(x), int(y)) for (x, y) in pts]
