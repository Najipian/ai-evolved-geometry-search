# g011 (parent g010): Spend half the time on a faster rot4/dia2 orbit DFS, then orbit ILS; exactly complete near-full partials asymmetrically on deficit row/column cells.
# Orbit DFS (rot4/dia2) with reduce-based conflict ORs, orbit ILS, and exact asymmetric completion of near-full partials.
import time
import random
from math import gcd
from functools import reduce
from operator import or_
import numpy as np

try:
    _pc = int.bit_count
    _pc(3)
except Exception:
    def _pc(x):
        return bin(x).count('1')


class _Abort(Exception):
    pass


class _Found(Exception):
    def __init__(self, pts):
        Exception.__init__(self)
        self.pts = pts


def _maps(sym, n):
    m = n - 1
    r4 = [lambda r, c: (r, c), lambda r, c: (c, m - r),
          lambda r, c: (m - r, m - c), lambda r, c: (m - c, r)]
    refl = [lambda r, c: (c, r), lambda r, c: (m - c, m - r),
            lambda r, c: (m - r, c), lambda r, c: (r, m - c)]
    if sym == 'rot4':
        return r4
    if sym == 'dia2':
        return [r4[0], refl[0], refl[1], r4[2]]
    return r4 + refl


def _build(n, sym):
    maps = _maps(sym, n)
    N2 = n * n
    orb = [-1] * N2
    cellsof = []
    for p in range(N2):
        if orb[p] >= 0:
            continue
        r, c = divmod(p, n)
        s = set()
        for f in maps:
            a, b = f(r, c)
            s.add(a * n + b)
        k = len(cellsof)
        for q in s:
            orb[q] = k
        cellsof.append(sorted(s))
    norb = len(cellsof)

    def uimg(f, u):
        if u < n:
            a = f(u, 0)
            b = f(u, 1)
        else:
            a = f(0, u - n)
            b = f(1, u - n)
        return a[0] if a[0] == b[0] else n + a[1]

    ucl = [-1] * (2 * n)
    reps = []
    for u in range(2 * n):
        if ucl[u] >= 0:
            continue
        k = len(reps)
        reps.append(u)
        for f in maps:
            ucl[uimg(f, u)] = k
    H = len(reps)
    contrib = []
    invalid = 0
    for o in range(norb):
        cnt = {}
        for q in cellsof[o]:
            r, c = divmod(q, n)
            k = ucl[r]
            if reps[k] == r:
                cnt[k] = cnt.get(k, 0) + 1
            k = ucl[n + c]
            if reps[k] == n + c:
                cnt[k] = cnt.get(k, 0) + 1
        lst = list(cnt.items())
        if any(v > 2 for _, v in lst):
            invalid |= 1 << o
        contrib.append(lst)
    umask = [0] * H
    c2 = [0] * H
    for o, lst in enumerate(contrib):
        for k, v in lst:
            umask[k] |= 1 << o
            if v >= 2:
                c2[k] |= 1 << o
    S = max(len(x) for x in cellsof)
    Q = np.full((norb, S), -1, dtype=np.int64)
    for o in range(norb):
        for j, q in enumerate(cellsof[o]):
            Q[o, j] = q
    Vm = (Q >= 0)
    Vm8 = Vm.astype(np.int8)
    orbarr = np.array(orb, dtype=np.int64)
    idx = np.arange(N2, dtype=np.int64)
    R = idx // n
    Cc = idx % n
    M = [[0] * norb for _ in range(norb)]
    jj = np.arange(S)[None, :]
    for o1 in range(norb):
        p1 = cellsof[o1][0]
        r1, c1 = divmod(p1, n)
        DX = R - r1
        DY = Cc - c1
        G = np.gcd(DX, DY)
        G[p1] = 1
        A = DX // G
        B = DY // G
        neg = (A < 0) | ((A == 0) & (B < 0))
        A = np.where(neg, -A, A)
        B = np.where(neg, -B, B)
        key = A * (2 * n + 1) + (B + n)
        key[p1] = -1
        uk, kidx = np.unique(key, return_inverse=True)
        kidx = kidx.ravel()
        nk = len(uk)
        Cnt = np.bincount(kidx * norb + orbarr, minlength=nk * norb).astype(np.int8).reshape(nk, norb)
        Cnt[kidx[p1]] = 0
        sub = Q[o1:]
        m = sub.shape[0]
        Qc = np.where(sub >= 0, sub, p1)
        kq = kidx[Qc]
        Rr = Cnt[kq]
        ii = np.arange(m)[:, None]
        kk = (o1 + np.arange(m))[:, None]
        Rr[ii, jj, kk] -= Vm8[o1:]
        Rr[0, 0, o1] += 1
        Bm = (Rr > 0).any(axis=1)
        pk = np.packbits(Bm, axis=1, bitorder='little')
        w = pk.shape[1]
        bs = pk.tobytes()
        row = M[o1]
        for t in range(m):
            v = int.from_bytes(bs[t * w:(t + 1) * w], 'little')
            o2 = o1 + t
            row[o2] = v
            M[o2][o1] = v
    selfhit = [0] * norb
    for X in range(norb):
        mm = M[X][X]
        if (mm >> X) & 1:
            invalid |= 1 << X
        while mm:
            l = mm & -mm
            b = l.bit_length() - 1
            mm ^= l
            selfhit[b] |= 1 << X
    own = [selfhit[X] | M[X][X] | (1 << X) for X in range(norb)]
    ALL = (1 << norb) - 1
    return {'M': M, 'selfhit': selfhit, 'contrib': contrib, 'own': own,
            'size': [len(x) for x in cellsof], 'cells': cellsof,
            'umask': umask, 'c2': c2, 'H': H, 'free0': ALL & ~invalid,
            'sym': sym}


def _complete(n, pts, node_cap=20000):
    P = [int(p) for p in pts]
    occ = set(P)
    rc = [0] * n
    cc = [0] * n
    for p in P:
        rc[p // n] += 1
        cc[p % n] += 1
    rows = [r for r in range(n) if rc[r] < 2]
    cols = [c for c in range(n) if cc[c] < 2]
    if not rows or not cols or len(rows) * len(cols) > 900:
        return P
    coords = [divmod(p, n) for p in P]
    cands = []
    dirs = {}
    for r in rows:
        for c in cols:
            z = r * n + c
            if z in occ:
                continue
            s = set()
            ok = True
            for a, b in coords:
                dx = a - r
                dy = b - c
                g = gcd(dx, dy)
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                k = dx * 4096 + dy
                if k in s:
                    ok = False
                    break
                s.add(k)
            if ok:
                cands.append(z)
                dirs[z] = s
    if not cands:
        return P
    ub = min(sum(2 - rc[r] for r in rows), sum(2 - cc[c] for c in cols))
    best = [[]]
    chosen = []
    cnt = [0]
    L = len(cands)

    def rec(i):
        cnt[0] += 1
        if len(chosen) > len(best[0]):
            best[0] = list(chosen)
        if len(best[0]) >= ub:
            return True
        if cnt[0] > node_cap:
            return True
        if len(chosen) + (L - i) <= len(best[0]):
            return False
        for j in range(i, L):
            z = cands[j]
            r, c = divmod(z, n)
            if rc[r] >= 2 or cc[c] >= 2:
                continue
            dz = dirs[z]
            ok = True
            ds = set()
            for y in chosen:
                yr, yc = divmod(y, n)
                dx = yr - r
                dy = yc - c
                g = gcd(dx, dy)
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                k = dx * 4096 + dy
                if k in dz or k in ds:
                    ok = False
                    break
                ds.add(k)
            if not ok:
                continue
            chosen.append(z)
            rc[r] += 1
            cc[c] += 1
            stop = rec(j + 1)
            chosen.pop()
            rc[r] -= 1
            cc[c] -= 1
            if stop:
                return True
        return False

    rec(0)
    return P + best[0]


def _dfs(D, rng, budget, deadline, noise, hooks, target, gb):
    consider, try_complete = hooks
    M = D['M']
    own = D['own']
    contrib = D['contrib']
    size = D['size']
    cells = D['cells']
    umask = D['umask']
    c2 = D['c2']
    H = D['H']
    deficit = [2] * H
    used = []
    cnt = [0]
    rnd = rng.random
    pc = _pc
    tm = time.time
    cthr = target - 4

    def rec(free, pts):
        cnt[0] += 1
        if cnt[0] > budget:
            raise _Abort()
        if not (cnt[0] & 511) and tm() > deadline:
            raise _Abort()
        if pts > gb[0] or pts >= cthr:
            fl = [q for o in used for q in cells[o]]
            if pts > gb[0]:
                consider(fl)
            if pts >= cthr:
                try_complete(fl)
        bu = -1
        bc = 1 << 30
        ba = 0
        for u in range(H):
            d = deficit[u]
            if d:
                a = umask[u] & free
                if not a:
                    return None
                c = pc(a)
                if d == 2:
                    if c == 1 and not (a & c2[u]):
                        return None
                    c = 2 * c - 1
                else:
                    c = 2 * c
                if c < bc or (c == bc and rnd() < 0.5):
                    bc = c
                    bu = u
                    ba = a
        if bu < 0:
            return list(used)
        cands = []
        a = ba
        while a:
            l = a & -a
            O = l.bit_length() - 1
            a ^= l
            Mo = M[O]
            G = reduce(or_, map(Mo.__getitem__, used), own[O])
            nf = free & ~G
            for (u, k) in contrib[O]:
                dn = deficit[u] - k
                if dn == 0:
                    nf &= ~umask[u]
                elif dn == 1:
                    nf &= ~c2[u]
            cands.append((pc(nf) + rnd() * noise, O, nf))
        cands.sort(reverse=True)
        excl = 0
        for sc, O, nf in cands:
            cl = contrib[O]
            for (u, k) in cl:
                deficit[u] -= k
            used.append(O)
            r = rec(nf & ~excl, pts + size[O])
            used.pop()
            for (u, k) in cl:
                deficit[u] += k
            if r is not None:
                return r
            excl |= 1 << O
        return None

    return rec(D['free0'], 0)


def _luby(i):
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i -= (1 << (k - 1)) - 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1


def _bits(x):
    out = []
    while x:
        l = x & -x
        out.append(l.bit_length() - 1)
        x ^= l
    return out


def _oils(D, rng, deadline, target, hooks, gb):
    consider, try_complete = hooks
    M = D['M']
    selfhit = D['selfhit']
    contrib = D['contrib']
    size = D['size']
    umask = D['umask']
    c2 = D['c2']
    H = D['H']
    cells = D['cells']
    free0 = D['free0']
    own = D['own']
    pc = _pc
    valid = _bits(free0)
    if not valid:
        return
    cache = set()

    def hook(S, cur):
        if cur > gb[0] or cur >= target - 8:
            fs = frozenset(S)
            if fs in cache:
                return
            if len(cache) > 30000:
                cache.clear()
            cache.add(fs)
            fl = [q for o in S for q in cells[o]]
            if cur > gb[0]:
                consider(fl)
            if cur >= target - 8:
                try_complete(fl)

    def capm(dfc):
        cm = free0
        for u in range(H):
            d = dfc[u]
            if d <= 0:
                cm &= ~umask[u]
            elif d == 1:
                cm &= ~c2[u]
        return cm

    def forb_of(S):
        f = 0
        for i, X in enumerate(S):
            f |= own[X]
            MX = M[X]
            for j in range(i):
                f |= MX[S[j]]
        return f

    def dfc_of(S):
        d = [2] * H
        for X in S:
            for u, k in contrib[X]:
                d[u] -= k
        return d

    def fill(S, dfc, forb, tabu):
        while True:
            fr = capm(dfc) & ~forb & ~tabu
            if not fr:
                return forb
            bl = _bits(fr)
            if len(bl) <= 2 or rng.random() < 0.3:
                Z = rng.choice(bl)
                MZ = M[Z]
                g = reduce(or_, map(MZ.__getitem__, S), own[Z])
            else:
                bv = -1
                Z = -1
                g = 0
                for Z0 in rng.sample(bl, 3):
                    MZ = M[Z0]
                    g0 = reduce(or_, map(MZ.__getitem__, S), own[Z0])
                    v = pc(fr & ~g0)
                    if v > bv:
                        bv = v
                        Z = Z0
                        g = g0
            forb |= g
            S.append(Z)
            for u, k in contrib[Z]:
                dfc[u] -= k

    def removal(Z, S, Sb, dfc):
        rem = (selfhit[Z] | M[Z][Z]) & Sb
        MZ = M[Z]
        order = list(S)
        rng.shuffle(order)
        for X in order:
            if (rem >> X) & 1:
                continue
            m = MZ[X] & Sb & ~rem & ~(1 << X)
            while m:
                if rng.random() < 0.5:
                    rem |= 1 << X
                    break
                rem |= m & -m
                m = MZ[X] & Sb & ~rem & ~(1 << X)
        d = list(dfc)
        for X in _bits(rem):
            for u, k in contrib[X]:
                d[u] += k
        for u, k in contrib[Z]:
            d[u] -= k
        for u, k in contrib[Z]:
            if d[u] < 0:
                for X in order:
                    if d[u] >= 0:
                        break
                    if (rem >> X) & 1:
                        continue
                    hit = False
                    for uu, kk in contrib[X]:
                        if uu == u:
                            hit = True
                    if hit:
                        rem |= 1 << X
                        for uu, kk in contrib[X]:
                            d[uu] += kk
        return rem

    S = []
    dfc = [2] * H
    fill(S, dfc, 0, 0)
    cur = sum(size[X] for X in S)
    hook(S, cur)
    tabu = {}
    it = 0
    while time.time() < deadline:
        it += 1
        tmask = 0
        for X in list(tabu.keys()):
            if tabu[X] <= it:
                del tabu[X]
            else:
                tmask |= 1 << X
        Sb = 0
        for X in S:
            Sb |= 1 << X
        dfc = dfc_of(S)
        bz = -1
        brem = 0
        bc = 1e9
        for _ in range(5):
            Z = valid[rng.randrange(len(valid))]
            if (Sb >> Z) & 1 or (tmask >> Z) & 1:
                continue
            rem = removal(Z, S, Sb, dfc)
            c = sum(size[X] for X in _bits(rem)) - size[Z] + rng.random()
            if c < bc:
                bc = c
                bz = Z
                brem = rem
        if bz < 0:
            continue
        saveS = list(S)
        savecur = cur
        rl = _bits(brem)
        for X in rl:
            tabu[X] = it + rng.randint(3, 10)
        S = [X for X in S if not (brem >> X) & 1]
        S.append(bz)
        dfc = dfc_of(S)
        forb = forb_of(S)
        fill(S, dfc, forb, tmask | brem)
        cur = sum(size[X] for X in S)
        hook(S, cur)
        if cur < savecur and rng.random() < 0.85:
            S = saveS
            cur = savecur


def _fill(n, pts, rng, deadline):
    pts = list(pts)
    occ = set(pts)
    cand = [p for p in range(n * n) if p not in occ]
    rng.shuffle(cand)
    rc = [0] * n
    cc = [0] * n
    for p in pts:
        rc[p // n] += 1
        cc[p % n] += 1
    for z in cand:
        if time.time() > deadline:
            break
        zr, zc = divmod(z, n)
        if rc[zr] >= 2 or cc[zc] >= 2:
            continue
        seen = set()
        ok = True
        for q in pts:
            dx = q // n - zr
            dy = q % n - zc
            g = gcd(dx, dy)
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            kk = dx * 4096 + dy
            if kk in seen:
                ok = False
                break
            seen.add(kk)
        if ok:
            pts.append(z)
            rc[zr] += 1
            cc[zc] += 1
    return pts


def _solve(n, t0, T, rng, gb):
    end = t0 + T - 0.6
    t_dfs = t0 + 0.5 * T
    target = 2 * n
    ct = [0.0]

    def consider(cells):
        if len(cells) > gb[0]:
            gb[0] = len(cells)
            gb[1] = list(cells)
        if len(cells) >= target:
            raise _Found(list(cells))

    def try_complete(cells):
        now = time.time()
        if ct[0] > 0.25 * (now - t0) + 0.1:
            return
        res = _complete(n, cells)
        ct[0] += time.time() - now
        consider(res)

    hooks = (consider, try_complete)
    try:
        classes = []
        for sym in ('rot4', 'dia2'):
            if time.time() > t_dfs:
                break
            classes.append(_build(n, sym))
        active = list(classes)
        i = 0
        base = 150
        while active and time.time() < t_dfs:
            D = active[i % len(active)]
            budget = base * _luby(i // len(active) + 1)
            noise = rng.choice([1.0, 3.0, 6.0, 12.0])
            i += 1
            try:
                r = _dfs(D, rng, budget, t_dfs, noise, hooks, target, gb)
                if r is not None:
                    return [q for o in r for q in D['cells'][o]]
                active.remove(D)
            except _Abort:
                pass
        for j, D in enumerate(classes):
            now = time.time()
            remaining = end - now
            if remaining <= 0.2:
                break
            frac = 0.5 if (j == 0 and len(classes) > 1) else 1.0
            _oils(D, rng, now + remaining * frac, target, hooks, gb)
    except _Found as e:
        return e.pts
    return gb[1]


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
    gb = [0, []]
    res = None
    if n <= 64:
        try:
            res = _solve(n, t0, time_limit, rng, gb)
        except Exception:
            res = None
    if res is None or len(res) < gb[0]:
        res = gb[1] if gb[1] else []
    if len(res) < 2 * n and time.time() < t0 + time_limit - 0.3:
        try:
            res = _fill(n, res, rng, t0 + time_limit - 0.25)
        except Exception:
            pass
    seen = set()
    out = []
    for c in res:
        c = int(c)
        if c in seen:
            continue
        seen.add(c)
        out.append((c // n, c % n))
    return out
