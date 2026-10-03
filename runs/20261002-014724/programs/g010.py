# g010 (parent g007): Keep the rot4/dia2 orbit DFS for exact solutions. Add an ARW-style orbit iterated local search, then a numpy full-grid tabu ILS to polish partials.
import time
import random
from math import gcd
import numpy as np

try:
    _pc = int.bit_count
    _pc(3)
except Exception:
    def _pc(x):
        return bin(x).count('1')


class _Abort(Exception):
    pass


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
    ALL = (1 << norb) - 1
    return {'M': M, 'selfhit': selfhit, 'contrib': contrib,
            'size': [len(x) for x in cellsof], 'cells': cellsof,
            'umask': umask, 'c2': c2, 'H': H, 'free0': ALL & ~invalid,
            'sym': sym}


def _dfs(D, rng, budget, deadline, best, noise):
    M = D['M']
    selfhit = D['selfhit']
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

    def rec(free, pts):
        cnt[0] += 1
        if cnt[0] > budget:
            raise _Abort()
        if not (cnt[0] & 511) and tm() > deadline:
            raise _Abort()
        if pts > best[0]:
            best[0] = pts
            fl = []
            for o in used:
                fl.extend(cells[o])
            best[1] = fl
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
            G = Mo[O] | selfhit[O] | l
            for U in used:
                G |= Mo[U]
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


def _oils(D, rng, deadline, target):
    M = D['M']
    selfhit = D['selfhit']
    contrib = D['contrib']
    size = D['size']
    umask = D['umask']
    c2 = D['c2']
    H = D['H']
    free0 = D['free0']
    norb = len(size)
    pc = _pc
    own = [selfhit[X] | M[X][X] | (1 << X) for X in range(norb)]
    valid = _bits(free0)
    if not valid:
        return [], 0

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
                g = own[Z]
                for X in S:
                    g |= MZ[X]
            else:
                bv = -1
                Z = -1
                g = 0
                for Z0 in rng.sample(bl, 3):
                    MZ = M[Z0]
                    g0 = own[Z0]
                    for X in S:
                        g0 |= MZ[X]
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
    best = cur
    bestS = list(S)
    tabu = {}
    it = 0
    while cur < target and time.time() < deadline:
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
        if cur > best:
            best = cur
            bestS = list(S)
        elif cur < savecur and rng.random() < 0.85:
            S = saveS
            cur = savecur
    return bestS, best


def _gils(n, seed_pts, rng, deadline):
    N2 = n * n
    idx = np.arange(N2, dtype=np.int16)
    R = idx // n
    C = idx % n
    DX = R[None, :] - R[:, None]
    DY = C[None, :] - C[:, None]
    G = np.gcd(DX, DY)
    G[G == 0] = 1
    A = DX // G
    B = DY // G
    del DX, DY, G
    neg = (A < 0) | ((A == 0) & (B < 0))
    A = np.where(neg, -A, A)
    B = np.where(neg, -B, B)
    del neg
    W = 2 * n - 1
    K = (A.astype(np.int32) * W + B.astype(np.int32) + (n - 1)).astype(np.int32)
    del A, B
    dummy = n * W
    np.fill_diagonal(K, dummy)
    ND = dummy + 1
    Ri = R.astype(np.int64)
    Ci = C.astype(np.int64)
    occ = np.zeros(N2, dtype=bool)
    block = np.zeros(N2, dtype=np.int64)
    rowc = np.zeros(n, dtype=np.int64)
    colc = np.zeros(n, dtype=np.int64)
    tabu_until = np.zeros(N2, dtype=np.int64)
    pts = []

    def add(p):
        if pts:
            cnt = np.bincount(K[p, pts], minlength=ND)
            cnt[dummy] = 0
            block += cnt[K[p]] if False else 0
            np.add(block, cnt[K[p]], out=block)
        pts.append(p)
        occ[p] = True
        rowc[p // n] += 1
        colc[p % n] += 1

    def remove(p):
        pts.remove(p)
        occ[p] = False
        rowc[p // n] -= 1
        colc[p % n] -= 1
        if pts:
            cnt = np.bincount(K[p, pts], minlength=ND)
            cnt[dummy] = 0
            np.subtract(block, cnt[K[p]], out=block)
            block[p] = int((cnt * (cnt - 1)).sum() // 2)
        else:
            block[p] = 0

    def rebuild(P, safe=False):
        block[:] = 0
        occ[:] = False
        rowc[:] = 0
        colc[:] = 0
        del pts[:]
        for p in P:
            if safe:
                if occ[p] or block[p] != 0 or rowc[p // n] >= 2 or colc[p % n] >= 2:
                    continue
            add(p)

    def fill(it):
        while True:
            m = (block == 0) & ~occ & (rowc < 2)[Ri] & (colc < 2)[Ci] & (tabu_until <= it)
            f = np.flatnonzero(m)
            if f.size == 0:
                break
            add(int(f[rng.randrange(f.size)]))

    rebuild(list(seed_pts), safe=True)
    fill(0)
    best = list(pts)
    target = 2 * n
    it = 0
    BIG = 1 << 40
    while len(best) < target and time.time() < deadline:
        it += 1
        sc = block + (rowc >= 2)[Ri] + (colc >= 2)[Ci]
        sc[occ] = BIG
        sc[tabu_until > it] = BIG
        mn = int(sc.min())
        if mn >= BIG:
            continue
        thr = mn + (1 if rng.random() < 0.3 else 0)
        f = np.flatnonzero(sc <= thr)
        z = int(f[rng.randrange(f.size)])
        P = list(pts)
        rem = set()
        if P:
            kz = K[z, P]
            cnt = np.bincount(kz, minlength=ND)
            for d in np.flatnonzero(cnt >= 2):
                line = [P[i] for i in np.flatnonzero(kz == d)]
                rng.shuffle(line)
                rem.update(line[1:])
        zr, zc = divmod(z, n)
        rp = [q for q in P if q // n == zr and q not in rem]
        if len(rp) >= 2:
            rem.add(rng.choice(rp))
        cp = [q for q in P if q % n == zc and q not in rem]
        if len(cp) >= 2:
            rem.add(rng.choice(cp))
        savecur = len(P)
        for q in rem:
            remove(q)
            tabu_until[q] = it + rng.randint(3, 12)
        add(z)
        fill(it)
        cur = len(pts)
        if cur > len(best):
            best = list(pts)
        elif cur < savecur and rng.random() < 0.9:
            rebuild(P)
    return best


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


def _solve(n, t0, T, rng, best):
    end = t0 + T - 0.7
    t_dfs = t0 + 0.45 * T
    t_o = t0 + 0.72 * T
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
            r = _dfs(D, rng, budget, t_dfs, best, noise)
            if r is not None:
                return [q for o in r for q in D['cells'][o]]
            active.remove(D)
        except _Abort:
            pass
    cand_lists = [best[1]]
    for j, D in enumerate(classes):
        now = time.time()
        remaining = t_o - now
        if remaining <= 0.2:
            break
        frac = 0.6 if (j == 0 and len(classes) > 1) else 1.0
        S, p = _oils(D, rng, now + remaining * frac, 2 * n)
        cells = [q for o in S for q in D['cells'][o]]
        if p >= 2 * n:
            return cells
        cand_lists.append(cells)
    seed = max(cand_lists, key=len)
    best[1] = seed
    best[0] = len(seed)
    if time.time() < end - 0.3:
        g = _gils(n, seed, rng, end)
        if len(g) >= len(seed):
            return g
    return seed


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
    best = [0, []]
    res = None
    if n <= 64:
        try:
            res = _solve(n, t0, time_limit, rng, best)
        except Exception:
            res = None
    if res is None:
        base = best[1] if best[1] else []
        try:
            res = _fill(n, base, rng, t0 + time_limit - 0.25)
        except Exception:
            res = base
    seen = set()
    out = []
    for c in res:
        c = int(c)
        if c in seen:
            continue
        seen.add(c)
        out.append((c // n, c % n))
    return out
