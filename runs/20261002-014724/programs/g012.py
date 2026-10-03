# g012 (parent g010): Keep the rot4/dia2 orbit DFS for exact solutions, but replace the weak orbit and grid ILS with an asymmetric min-conflicts tabu search. It minimises collinear triples, repairs by greedy deletion, then refills.
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


def _pairs(Km):
    if Km.shape[1] < 2:
        return np.zeros(Km.shape[0], dtype=np.int64)
    S = np.sort(Km, axis=1)
    eq = S[:, 1:] == S[:, :-1]
    ix = np.arange(eq.shape[1])
    last = np.maximum.accumulate(np.where(eq, -1, ix), axis=1)
    return np.where(eq, ix - last, 0).sum(axis=1).astype(np.int64)


def _mcls(n, seed_cells, rng, deadline):
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
    BIG = 1 << 40
    twoN = 2 * n

    rows = [[] for _ in range(n)]
    colc = [0] * n
    for cl in seed_cells:
        cl = int(cl)
        r = cl // n
        if len(rows[r]) < 2 and cl not in rows[r]:
            rows[r].append(cl)
            colc[cl % n] += 1
    P = np.zeros(twoN, dtype=np.int64)
    for r in range(n):
        while len(rows[r]) < 2:
            ch = [cc for cc in range(n) if r * n + cc not in rows[r]]
            ch.sort(key=lambda cc: (colc[cc], rng.random()))
            cc = ch[0]
            rows[r].append(r * n + cc)
            colc[cc] += 1
        P[2 * r] = rows[r][0]
        P[2 * r + 1] = rows[r][1]

    def confs(Q):
        return _pairs(K[Q[:, None], Q[None, :]])

    def repair(Q):
        Q = Q.copy()
        while len(Q) > 2:
            cc = confs(Q)
            mx = int(cc.max())
            if mx == 0:
                break
            w = np.flatnonzero(cc == mx)
            j = int(w[rng.randrange(len(w))])
            Q = np.delete(Q, j)
        return Q

    def fillv(Q, dl):
        Q = np.array(Q, dtype=np.int64)
        while time.time() < dl:
            if len(Q) < 2:
                break
            pc = _pairs(K[:, Q])
            ok = pc == 0
            ok[Q] = False
            f = np.flatnonzero(ok)
            if f.size == 0:
                break
            Q = np.append(Q, int(f[rng.randrange(f.size)]))
        return Q

    tabu = np.zeros(N2, dtype=np.int64)
    cols = np.arange(n, dtype=np.int64)
    c = confs(P)
    T = int(c.sum()) // 3
    bestT = T
    bestP = P.copy()
    bestV = repair(P)
    it = 0
    last_imp = 0
    stag = 1500 + 20 * n
    while T > 0:
        if (it & 15) == 0 and time.time() > deadline:
            break
        it += 1
        rw = rng.random() < 0.03
        if rw:
            i = rng.randrange(twoN)
        else:
            cf = np.flatnonzero(c > 0)
            if cf.size == 0:
                break
            i = int(cf[rng.randrange(cf.size)])
        r = i // 2
        cur = int(P[i])
        oth = int(P[i ^ 1])
        base = r * n
        others = np.delete(P, i)
        cells = base + cols
        cz = _pairs(K[cells[:, None], others[None, :]])
        cz[oth - base] = BIG
        cz[cur - base] = BIG
        if rw:
            ch = np.flatnonzero(cz < BIG)
            if ch.size == 0:
                continue
            z = int(ch[rng.randrange(ch.size)])
        else:
            newT = T - int(c[i]) + cz
            tab = tabu[cells] > it
            cm = np.where(tab & (newT >= bestT), BIG, cz)
            mn = int(cm.min())
            if mn >= BIG:
                continue
            ch = np.flatnonzero(cm == mn)
            z = int(ch[rng.randrange(ch.size)])
        tabu[cur] = it + 3 + rng.randrange(9)
        P[i] = base + z
        c = confs(P)
        T = int(c.sum()) // 3
        if T < bestT:
            bestT = T
            bestP = P.copy()
            last_imp = it
            V = repair(P)
            if len(V) > len(bestV):
                bestV = V
        elif T == bestT and (it & 3) == 0:
            V = repair(P)
            if len(V) > len(bestV):
                bestV = V
        elif it - last_imp > stag:
            P = bestP.copy()
            for _ in range(2 + rng.randrange(4)):
                j = rng.randrange(twoN)
                rr = j // 2
                occ = (int(P[j]) % n, int(P[j ^ 1]) % n)
                ch = [cc for cc in range(n) if cc not in occ]
                P[j] = rr * n + rng.choice(ch)
            c = confs(P)
            T = int(c.sum()) // 3
            last_imp = it
    if T == 0:
        return [int(x) for x in P], True
    V = fillv(bestV, deadline + 0.4)
    return [int(x) for x in V], False


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
    end = t0 + T - 1.0
    t_dfs = t0 + 0.5 * T
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
    seed = list(best[1])
    if time.time() < end - 0.5:
        V, solved = _mcls(n, seed, rng, end)
        if solved:
            return V
        if len(V) >= len(seed):
            best[1] = V
            best[0] = len(V)
            return V
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
