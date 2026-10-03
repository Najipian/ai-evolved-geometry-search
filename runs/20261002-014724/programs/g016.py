# g016 (parent g012): Keep the rot4/dia2 orbit DFS that solves n=32. Replace min-conflicts tabu with full-grid LNS: destroy rows/columns, plus blockers of empty cells, then refill exactly by branch-and-bound.
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


class _Done(Exception):
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


def _keys(n):
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
    K = (A.astype(np.int32) * (2 * n + 1) + B.astype(np.int32) + n).astype(np.int32)
    del A, B
    np.fill_diagonal(K, -1)
    return K


def _fixvalid(n, K, S, rng):
    S = list(dict.fromkeys(int(x) for x in S))
    rc = [0] * n
    cc = [0] * n
    T = []
    for p in S:
        r, c = divmod(p, n)
        if rc[r] < 2 and cc[c] < 2:
            rc[r] += 1
            cc[c] += 1
            T.append(p)
    S = T
    while len(S) > 2:
        Sa = np.array(S, dtype=np.int64)
        cf = _pairs(K[Sa[:, None], Sa[None, :]])
        mx = int(cf.max())
        if mx == 0:
            break
        w = np.flatnonzero(cf == mx)
        j = int(w[rng.randrange(len(w))])
        S.pop(j)
    return S


def _complete(n, K, Sp, rng, need, budget):
    rc = [0] * n
    cc = [0] * n
    for p in Sp:
        rc[p // n] += 1
        cc[p % n] += 1
    drows = [r for r in range(n) if rc[r] < 2]
    dcols = [c for c in range(n) if cc[c] < 2]
    empty = [] if need <= 0 else None
    if not drows or not dcols:
        return empty
    occ = set(Sp)
    cl = [r * n + c for r in drows for c in dcols if r * n + c not in occ]
    if not cl:
        return empty
    cells = np.array(cl, dtype=np.int64)
    s = len(Sp)
    KS = None
    if s >= 1:
        Sa = np.array(Sp, dtype=np.int64)
        KS = np.sort(K[cells[:, None], Sa[None, :]], axis=1)
        if s >= 2:
            ok = ~(KS[:, 1:] == KS[:, :-1]).any(axis=1)
            cells = cells[ok]
            KS = KS[ok]
    c = int(cells.size)
    if c == 0:
        return empty
    perm = np.array(rng.sample(range(c), c), dtype=np.int64)
    if c > 200:
        perm = perm[:200]
    cells = cells[perm]
    if KS is not None:
        KS = KS[perm]
    c = int(cells.size)
    KC = K[cells[:, None], cells[None, :]].astype(np.int64)
    if KS is not None and KS.shape[1] > 0:
        off = (np.arange(c, dtype=np.int64) << 16)[:, None]
        flat = (KS.astype(np.int64) + off).ravel()
        q = (KC + off).ravel()
        pos = np.searchsorted(flat, q)
        pos[pos >= flat.size] = flat.size - 1
        bad = (flat[pos] == q).reshape(c, c)
    else:
        bad = np.zeros((c, c), dtype=bool)
    ar = np.arange(c)
    bad[ar, ar] = True
    pk = np.packbits(bad, axis=1, bitorder='little')
    w = pk.shape[1]
    bs = pk.tobytes()
    confl = [int.from_bytes(bs[i * w:(i + 1) * w], 'little') for i in range(c)]
    KCl = KC.tolist()
    cr = (cells // n).tolist()
    ccl = (cells % n).tolist()
    rowm = {}
    colm = {}
    for i in range(c):
        rowm[cr[i]] = rowm.get(cr[i], 0) | (1 << i)
        colm[ccl[i]] = colm.get(ccl[i], 0) | (1 << i)
    rcap = {r: 2 - rc[r] for r in rowm}
    ccap = {x: 2 - cc[x] for x in colm}
    pc = _pc
    mr = sum(min(rcap[r], pc(rowm[r])) for r in rowm)
    mc = sum(min(ccap[x], pc(colm[x])) for x in colm)
    maxposs = min(mr, mc)
    if maxposs < need:
        return None
    rows = list(rowm)
    cols = list(colm)
    grp = [None] * c
    chosen = []
    best = [need - 1, None]
    nodes = [0]

    def tri(i, j):
        g = grp[i]
        if g is None:
            g = {}
            row = KCl[i]
            for l in range(c):
                k = row[l]
                g[k] = g.get(k, 0) | (1 << l)
            grp[i] = g
        return g[KCl[i][j]]

    def rec(avail):
        nodes[0] += 1
        L = len(chosen)
        if L > best[0]:
            best[0] = L
            best[1] = list(chosen)
            if L >= maxposs:
                raise _Done()
        if nodes[0] > budget:
            raise _Abort()
        if not avail:
            return
        rb = 0
        br = -1
        bcnt = 1 << 30
        ba = 0
        for r in rows:
            k = rcap[r]
            if k:
                a = avail & rowm[r]
                if a:
                    x = pc(a)
                    rb += k if x > k else x
                    if x < bcnt:
                        bcnt = x
                        br = r
                        ba = a
        if br < 0 or L + rb <= best[0]:
            return
        cb = 0
        for x in cols:
            k = ccap[x]
            if k:
                a = avail & colm[x]
                if a:
                    y = pc(a)
                    cb += k if y > k else y
        if L + cb <= best[0]:
            return
        r = br
        a = ba
        while a:
            l = a & -a
            i = l.bit_length() - 1
            a ^= l
            na = avail & ~confl[i]
            for j in chosen:
                na &= ~tri(i, j)
            col = ccl[i]
            rcap[r] -= 1
            ccap[col] -= 1
            if not rcap[r]:
                na &= ~rowm[r]
            if not ccap[col]:
                na &= ~colm[col]
            chosen.append(i)
            rec(na)
            chosen.pop()
            rcap[r] += 1
            ccap[col] += 1
            avail &= ~l
        rec(avail & ~rowm[r])

    try:
        rec((1 << c) - 1)
    except (_Abort, _Done):
        pass
    if best[1] is None:
        return None
    return [int(cells[i]) for i in best[1]]


def _lns(n, K, S, rng, deadline, holder):
    S = list(S)
    bestS = list(S)
    twoN = 2 * n
    stall = 0
    while len(bestS) < twoN and time.time() < deadline:
        rc = [0] * n
        cc = [0] * n
        for p in S:
            rc[p // n] += 1
            cc[p % n] += 1
        rset = set()
        cset = set()
        nl = rng.choice((2, 2, 3, 3, 4))
        if rng.random() < 0.6 and len(S) >= 2:
            drows = [r for r in range(n) if rc[r] < 2]
            dcols = [x for x in range(n) if cc[x] < 2]
            if drows and dcols:
                z = rng.choice(drows) * n + rng.choice(dcols)
                Sset = set(S)
                if z not in Sset:
                    Sa = np.array(S, dtype=np.int64)
                    ks = K[z, Sa]
                    order = np.argsort(ks, kind='stable')
                    sk = ks[order]
                    d = np.flatnonzero(sk[1:] == sk[:-1])
                    if d.size:
                        t = int(d[rng.randrange(d.size)])
                        q = int(Sa[order[t + rng.randrange(2)]])
                        if rng.random() < 0.5:
                            rset.add(q // n)
                        else:
                            cset.add(q % n)
        while len(rset) + len(cset) < nl:
            if rng.random() < 0.5:
                rset.add(rng.randrange(n))
            else:
                cset.add(rng.randrange(n))
        removed = [p for p in S if (p // n) in rset or (p % n) in cset]
        keep = [p for p in S if not ((p // n) in rset or (p % n) in cset)]
        perturb = stall > 250
        need = max(0, len(removed) - 2) if perturb else len(removed)
        add = _complete(n, K, keep, rng, need, 3000)
        if add is not None:
            newS = keep + add
            if len(newS) > len(S):
                stall = 0
            else:
                stall += 1
            if perturb:
                stall = 0
            S = newS
            if len(S) > len(bestS):
                bestS = list(S)
                if len(bestS) > holder[0] or True:
                    holder[2] = list(bestS)
        else:
            stall += 1
    return bestS


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
    if n > 50 or time.time() > end - 1.5:
        return _fill(n, seed, rng, end)
    K = _keys(n)
    S = _fixvalid(n, K, seed, rng)
    add = _complete(n, K, S, rng, 0, 8000)
    if add:
        S = S + add
    S = _fill(n, S, rng, end)
    S = _fixvalid(n, K, S, rng)
    best[2] = list(S)
    S = _lns(n, K, S, rng, end, best)
    return S


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
    best = [0, [], None]
    res = None
    if n <= 64:
        try:
            res = _solve(n, t0, time_limit, rng, best)
        except Exception:
            res = None
    if res is None:
        if best[2]:
            res = best[2]
        else:
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
