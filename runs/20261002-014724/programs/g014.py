# g014 (parent g010): Faster orbit DFS with itemgetter/reduce, asymmetric completion at deep nodes, and a fix for the crashing grid tabu polisher.
import time
import random
from math import gcd
from functools import reduce
from operator import or_, itemgetter
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
    own = [M[X][X] | selfhit[X] | (1 << X) for X in range(norb)]
    ALL = (1 << norb) - 1
    return {'M': M, 'own': own, 'contrib': contrib,
            'size': [len(x) for x in cellsof], 'cells': cellsof,
            'umask': umask, 'c2': c2, 'H': H, 'free0': ALL & ~invalid,
            'sym': sym}


def _ktab(n):
    m = n - 1
    W = 2 * n - 1
    KT = [-1] * (W * W)
    for dr in range(-m, m + 1):
        for dc in range(-m, m + 1):
            if dr == 0 and dc == 0:
                continue
            g = gcd(dr, dc)
            a = dr // g
            b = dc // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            KT[(dr + m) * W + dc + m] = (a + m) * W + (b + m)
    return KT


def _asym(n, KT, base, best, limit):
    m = n - 1
    W = 2 * n - 1
    rc = [0] * n
    cc = [0] * n
    occ = set(base)
    for p in base:
        rc[p // n] += 1
        cc[p % n] += 1
    drows = [r for r in range(n) if rc[r] < 2]
    dcols = [c for c in range(n) if cc[c] < 2]
    bq = [(p // n) * W + p % n for p in base]
    cset = {}
    offs = {}
    rowc = {}
    for r in drows:
        lst = []
        for c in dcols:
            z = r * n + c
            if z in occ:
                continue
            off = (m - r) * W + (m - c)
            ks = [KT[off + b] for b in bq]
            s = set(ks)
            if len(s) == len(ks):
                cset[z] = s
                offs[z] = off
                lst.append(z)
        rowc[r] = lst
    added = []
    addb = []
    last = {}
    steps = [0]
    nb = len(base)

    def ok(z):
        s = cset[z]
        off = offs[z]
        seen = set()
        for b in addb:
            k = KT[off + b]
            if k in s or k in seen:
                return False
            seen.add(k)
        return True

    def rec():
        steps[0] += 1
        if steps[0] > limit:
            return False
        tot = nb + len(added)
        if tot > best[0]:
            best[0] = tot
            best[1] = list(base) + list(added)
        br = -1
        bl = None
        bc = 1 << 30
        for r in drows:
            d = 2 - rc[r]
            if d <= 0:
                continue
            lr = last.get(r, -1)
            v = [z for z in rowc[r] if z > lr and cc[z % n] < 2 and ok(z)]
            if len(v) < d:
                return False
            c = len(v) - d
            if c < bc:
                bc = c
                br = r
                bl = v
        if br < 0:
            return True
        r = br
        old = last.get(r, -1)
        for z in bl:
            c = z % n
            rc[r] += 1
            cc[c] += 1
            added.append(z)
            addb.append(r * W + c)
            last[r] = z
            if rec():
                return True
            rc[r] -= 1
            cc[c] -= 1
            added.pop()
            addb.pop()
        last[r] = old
        return False

    if rec():
        return list(added)
    return None


def _dfs(D, rng, budget, deadline, best, noise, ctx):
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
    n = ctx['n']
    n2 = 2 * n
    thr = ctx['thr']
    KT = ctx['KT']

    def rec(free, pts):
        cnt[0] += 1
        if cnt[0] > budget:
            raise _Abort()
        if not (cnt[0] & 255) and tm() > deadline:
            raise _Abort()
        if pts > best[0]:
            best[0] = pts
            fl = []
            for o in used:
                fl.extend(cells[o])
            best[1] = fl
        if pts >= thr and pts < n2:
            now = tm()
            if ctx['at'] < 0.35 * (now - ctx['t0']) + 0.02:
                fl = []
                for o in used:
                    fl.extend(cells[o])
                r = _asym(n, KT, fl, best, 300)
                ctx['at'] += tm() - now
                if r is not None:
                    ctx['res'] = fl + r
                    raise _Found()
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
        L = len(used)
        ig = itemgetter(*used) if L >= 2 else None
        u0 = used[0] if L == 1 else -1
        cands = []
        a = ba
        while a:
            l = a & -a
            O = l.bit_length() - 1
            a ^= l
            Mo = M[O]
            if L >= 2:
                G = reduce(or_, ig(Mo), own[O])
            elif L == 1:
                G = own[O] | Mo[u0]
            else:
                G = own[O]
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
    end = t0 + T - 0.6
    t_dfs = t0 + 0.68 * T
    ctx = {'n': n, 'thr': 2 * n - 8, 'at': 0.0, 't0': time.time(),
           'KT': _ktab(n), 'res': None}
    classes = []
    for sym in ('rot4', 'dia2'):
        if time.time() > t_dfs - 1.0:
            break
        classes.append(_build(n, sym))
    ctx['t0'] = time.time()
    active = list(classes)
    i = 0
    base = 150
    while active and time.time() < t_dfs:
        D = active[i % len(active)]
        budget = base * _luby(i // len(active) + 1)
        noise = rng.choice([1.0, 3.0, 6.0, 12.0])
        i += 1
        try:
            r = _dfs(D, rng, budget, t_dfs, best, noise, ctx)
            if r is not None:
                return [q for o in r for q in D['cells'][o]]
            active.remove(D)
        except _Abort:
            pass
        except _Found:
            return ctx['res']
    seed = list(best[1])
    if time.time() < end - 0.3:
        try:
            g = _gils(n, seed, rng, end)
            if len(g) >= len(seed):
                return g
        except Exception:
            pass
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
