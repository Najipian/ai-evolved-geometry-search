# g007 (parent g003): Randomised DFS over rot4, dia2 and full-D4 orbits. Precomputed orbit-pair conflict bitmasks give exact forward checking. Uses MRV, least-constraining-value ordering, Luby restarts and a greedy fill.
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
    # line conflict masks between orbits
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


def _fill(n, pts, rng, deadline):
    pts = list(pts)
    occ = set(pts)
    cand = [p for p in range(n * n) if p not in occ]
    rng.shuffle(cand)
    for z in cand:
        if time.time() > deadline:
            break
        zr, zc = divmod(z, n)
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
    return pts


def _solve(n, t0, time_limit, rng, best):
    deadline = t0 + time_limit - 0.8
    classes = []
    D = _build(n, 'full')
    try:
        r = _dfs(D, rng, 10 ** 9, min(deadline, time.time() + 1.0), best, 3.0)
        if r is not None:
            return [q for o in r for q in D['cells'][o]]
    except _Abort:
        pass
    D = None
    for sym in ('rot4', 'dia2'):
        if time.time() > deadline:
            break
        classes.append(_build(n, sym))
    i = 0
    base = 150
    while classes and time.time() < deadline:
        D = classes[i % len(classes)]
        budget = base * _luby(i // len(classes) + 1)
        noise = rng.choice([1.0, 3.0, 6.0, 12.0])
        i += 1
        try:
            r = _dfs(D, rng, budget, deadline, best, noise)
            if r is not None:
                return [q for o in r for q in D['cells'][o]]
            classes.remove(D)
        except _Abort:
            pass
    return None


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
