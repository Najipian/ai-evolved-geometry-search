# g008 (parent g005): Destroy-and-repair search starting from the warm start. Remove 3–8 points (random, slab or ball), then do bounded exact DFS over freed cells (per-layer ≤3 caps) to add r+1 points; otherwise take a plateau swap.
import math
import random
import time
import itertools
import numpy as np
from math import gcd


def _verify(pts):
    k = len(pts)
    if len(set(pts)) != k:
        return False
    planes = {}
    for i in range(k):
        q = pts[i]
        for j in range(i + 1, k):
            a = pts[j]
            ux, uy, uz = a[0] - q[0], a[1] - q[1], a[2] - q[2]
            for l in range(j + 1, k):
                b = pts[l]
                vx, vy, vz = b[0] - q[0], b[1] - q[1], b[2] - q[2]
                x, y, z = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                if x == 0 and y == 0 and z == 0:
                    return False
                g = gcd(gcd(x, y), z)
                x, y, z = x // g, y // g, z // g
                if x < 0 or (x == 0 and (y < 0 or (y == 0 and z < 0))):
                    x, y, z = -x, -y, -z
                key = (x, y, z, x * q[0] + y * q[1] + z * q[2])
                s = planes.get(key)
                if s is None:
                    planes[key] = {i, j, l}
                else:
                    s.update((i, j, l))
                    if len(s) >= 4:
                        return False
    return True


def construct(n, time_limit, seed, start=None):
    t0 = time.time()
    deadline = t0 + time_limit * 0.85
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    n2 = n * n
    ncell = n ** 3
    ar = np.arange(ncell, dtype=np.int64)
    allc = np.stack([ar // n2, (ar // n) % n, ar % n], axis=1).astype(np.int64)
    g = np.arange(n2, dtype=np.int64)
    G1 = g // n
    G2 = g % n
    AX = [(1, 2, n2, n, 1), (0, 2, n, n2, 1), (0, 1, 1, n2, n)]
    tcache = {}
    pcache = {}

    def triples(s):
        t = tcache.get(s)
        if t is None:
            arr = np.array(list(itertools.combinations(range(s), 3)), dtype=np.int64).reshape(-1, 3)
            t = (arr[:, 0], arr[:, 1], arr[:, 2])
            tcache[s] = t
        return t

    def pairs(s):
        t = pcache.get(s)
        if t is None:
            t = np.triu_indices(s, 1)
            pcache[s] = t
        return t

    CH = 1500

    def free_mask(A):
        blk = np.zeros(ncell, dtype=bool)
        s = len(A)
        if s >= 3:
            i0, i1, i2 = triples(s)
            P0 = A[i0]
            N = np.cross(A[i1] - P0, A[i2] - P0)
            dv = (N * P0).sum(1)
            absN = np.abs(N)
            ax = absN.argmax(1)
            nz = absN.max(1) > 0
            for t, (i, j, mw, m1, m2) in enumerate(AX):
                sel = np.flatnonzero((ax == t) & nz)
                for ch in range(0, len(sel), CH):
                    ss = sel[ch:ch + CH]
                    Ns = N[ss]
                    Nt = Ns[:, t:t + 1]
                    num = dv[ss, None] - Ns[:, i:i + 1] * G1 - Ns[:, j:j + 1] * G2
                    w, r = np.divmod(num, Nt)
                    ok = (r == 0) & (w >= 0) & (w < n)
                    rr, cc = np.nonzero(ok)
                    blk[w[rr, cc] * mw + G1[cc] * m1 + G2[cc] * m2] = True
        if s >= 2:
            pa, pb = pairs(s)
            D = A[pb] - A[pa]
            gg = np.gcd.reduce(np.abs(D), axis=1)
            gg[gg == 0] = 1
            D = D // gg[:, None]
            ts = np.arange(-n, n + 1, dtype=np.int64)
            Pts = A[pa][:, None, :] + ts[None, :, None] * D[:, None, :]
            ok = ((Pts >= 0) & (Pts < n)).all(2)
            q = Pts[ok]
            blk[q[:, 0] * n2 + q[:, 1] * n + q[:, 2]] = True
        if s >= 1:
            blk[A[:, 0] * n2 + A[:, 1] * n + A[:, 2]] = True
        return ~blk

    def greedy():
        S = []
        while time.time() < deadline:
            A = np.array(S, dtype=np.int64).reshape(-1, 3)
            fi = np.flatnonzero(free_mask(A))
            if len(fi) == 0:
                break
            c = int(fi[rng.randrange(len(fi))])
            S.append(tuple(int(v) for v in allc[c]))
        return S

    def repair(A, Fc, rr, removed, node_limit, tlim):
        m = len(Fc)
        s = len(A)
        if m < rr:
            return None, None
        L = [Fc[:, 0].copy(), Fc[:, 1].copy(), Fc[:, 2].copy()]
        if s:
            caps0 = [np.maximum(3 - np.bincount(A[:, d], minlength=n), 0) for d in range(3)]
        else:
            caps0 = [np.full(n, 3, dtype=np.int64) for _ in range(3)]
        if s >= 2:
            pa, pb = pairs(s)
            Apa = A[pa]
            Apb = A[pb]
        cache = {}
        noise = nrng.random(m)
        st = {'nodes': 0, 'plate': None, 'found': None, 'stop': False}

        def crow(i):
            row = cache.get(i)
            if row is None:
                c = Fc[i]
                hit = np.zeros(m, dtype=bool)
                if s >= 2:
                    N = np.cross(Apa - c, Apb - c)
                    dv = N @ c
                    hit |= ((Fc @ N.T) == dv[None, :]).any(1)
                if s >= 1:
                    U = A - c
                    V = Fc - c
                    cr = np.cross(V[:, None, :], U[None, :, :])
                    hit |= (cr == 0).all(2).any(1)
                row = ~hit
                row[i] = False
                cache[i] = row
            return row

        def bound(alive, caps):
            b = 10 ** 9
            for d in range(3):
                cnt = np.bincount(L[d][alive], minlength=n)
                v = int(np.minimum(cnt, caps[d]).sum())
                if v < b:
                    b = v
            return b

        def dfs(alive, C, caps):
            st['nodes'] += 1
            lc = len(C)
            if lc == rr and st['plate'] is None:
                cells = frozenset(tuple(int(v) for v in Fc[i]) for i in C)
                if cells != removed:
                    st['plate'] = list(C)
            if lc >= rr + 1:
                st['found'] = list(C)
                return True
            if st['nodes'] > node_limit or time.time() > tlim:
                st['stop'] = True
                return False
            cand = np.flatnonzero(alive)
            if len(cand) == 0:
                return False
            sc = noise[cand].copy()
            for d in range(3):
                cnt = np.bincount(L[d][cand], minlength=n)
                sc += cnt[L[d][cand]]
            order = cand[np.argsort(sc)]
            X = None
            CC = None
            if lc:
                CC = Fc[np.array(C, dtype=np.int64)]
                Xl = []
                Yl = []
                for a in range(lc):
                    if s:
                        Xl.append(np.repeat(CC[a:a + 1], s, axis=0))
                        Yl.append(A)
                    if a + 1 < lc:
                        Xl.append(np.repeat(CC[a:a + 1], lc - a - 1, axis=0))
                        Yl.append(CC[a + 1:])
                if Xl:
                    X = np.concatenate(Xl)
                    Y = np.concatenate(Yl)
            alive = alive.copy()
            for i in order:
                i = int(i)
                if not alive[i]:
                    continue
                need = rr + 1 if st['plate'] is not None else rr
                if lc + bound(alive, caps) < need:
                    return False
                na = alive & crow(i)
                na[i] = False
                if lc and na.any():
                    idx = np.flatnonzero(na)
                    c = Fc[i]
                    V = Fc[idx]
                    kill = np.zeros(len(idx), dtype=bool)
                    if X is not None:
                        N = np.cross(X - c, Y - c)
                        dv = N @ c
                        kill |= ((V @ N.T) == dv[None, :]).any(1)
                    W = V - c
                    U = CC - c
                    cr = np.cross(W[:, None, :], U[None, :, :])
                    kill |= (cr == 0).all(2).any(1)
                    na[idx[kill]] = False
                nc2 = [cp.copy() for cp in caps]
                for d in range(3):
                    nc2[d][Fc[i, d]] -= 1
                if dfs(na, C + [i], nc2):
                    return True
                if st['stop']:
                    return False
                alive[i] = False
            return False

        dfs(np.ones(m, dtype=bool), [], caps0)
        return st['found'], st['plate']

    S = None
    if start:
        try:
            sp = list(dict.fromkeys(tuple(int(v) for v in p) for p in start))
            if all(0 <= v < n for p in sp for v in p) and _verify(sp):
                S = sp
        except Exception:
            S = None
    if S is None:
        S = greedy()
    best = list(S)

    while time.time() < deadline:
        k = len(S)
        if k < 3:
            S = greedy()
            if len(S) > len(best):
                best = list(S)
            if k >= len(S):
                break
            continue
        r = rng.choice([3, 4, 4, 5, 5, 5, 6, 6, 7, 8])
        r = min(r, k)
        mode = rng.random()
        if mode < 0.35:
            rem = rng.sample(range(k), r)
        elif mode < 0.7:
            d = rng.randrange(3)
            l = rng.uniform(-0.5, n - 0.5)
            keyed = sorted(range(k), key=lambda i: abs(S[i][d] - l) + rng.random() * 0.5)
            rem = keyed[:r]
        else:
            cx, cy, cz = [rng.uniform(0, n - 1) for _ in range(3)]
            keyed = sorted(range(k), key=lambda i: (S[i][0] - cx) ** 2 + (S[i][1] - cy) ** 2
                           + (S[i][2] - cz) ** 2 + rng.random() * 2)
            rem = keyed[:r]
        rs = set(rem)
        Sp = [S[i] for i in range(k) if i not in rs]
        removed = frozenset(S[i] for i in rem)
        A = np.array(Sp, dtype=np.int64).reshape(-1, 3)
        fm = free_mask(A)
        Fc = allc[np.flatnonzero(fm)]
        nl = rng.choice([300, 800, 800, 2000])
        tl = min(deadline, time.time() + 1.5)
        res, plat = repair(A, Fc, r, removed, nl, tl)
        if res is not None:
            S = Sp + [tuple(int(v) for v in Fc[i]) for i in res]
            if len(S) > len(best):
                best = list(S)
        elif plat is not None:
            S = Sp + [tuple(int(v) for v in Fc[i]) for i in plat]

    best = [tuple(int(v) for v in p) for p in best]
    if best and not _verify(best):
        if start:
            return [tuple(int(v) for v in p) for p in start]
        return []
    if start and len(start) > len(best):
        return [tuple(int(v) for v in p) for p in start]
    return best
