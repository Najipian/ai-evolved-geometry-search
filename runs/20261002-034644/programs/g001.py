# g001 (parent s0): Numpy-backed blocked-cell counts with SA swap moves: add a low-blocked cell, remove a greedy hitting set of its blocking triples, refill greedily.
import math
import random
import time
import itertools
import numpy as np


def construct(n, time_limit, seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    t0 = time.time()
    deadline = t0 + time_limit * 0.9
    n2 = n * n
    n3 = n2 * n
    coords = np.array([(x, y, z) for x in range(n) for y in range(n) for z in range(n)], dtype=np.int64)
    Ug, Vg = np.meshgrid(np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), indexing='ij')
    U = Ug.ravel()
    V = Vg.ravel()
    base0 = (U * n + V)[None, :]
    base1 = (U * n2 + V)[None, :]
    base2 = (U * n2 + V * n)[None, :]
    OTH = [(1, 2), (0, 2), (0, 1)]

    pair_cache = {}
    trip_cache = {}

    def get_pairs(N):
        r = pair_cache.get(N)
        if r is None:
            I, J = np.triu_indices(N, 1)
            r = (I.astype(np.int64), J.astype(np.int64))
            pair_cache[N] = r
        return r

    def get_trips(N):
        r = trip_cache.get(N)
        if r is None:
            T = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int64).reshape(-1, 3)
            r = (T[:, 0].copy(), T[:, 1].copy(), T[:, 2].copy(), T)
            trip_cache[N] = r
        return r

    zeros = np.zeros(n3, dtype=np.int64)

    def incidence(p, P):
        N = P.shape[0]
        if N < 2:
            return zeros
        I, J = get_pairs(N)
        A = P[I] - p
        B = P[J] - p
        nx = A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
        ny = A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2]
        nz = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
        M = np.stack((nx, ny, nz), axis=1)
        d = M @ p
        ab = np.abs(M)
        k = ab.argmax(axis=1)
        nzm = ab.max(axis=1) > 0
        parts = []
        for axis in range(3):
            sel = np.flatnonzero((k == axis) & nzm)
            if sel.size == 0:
                continue
            o0, o1 = OTH[axis]
            c = M[sel, axis][:, None]
            a = M[sel, o0][:, None]
            b = M[sel, o1][:, None]
            num = d[sel][:, None] - a * U[None, :] - b * V[None, :]
            w = num // c
            ok = (w * c == num) & (w >= 0) & (w < n)
            if axis == 0:
                idx = w * n2 + base0
            elif axis == 1:
                idx = base1 + w * n
            else:
                idx = base2 + w
            parts.append(idx[ok])
        if not parts:
            return zeros
        return np.bincount(np.concatenate(parts), minlength=n3)

    pts = []  # list of cell indices
    cnt = np.zeros(n3, dtype=np.int64)
    chosen = np.zeros(n3, dtype=bool)

    def parr():
        if not pts:
            return np.zeros((0, 3), dtype=np.int64)
        return coords[np.array(pts, dtype=np.int64)]

    def add(c, inc=None):
        if inc is None:
            inc = incidence(coords[c], parr())
        cnt[:] += inc
        pts.append(c)
        chosen[c] = True

    def remove(i):
        c = pts[i]
        del pts[i]
        chosen[c] = False
        inc = incidence(coords[c], parr())
        cnt[:] -= inc

    def blocking(qc):
        P = parr()
        N = P.shape[0]
        I, J, K, T = get_trips(N)
        A = P - coords[qc]
        a = A[I]
        b = A[J]
        c = A[K]
        det = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
               - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
               + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))
        return T[det == 0]

    def hitting(T, N):
        removed = []
        while len(T):
            bc = np.bincount(T.ravel(), minlength=N).astype(float) + nrng.random(N) * 0.5
            x = int(bc.argmax())
            removed.append(x)
            T = T[(T != x).all(axis=1)]
        return removed

    def refill():
        while True:
            free = np.flatnonzero((cnt == 0) & ~chosen)
            if free.size == 0:
                return
            if len(pts) < 3:
                add(int(free[0]))
                continue
            if free.size > 10:
                cand = nrng.choice(free, 10, replace=False)
            else:
                cand = free
            P = parr()
            bestd = None
            bestc = None
            besti = None
            for f in cand:
                f = int(f)
                inc = incidence(coords[f], P)
                dmg = int(np.count_nonzero(inc[free] > 0))
                dmg += rng.random() * 0.5
                if bestd is None or dmg < bestd:
                    bestd, bestc, besti = dmg, f, inc
            add(bestc, besti)

    # initial 3 non-collinear points
    while True:
        cs = rng.sample(range(n3), 3)
        a, b, c = coords[cs[0]], coords[cs[1]], coords[cs[2]]
        if np.any(np.cross(b - a, c - a) != 0):
            break
    for c in cs:
        add(c)
    refill()

    best = (list(pts), cnt.copy(), chosen.copy())
    tabu_until = np.zeros(n3, dtype=np.int64)
    it = 0
    last_imp = 0
    tenure = 8
    stag_limit = 400
    T = 0.45
    while time.time() < deadline:
        it += 1
        N = len(pts)
        s = nrng.integers(0, n3, 60)
        s = s[(~chosen[s]) & (tabu_until[s] <= it)]
        if s.size == 0:
            continue
        s = np.unique(s)
        order = np.argsort(cnt[s] + nrng.random(s.size) * 0.5)
        cands = s[order[:6]]
        bestR = None
        bestq = None
        for q in cands:
            q = int(q)
            Tq = blocking(q)
            R = hitting(Tq, N)
            if bestR is None or len(R) < len(bestR):
                bestR, bestq = R, q
        if bestR is None or N - len(bestR) < 3:
            continue
        saved = (list(pts), cnt.copy(), chosen.copy())
        for i in sorted(bestR, reverse=True):
            tabu_until[pts[i]] = it + tenure
            remove(i)
        if cnt[bestq] != 0 or chosen[bestq]:
            pts[:] = saved[0]
            cnt[:] = saved[1]
            chosen[:] = saved[2]
            continue
        add(bestq)
        refill()
        delta = len(pts) - N
        if delta < 0 and rng.random() >= math.exp(delta / T):
            pts[:] = saved[0]
            cnt[:] = saved[1]
            chosen[:] = saved[2]
        elif len(pts) > len(best[0]):
            best = (list(pts), cnt.copy(), chosen.copy())
            last_imp = it
        if it - last_imp > stag_limit:
            pts[:] = best[0]
            cnt[:] = best[1]
            chosen[:] = best[2]
            r = rng.randint(1, 3)
            for _ in range(r):
                if len(pts) > 4:
                    remove(rng.randrange(len(pts)))
            refill()
            if len(pts) > len(best[0]):
                best = (list(pts), cnt.copy(), chosen.copy())
            last_imp = it

    return [tuple(int(v) for v in coords[c]) for c in best[0]]
