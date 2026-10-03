# g002 (parent s0): Plateau search with numpy-vectorised "free cell" counts (cells not coplanar with any triple). It does best-removal swaps with tabu, and kicks on stagnation, warm-started from `start`.
import random
import time
import numpy as np


def construct(n, time_limit, seed, start=None):
    t0 = time.time()
    deadline = t0 + time_limit * 0.92
    rng = random.Random(seed)
    N = n * n * n
    idx = np.indices((n, n, n)).reshape(3, -1)  # (3,N)
    Xt = idx.astype(np.float32)
    C = idx.T.astype(np.float32)  # (N,3)
    pair_cache = {}

    def pairs(m):
        r = pair_cache.get(m)
        if r is None:
            r = np.triu_indices(m, 1)
            pair_cache[m] = r
        return r

    zeros = np.zeros(N, dtype=np.int32)

    def D_for(p, others):
        m = len(others)
        if m < 2:
            return zeros
        pp = C[p]
        d = C[others] - pp
        iu, ju = pairs(m)
        nrm = np.cross(d[iu], d[ju])
        off = nrm @ pp
        M = (nrm @ Xt) == off[:, None]
        return M.sum(0, dtype=np.int32)

    def to_idx(p):
        return (p[0] * n + p[1]) * n + p[2]

    def collinear3(a, b, c):
        d1 = C[b] - C[a]
        d2 = C[c] - C[a]
        return not np.any(np.cross(d1, d2))

    S = []
    cnt = np.zeros(N, dtype=np.int32)

    def add(t):
        nonlocal cnt
        if len(S) >= 2:
            cnt = cnt + D_for(t, S)
        S.append(t)

    def remove(s):
        nonlocal cnt
        others = [x for x in S if x != s]
        cnt = cnt - D_for(s, others)
        S.remove(s)

    def can_add(t):
        if t in S:
            return False
        if len(S) < 2:
            return True
        if len(S) == 2:
            return not collinear3(S[0], S[1], t)
        return cnt[t] == 0

    if start:
        for p in start:
            try:
                t = to_idx(tuple(p))
            except Exception:
                continue
            if 0 <= t < N and can_add(t):
                add(t)
    while len(S) < 3:
        t = rng.randrange(N)
        if can_add(t):
            add(t)

    tabu = np.zeros(N, dtype=np.int64)
    prot = {}
    it = 0

    def free_mask():
        m = cnt == 0
        m &= tabu <= it
        return m

    def fill():
        while True:
            fm = np.flatnonzero(cnt == 0)
            if len(fm) == 0:
                return
            add(int(fm[rng.randrange(len(fm))]))

    fill()
    best = list(S)
    last_improve = 0
    stall_limit = 1500

    while time.time() < deadline:
        it += 1
        fm = free_mask()
        nz = np.flatnonzero(fm)
        if len(nz):
            add(int(nz[rng.randrange(len(nz))]))
            if len(S) > len(best):
                best = list(S)
                last_improve = it
            continue
        if it - last_improve > stall_limit:
            # kick: remove a few random points and refill
            r = rng.randint(2, 4)
            for _ in range(r):
                s = S[rng.randrange(len(S))]
                remove(s)
            tabu[:] = 0
            prot.clear()
            fill()
            if len(S) > len(best):
                best = list(S)
            last_improve = it - stall_limit // 2
            continue
        pool = [s for s in S if prot.get(s, 0) <= it]
        if not pool:
            pool = list(S)
        rng.shuffle(pool)
        results = []
        for s in pool[:7]:
            others = [x for x in S if x != s]
            Ds = D_for(s, others)
            F = (cnt - Ds) == 0
            F &= tabu <= it
            F[s] = False
            c = int(np.count_nonzero(F))
            if c > 0:
                results.append((c, s, Ds, F))
        if not results:
            continue
        if rng.random() < 0.6:
            mx = max(r_[0] for r_ in results)
            sel = [r_ for r_ in results if r_[0] == mx]
            c, s, Ds, F = sel[rng.randrange(len(sel))]
        else:
            c, s, Ds, F = results[rng.randrange(len(results))]
        cells = np.flatnonzero(F)
        t = int(cells[rng.randrange(len(cells))])
        cnt = cnt - Ds
        S.remove(s)
        tabu[s] = it + rng.randint(6, 20)
        cnt = cnt + D_for(t, S)
        S.append(t)
        prot[t] = it + rng.randint(2, 6)
        if len(prot) > 200:
            prot = {k_: v for k_, v in prot.items() if v > it}

    out = [(s // (n * n), (s // n) % n, s % n) for s in best]
    return out
