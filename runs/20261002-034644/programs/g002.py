# g002 (parent s0): Iterated local search on an incrementally maintained count of blocking planes per cell, with exact (1,2)-swap scans, tabu plateau swaps and forced insertions.
import random
import time
import itertools
import numpy as np


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit * 0.88
    rng = random.Random(seed)
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    if n == 2:
        return [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    G = np.array(list(itertools.product(range(n), repeat=3)), dtype=np.int64)
    N = len(G)
    H = np.empty((4, N), np.float32)
    H[:3] = G.T
    H[3] = -1.0
    cap = 3 * n + 8
    M = np.zeros((cap, N), np.float32)
    count = np.zeros(N, np.float32)
    occ = np.zeros(N, bool)
    S = []
    tabu = np.zeros(N, np.int64)
    itc = [0]

    pc = {}

    def pairs(k):
        r = pc.get(k)
        if r is None:
            i, j = np.triu_indices(k, 1)
            m = len(i)
            inc = np.zeros((k, m), np.float32)
            ar = np.arange(m)
            if m:
                inc[i, ar] = 1
                inc[j, ar] = 1
            r = (i, j, inc)
            pc[k] = r
        return r

    c3 = {}

    def comb3(k):
        r = c3.get(k)
        if r is None:
            r = np.array(list(itertools.combinations(range(k), 3)), dtype=np.int64).reshape(-1, 3)
            c3[k] = r
        return r

    def normals(c, others):
        i, j, inc = pairs(len(others))
        P = G[c]
        O = G[others] if len(others) else np.zeros((0, 3), np.int64)
        u = O[i] - P
        v = O[j] - P
        nr = np.cross(u, v).reshape(-1, 3)
        d = nr @ P
        deg = ~nr.any(axis=1)
        if deg.any():
            d[deg] = 1
        W = np.empty((len(i), 4), np.float32)
        W[:, :3] = nr
        W[:, 3] = d
        return W, inc

    def add(c):
        k = len(S)
        if k >= 2:
            W, inc = normals(c, S)
            mf = ((W @ H) == 0).astype(np.float32)
            tot = mf.sum(axis=0)
            count[...] += tot
            M[k] = tot
            M[:k] += inc @ mf
        else:
            M[k] = 0
        S.append(c)
        occ[c] = True

    def remove(slot):
        c = S[slot]
        k = len(S)
        others = S[:slot] + S[slot + 1:]
        if k - 1 >= 2:
            W, inc = normals(c, others)
            mf = ((W @ H) == 0).astype(np.float32)
            tot = mf.sum(axis=0)
            count[...] -= tot
            rows = list(range(slot)) + list(range(slot + 1, k))
            M[rows] -= inc @ mf
        last = k - 1
        if slot != last:
            M[slot] = M[last]
            S[slot] = S[last]
        M[last] = 0
        S.pop()
        occ[c] = False
        return c

    def choose(F, ns):
        if F.size == 1:
            return int(F[0])
        if F.size > ns:
            cands = [int(F[x]) for x in rng.sample(range(F.size), ns)]
        else:
            cands = [int(x) for x in F]
        HF = H[:, F]
        best_c, best_s = [], None
        for c in cands:
            W, _ = normals(c, S)
            s = int(((W @ HF) == 0).any(axis=0).sum())
            if best_s is None or s < best_s:
                best_s = s
                best_c = [c]
            elif s == best_s:
                best_c.append(c)
        return rng.choice(best_c)

    def fill(ns=4):
        added = 0
        while len(S) < cap - 1:
            F = np.flatnonzero((count == 0) & ~occ)
            if F.size == 0:
                break
            add(choose(F, ns))
            added += 1
        return added

    def rebuild(cells):
        M[:] = 0
        count[:] = 0
        occ[:] = False
        S.clear()
        for c in cells:
            add(c)

    def find_12(eq):
        k = len(S)
        sizes = eq.sum(axis=1)
        order = list(np.flatnonzero(sizes >= 2))
        rng.shuffle(order)
        if not order:
            return None
        i, j, _ = pairs(k - 1)
        for p in order:
            p = int(p)
            E = np.flatnonzero(eq[p])
            others = S[:p] + S[p + 1:]
            O = G[others]
            P = G[E]
            u = O[i][None, :, :] - P[:, None, :]
            v = O[j][None, :, :] - P[:, None, :]
            nr = np.cross(u, v)
            d = (nr * P[:, None, :]).sum(-1)
            vals = np.matmul(nr, P.T) - d[:, :, None]
            blk = (vals == 0).any(axis=1)
            np.fill_diagonal(blk, True)
            ok = np.argwhere(~blk)
            if len(ok):
                a, b = ok[rng.randrange(len(ok))]
                return p, int(E[a]), int(E[b])
        return None

    def tenure():
        return 5 + rng.randrange(max(2, len(S) // 2))

    def force_insert():
        k = len(S)
        cand = np.flatnonzero(~occ & (tabu <= itc[0]))
        if cand.size == 0:
            return
        ns = min(40, cand.size)
        sample = cand[rng.sample(range(cand.size), ns)]
        cs = count[sample]
        mn = cs.min()
        opts = sample[cs == mn]
        c = int(opts[rng.randrange(len(opts))])
        ti = comb3(k)
        Sa = np.array(S)
        A = G[Sa[ti[:, 0]]]
        B = G[Sa[ti[:, 1]]]
        C = G[Sa[ti[:, 2]]]
        nr = np.cross(B - A, C - A)
        d = (nr * A).sum(1)
        on = ((nr @ G[c] - d) == 0) & nr.any(1)
        trip = ti[on]
        removed = []
        while trip.size:
            cnt = np.bincount(trip.ravel(), minlength=k)
            mx = cnt.max()
            ch = np.flatnonzero(cnt == mx)
            s = int(ch[rng.randrange(len(ch))])
            removed.append(S[s])
            trip = trip[~(trip == s).any(axis=1)]
        for cell in removed:
            remove(S.index(cell))
            tabu[cell] = itc[0] + tenure()
        add(c)
        fill()

    # init
    while True:
        cs = rng.sample(range(N), 3)
        a, b, c = G[cs]
        if np.cross(b - a, c - a).any():
            break
    for c in cs:
        add(c)
    fill(8)
    best = list(S)
    last_best = 0
    stag = 350
    pforce = 0.08

    while time.time() < deadline:
        itc[0] += 1
        it = itc[0]
        k = len(S)
        eq = (M[:k] == count[None, :]) & ((count > 0) & ~occ)[None, :]
        r = find_12(eq)
        if r is not None:
            p, c1, c2 = r
            cell = remove(p)
            tabu[cell] = it + tenure()
            add(c1)
            add(c2)
            fill()
        else:
            eqt = eq & (tabu <= it)[None, :]
            nz = np.flatnonzero(eqt.ravel())
            if nz.size and rng.random() > pforce:
                idx = int(nz[rng.randrange(nz.size)])
                p, c = divmod(idx, N)
                cell = remove(p)
                tabu[cell] = it + tenure()
                add(c)
                fill()
            else:
                force_insert()
        if len(S) > len(best):
            best = list(S)
            last_best = it
        elif len(S) < len(best) - 2 or it - last_best > stag:
            rebuild(best)
            nk = rng.randint(1, 3) if it - last_best > stag else 0
            for _ in range(nk):
                if len(S) > 4:
                    cell = remove(rng.randrange(len(S)))
                    tabu[cell] = it + tenure()
            fill()
            last_best = it
            if len(S) > len(best):
                best = list(S)

    pts = [tuple(int(v) for v in G[c]) for c in best]
    return _repair(pts)


def _repair(pts):
    pts = list(pts)
    while len(pts) >= 3:
        P = np.array(pts, dtype=np.int64)
        k = len(P)
        ti = np.array(list(itertools.combinations(range(k), 3)), dtype=np.int64)
        A, B, C = P[ti[:, 0]], P[ti[:, 1]], P[ti[:, 2]]
        nr = np.cross(B - A, C - A)
        d = (nr * A).sum(1)
        deg = ~nr.any(1)
        vals = nr @ P.T - d[:, None]
        onc = (vals == 0).sum(1)
        bad = deg | (onc > 3)
        if not bad.any():
            break
        cnt = np.bincount(ti[bad].ravel(), minlength=k)
        pts.pop(int(cnt.argmax()))
    return pts
