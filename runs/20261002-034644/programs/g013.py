# g013 (parent g003): Use a short symmetric orbit search, then a fixed-k min-conflict tabu search with bincount updates that grows k whenever conflicts reach zero.
# Phase 1: orbit-level ILS under involution (x,y,z)->(y,x,n-1-z).
# Phase 2: fixed-k min-conflict tabu with sparse bincount incremental updates; grows k at zero conflicts,
# with perturbation and elite restarts from best feasible set.
import random
import time
import itertools
import numpy as np


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit * 0.91
    sym_deadline = t0 + time_limit * 0.18
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

    # ================= Phase 1: symmetric orbit search =================
    partner = (G[:, 1] * n + G[:, 0]) * n + (n - 1 - G[:, 2])
    ar_N = np.arange(N)
    oid = np.minimum(ar_N, partner)
    reps = np.flatnonzero(oid == ar_N)

    def orb(o):
        q = int(partner[o])
        return [int(o)] if q == o else [int(o), q]

    scount = np.zeros(N, np.float32)
    socc = np.zeros(N, bool)
    SS = []
    stabu = np.zeros(N, np.int64)

    def sadd(c):
        if len(SS) >= 2:
            W, _ = normals(c, SS)
            scount[...] += ((W @ H) == 0).astype(np.float32).sum(0)
        SS.append(c)
        socc[c] = True

    def sremove(c):
        SS.remove(c)
        socc[c] = False
        if len(SS) >= 2:
            W, _ = normals(c, SS)
            scount[...] -= ((W @ H) == 0).astype(np.float32).sum(0)

    def srebuild(cells):
        scount[:] = 0
        socc[:] = False
        SS.clear()
        for c in cells:
            sadd(c)

    def orbit_ok(o):
        pts = orb(o)
        for p in pts:
            if socc[p] or scount[p] != 0:
                return False
        for t in range(len(pts)):
            p = pts[t]
            E = SS + pts[:t]
            if len(E) < 2:
                continue
            W, _ = normals(p, E)
            if (~W[:, :3].any(1)).any():
                return False
            on = (W @ H[:, E]) == 0
            if on.sum(1).max() > 2:
                return False
        return True

    def sfill(ns=4):
        while len(SS) < cap - 2:
            f = (scount == 0) & ~socc
            ok = f & f[partner]
            cands = np.unique(oid[ok])
            if cands.size == 0:
                break
            cl = cands.tolist()
            rng.shuffle(cl)
            good = []
            for o in cl:
                if orbit_ok(o):
                    good.append(o)
                    if len(good) >= ns:
                        break
            if not good:
                break
            if len(good) == 1 or len(SS) < 2:
                ch = rng.choice(good)
            else:
                F = np.flatnonzero(f)
                HF = H[:, F]
                best_s, best_c = None, []
                for o in good:
                    blk = np.zeros(F.size, bool)
                    for p in orb(o):
                        W, _ = normals(p, SS)
                        blk |= ((W @ HF) == 0).any(0)
                    s = int(blk.sum())
                    if best_s is None or s < best_s:
                        best_s, best_c = s, [o]
                    elif s == best_s:
                        best_c.append(o)
                ch = rng.choice(best_c)
            for p in orb(ch):
                sadd(p)

    def conflicts(Sl, pts):
        T = Sl + pts
        tally = {}
        own = int(oid[pts[0]])
        for p in pts:
            others = [c for c in T if c != p]
            if len(others) < 2:
                continue
            W, _ = normals(p, others)
            i, j, _ = pairs(len(others))
            deg = ~W[:, :3].any(1)
            on = (W @ H[:, others]) == 0
            bad = deg | (on.sum(1) > 2)
            if not bad.any():
                continue
            bi = np.flatnonzero(bad)
            B = on[bi].copy()
            rr = np.arange(len(bi))
            B[rr, i[bi]] = True
            B[rr, j[bi]] = True
            colcnt = B.sum(0)
            for idx in np.flatnonzero(colcnt):
                q = int(oid[others[idx]])
                if q == own:
                    continue
                tally[q] = tally.get(q, 0) + int(colcnt[idx])
        return tally

    def stenure():
        return 4 + rng.randrange(max(2, len(SS) // 4))

    def sforce(it):
        cand = reps[~socc[reps] & (stabu[reps] <= it)]
        if cand.size == 0:
            return
        ns = min(30, cand.size)
        sample = cand[rng.sample(range(cand.size), ns)]
        sc = scount[sample] + scount[partner[sample]]
        mn = sc.min()
        opts = sample[sc == mn]
        o = int(opts[rng.randrange(len(opts))])
        pts = orb(o)
        Sl = list(SS)
        removed = []
        while True:
            tally = conflicts(Sl, pts)
            if not tally:
                break
            mx = max(tally.values())
            ch = [q for q, v in tally.items() if v == mx]
            r = rng.choice(ch)
            removed.append(r)
            for p in orb(r):
                if p in Sl:
                    Sl.remove(p)
        for r in removed:
            for p in orb(r):
                if socc[p]:
                    sremove(p)
            stabu[r] = it + stenure()
        for p in pts:
            sadd(p)
        sfill()

    sym_best = []
    try:
        sfill(8)
        sym_best = list(SS)
        sit = 0
        slast = 0
        sstag = 250
        while time.time() < sym_deadline:
            sit += 1
            sforce(sit)
            if len(SS) > len(sym_best):
                sym_best = list(SS)
                slast = sit
            elif len(SS) < len(sym_best) - 3 or sit - slast > sstag:
                srebuild(sym_best)
                if sit - slast > sstag:
                    orbs_in = list(set(int(oid[c]) for c in SS))
                    nk = rng.randint(1, 2)
                    for _ in range(nk):
                        if len(orbs_in) > 2:
                            r = orbs_in.pop(rng.randrange(len(orbs_in)))
                            for p in orb(r):
                                if socc[p]:
                                    sremove(p)
                            stabu[r] = sit + stenure()
                sfill()
                slast = sit
                if len(SS) > len(sym_best):
                    sym_best = list(SS)
    except Exception:
        pass
    sym_best = [int(c) for c in sym_best]
    if len(sym_best) >= 4:
        chk = _repair([tuple(int(v) for v in G[c]) for c in sym_best])
        if len(chk) != len(sym_best):
            sym_best = [int((x * n + y) * n + z) for (x, y, z) in chk]

    # ================= Phase 2: min-conflict tabu =================
    M = np.zeros((cap + 1, N), np.int64)
    cnt = np.zeros(N, np.int64)
    occ = np.zeros(N, bool)
    S = []
    pc2 = {}

    def pij(k):
        r = pc2.get(k)
        if r is None:
            a, b = np.triu_indices(k, 1)
            r = (a.astype(np.int64), b.astype(np.int64))
            pc2[k] = r
        return r

    def plane_hits(c, Oc):
        i, j = pij(len(Oc))
        P = G[c]
        u = Oc[i] - P
        v = Oc[j] - P
        nr = np.cross(u, v).reshape(-1, 3)
        d = nr @ P
        deg = ~nr.any(axis=1)
        if deg.any():
            d[deg] = 1
        W = np.empty((len(i), 4), np.float32)
        W[:, :3] = nr
        W[:, 3] = d
        return np.nonzero((W @ H) == 0)

    def add(x):
        k = len(S)
        if k >= 2:
            pl, ce = plane_hits(x, G[np.array(S)])
            i, j = pij(k)
            tot = np.bincount(ce, minlength=N)
            cnt[...] += tot
            upd = np.bincount(np.concatenate((i[pl] * N + ce, j[pl] * N + ce)), minlength=k * N)
            M[:k] += upd.reshape(k, N)
            M[k] = tot
        else:
            M[k] = 0
        S.append(int(x))
        occ[x] = True

    def replace(s, x):
        k = len(S)
        old = S[s]
        rows = np.delete(np.arange(k), s)
        Oc = G[np.array(S)[rows]]
        i, j = pij(k - 1)
        ri = rows[i]
        rj = rows[j]
        plo, ceo = plane_hits(old, Oc)
        pln, cen = plane_hits(x, Oc)
        totn = np.bincount(cen, minlength=N)
        cnt[...] += totn - np.bincount(ceo, minlength=N)
        up = np.bincount(np.concatenate((ri[pln] * N + cen, rj[pln] * N + cen)), minlength=k * N)
        dn = np.bincount(np.concatenate((ri[plo] * N + ceo, rj[plo] * N + ceo)), minlength=k * N)
        M[:k] += (up - dn).reshape(k, N)
        M[s] = totn
        S[s] = int(x)
        occ[old] = False
        occ[x] = True
        return old

    def rebuild(cells):
        M[:] = 0
        cnt[:] = 0
        occ[:] = False
        S.clear()
        for c in cells:
            add(c)

    def add_best():
        F = np.flatnonzero(~occ)
        v = cnt[F]
        opts = F[v == v.min()]
        add(int(opts[rng.randrange(len(opts))]))

    tabu_in = np.zeros(N, np.int64)
    lock = np.zeros(N, np.int64)
    BIG = 1 << 40

    best_feas = list(sym_best) if len(sym_best) >= 4 else []
    try:
        if len(sym_best) >= 4:
            rebuild(sym_best)
        else:
            while True:
                cs = rng.sample(range(N), 3)
                a, b, c = G[cs]
                if np.cross(b - a, c - a).any():
                    break
            for c in cs:
                add(c)
        it = 0
        bestQ = BIG
        bestS_k = list(S)
        last_imp = 0
        npert = 0
        while time.time() < deadline:
            it += 1
            k = len(S)
            Sa = np.array(S)
            conf = cnt[Sa] - M[np.arange(k), Sa]
            Q = int(conf.sum())
            if Q <= 0:
                if len(S) > len(best_feas):
                    best_feas = list(S)
                if k >= cap - 1:
                    break
                add_best()
                bestQ = BIG
                last_imp = it
                npert = 0
                continue
            if Q < bestQ:
                bestQ = Q
                bestS_k = list(S)
                last_imp = it
            stag = 120 + 4 * k
            if it - last_imp > stag:
                npert += 1
                last_imp = it
                if npert % 3 == 0 and len(best_feas) >= 4:
                    cells = list(best_feas)
                    nr_ = rng.randint(0, 2)
                    for _ in range(nr_):
                        if len(cells) > 4:
                            cl = cells.pop(rng.randrange(len(cells)))
                            tabu_in[cl] = it + 10 + rng.randrange(10)
                    rebuild(cells)
                    while len(S) < k:
                        add_best()
                    bestQ = BIG
                else:
                    if npert % 3 == 2:
                        rebuild(bestS_k)
                    for _ in range(rng.randint(2, 4)):
                        kk = len(S)
                        s = rng.randrange(kk)
                        row = cnt - M[s] + occ.astype(np.int64) * BIG
                        tp = min(30, N - kk - 1)
                        top = np.argpartition(row, tp)[:tp]
                        x = int(top[rng.randrange(tp)])
                        old = replace(s, x)
                        tabu_in[old] = it + 10 + rng.randrange(10)
                continue
            rows = np.flatnonzero(conf > 0)
            D = cnt[None, :] - M[rows] - conf[rows][:, None]
            D = D + (occ.astype(np.int64) * BIG)[None, :]
            asp = (Q + 4 * D) < bestQ
            tabmask = (tabu_in > it)[None, :] | (lock[Sa[rows]] > it)[:, None]
            V = np.where(tabmask & ~asp, BIG, D)
            mn = V.min()
            if mn >= BIG:
                r = rng.randrange(len(rows))
                x = int(np.argmin(D[r]))
            else:
                cand = np.flatnonzero(V.ravel() == mn)
                idx = int(cand[rng.randrange(cand.size)])
                r, x = divmod(idx, N)
            s = int(rows[r])
            old = replace(s, x)
            tabu_in[old] = it + 7 + rng.randrange(8 + k // 3)
            lock[x] = it + 2 + rng.randrange(max(2, k // 5))
    except Exception:
        pass

    if len(sym_best) > len(best_feas):
        best_feas = list(sym_best)
    pts = [tuple(int(v) for v in G[c]) for c in best_feas]
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
