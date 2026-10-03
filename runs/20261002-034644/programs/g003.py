# g003 (parent g002): Search first over solutions invariant under the 180° face-diagonal rotation (x,y,z)->(y,x,n-1-z). Then hand the best symmetric set to the asymmetric tabu ILS.
# Phase 1: orbit-level ILS under involution (x,y,z)->(y,x,n-1-z); Phase 2: g002 asymmetric ILS seeded with it.
import random
import time
import itertools
import numpy as np


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit * 0.88
    sym_deadline = t0 + time_limit * 0.45
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
    # verify sym_best
    if len(sym_best) >= 4:
        chk = _repair([tuple(int(v) for v in G[c]) for c in sym_best])
        if len(chk) != len(sym_best):
            sym_best = [int((x * n + y) * n + z) for (x, y, z) in chk]

    # ================= Phase 2: asymmetric ILS (g002) =================
    M = np.zeros((cap, N), np.float32)
    count = np.zeros(N, np.float32)
    occ = np.zeros(N, bool)
    S = []
    tabu = np.zeros(N, np.int64)
    itc = [0]

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
    if len(sym_best) >= 3:
        for c in sym_best:
            add(c)
    else:
        while True:
            cs = rng.sample(range(N), 3)
            a, b, c = G[cs]
            if np.cross(b - a, c - a).any():
                break
        for c in cs:
            add(c)
    fill(8)
    best = list(S)
    if len(sym_best) > len(best):
        best = list(sym_best)
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
