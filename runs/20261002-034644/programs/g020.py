# g020 (parent g011): Replace the old symmetric orbit ILS with orbit-level min-conflict tabu searches under C2 and C3 symmetry, then seed the asymmetric ILS and tabu phases.
import random
import time
import itertools
import numpy as np


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit * 0.93
    c2_deadline = t0 + time_limit * 0.13
    c3_deadline = t0 + time_limit * 0.26
    ils_deadline = t0 + time_limit * 0.36
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
    gx = np.repeat(np.arange(n, dtype=np.int64), n)
    gy = np.tile(np.arange(n, dtype=np.int64), n)
    gxf = gx.astype(np.float64)
    gyf = gy.astype(np.float64)
    UV = ((1, 2), (0, 2), (0, 1))
    EMPTY = np.zeros(0, np.int64)
    BIG = 1 << 40
    ar_N = np.arange(N)
    ar_k = np.arange(cap)

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

    def nint(c, others):
        i, j, _ = pairs(len(others))
        P = G[c]
        O = G[others]
        nr = np.cross(O[i] - P, O[j] - P).reshape(-1, 3)
        d = nr @ P
        return nr, d, i, j

    def plane_cells(nr, d):
        if len(nr) == 0:
            return EMPTY, EMPTY
        absn = np.abs(nr)
        mx = absn.max(1)
        ax = absn.argmax(1)
        nrf = nr.astype(np.float64)
        df = d.astype(np.float64)
        pls = []
        cls = []
        for t in range(3):
            sel = np.flatnonzero((ax == t) & (mx > 0))
            if sel.size == 0:
                continue
            u, v = UV[t]
            num = df[sel][:, None] - nrf[sel, u][:, None] * gxf[None, :] - nrf[sel, v][:, None] * gyf[None, :]
            q = num / nrf[sel, t][:, None]
            ok = (q >= 0) & (q < n) & (q == np.floor(q))
            pi, gi = np.nonzero(ok)
            w = q[pi, gi].astype(np.int64)
            xyz = [None, None, None]
            xyz[u] = gx[gi]
            xyz[v] = gy[gi]
            xyz[t] = w
            pls.append(sel[pi])
            cls.append((xyz[0] * n + xyz[1]) * n + xyz[2])
        if not pls:
            return EMPTY, EMPTY
        return np.concatenate(pls), np.concatenate(cls)

    M = np.zeros((cap, N), np.int32)
    count = np.zeros(N, np.int32)
    occ = np.zeros(N, bool)
    S = []
    tabu = np.zeros(N, np.int64)
    itc = [0]

    def add(c):
        k = len(S)
        if k >= 2:
            nr, d, i, j = nint(c, S)
            pl, cells = plane_cells(nr, d)
            tot = np.bincount(cells, minlength=N)
            count[...] += tot
            M[k] = tot
            if pl.size:
                rows = np.concatenate([i[pl], j[pl]])
                cc = np.concatenate([cells, cells])
                M[:k] += np.bincount(rows * N + cc, minlength=k * N).reshape(k, N)
        else:
            M[k] = 0
        S.append(c)
        occ[c] = True

    def remove(slot):
        c = S[slot]
        k = len(S)
        others = S[:slot] + S[slot + 1:]
        if k - 1 >= 2:
            nr, d, i, j = nint(c, others)
            pl, cells = plane_cells(nr, d)
            tot = np.bincount(cells, minlength=N)
            count[...] -= tot
            if pl.size:
                ridx = np.array(list(range(slot)) + list(range(slot + 1, k)), dtype=np.int64)
                rows = ridx[np.concatenate([i[pl], j[pl]])]
                cc = np.concatenate([cells, cells])
                M[:k] -= np.bincount(rows * N + cc, minlength=k * N).reshape(k, N)
        last = k - 1
        if slot != last:
            M[slot] = M[last]
            S[slot] = S[last]
        M[last] = 0
        S.pop()
        occ[c] = False
        return c

    def rebuild(cells):
        M[:] = 0
        count[:] = 0
        occ[:] = False
        S.clear()
        for c in cells:
            add(c)

    # ================= Phase 1: symmetric orbit min-conflict tabu =================
    def sym_tabu(perms, tdead):
        OM = np.stack([ar_N] + perms, axis=1)
        m = OM.shape[1]
        fixed = OM[:, 1] == ar_N
        rep = OM.min(1)
        isrep = rep == ar_N
        rebuild([])
        tb = np.zeros(N, np.int64)
        sbest = []
        it = 0
        bestV = None
        lastimp = 0

        def ocells(c):
            return sorted(set(int(v) for v in OM[c]))

        def free_reps():
            return isrep & ~occ[OM].any(1)

        def grow_orb():
            fr = np.flatnonzero(free_reps())
            if fr.size == 0:
                return False
            sc = np.where(fixed[fr], count[fr], count[OM[fr]].sum(1))
            mn = sc.min()
            opts = fr[sc == mn]
            x = int(opts[rng.randrange(opts.size)])
            oc = ocells(x)
            if len(S) + len(oc) > cap - 1:
                return False
            for e in oc:
                add(e)
            return True

        while time.time() < tdead:
            it += 1
            k = len(S)
            if k:
                Sa = np.array(S, np.int64)
                conf = (count[Sa] - M[ar_k[:k], Sa]).astype(np.int64)
                V = int(conf.sum())
            else:
                V = 0
            if V == 0:
                if k > len(sbest):
                    sbest = list(S)
                if not grow_orb():
                    break
                bestV = None
                lastimp = it
                continue
            if bestV is None or V < bestV:
                bestV = V
                lastimp = it
            elif it - lastimp > 100 + 6 * k:
                rebuild(sbest)
                nrem = rng.randint(1, 2) if rng.random() < 0.85 else rng.randint(2, 4)
                for _ in range(nrem):
                    if len(S) > 2 * m + 2:
                        c = S[rng.randrange(len(S))]
                        r = int(rep[c])
                        for e in ocells(c):
                            remove(S.index(e))
                        tb[r] = it + 8
                while len(S) <= len(sbest):
                    if not grow_orb():
                        break
                bestV = None
                lastimp = it
                continue
            P = np.flatnonzero(conf > 0)
            R = np.unique(rep[Sa[P]])
            if R.size > 1 and rng.random() < 0.03:
                R = R[rng.randrange(R.size):][:1]
            pos = {c: i for i, c in enumerate(S)}
            A = np.empty((R.size, N), np.int64)
            cr = np.empty(R.size, np.int64)
            for t, r in enumerate(R):
                sl = [pos[e] for e in ocells(int(r))]
                A[t] = count - M[sl].sum(0)
                cr[t] = conf[sl].sum()
            D = A[:, OM].sum(2)
            D = np.where(fixed[None, :], A, D) - cr[:, None]
            bad = ~free_reps()
            D[:, bad] = BIG
            tm = (tb > it) & ~bad
            if tm.any():
                sub = D[:, tm]
                D[:, tm] = np.where(V + sub < bestV, sub, BIG)
            mn = D.min()
            if mn >= BIG:
                tb[:] = 0
                continue
            idx = np.flatnonzero(D.ravel() == mn)
            ri, x = divmod(int(idx[rng.randrange(idx.size)]), N)
            r = int(R[ri])
            for e in ocells(r):
                remove(S.index(e))
            oc = ocells(x)
            if len(S) + len(oc) <= cap - 1:
                for e in oc:
                    add(e)
            tb[r] = it + 3 + rng.randrange(max(2, k // (2 * m) + 2))
        return sbest

    sym_best = []
    cands = []
    try:
        p2 = (G[:, 1] * n + G[:, 0]) * n + (n - 1 - G[:, 2])
        cands.append(sym_tabu([p2], c2_deadline))
    except Exception:
        pass
    try:
        g3 = (G[:, 1] * n + G[:, 2]) * n + G[:, 0]
        cands.append(sym_tabu([g3, g3[g3]], c3_deadline))
    except Exception:
        pass
    for cs in cands:
        cs = [int(c) for c in cs]
        if len(cs) >= 4:
            chk = _repair([tuple(int(v) for v in G[c]) for c in cs])
            if len(chk) != len(cs):
                cs = [int((x * n + y) * n + z) for (x, y, z) in chk]
        if len(cs) > len(sym_best) or (len(cs) == len(sym_best) and rng.random() < 0.5):
            sym_best = cs

    # ================= Phase 2: asymmetric ILS =================
    def choose(F, ns):
        if F.size == 1:
            return int(F[0])
        if len(S) < 2:
            return int(F[rng.randrange(F.size)])
        if F.size > ns:
            cands_ = [int(F[x]) for x in rng.sample(range(F.size), ns)]
        else:
            cands_ = [int(x) for x in F]
        HF = H[:, F]
        best_c, best_s = [], None
        for c in cands_:
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

    def find_12(eq):
        k = len(S)
        if k < 3:
            return None
        sizes = eq.sum(axis=1)
        order = [int(x) for x in np.flatnonzero(sizes >= 2)]
        if not order:
            return None
        rng.shuffle(order)
        I, J, _ = pairs(k)
        Sa = G[S].astype(np.float64)
        Cm = np.cross(Sa[I], Sa[J])
        Dm = Sa[I] - Sa[J]
        for p in order:
            cols = np.flatnonzero((I != p) & (J != p))
            if cols.size == 0:
                continue
            CT = Cm[cols].T
            DT = Dm[cols].T
            E = np.flatnonzero(eq[p])
            a, b = np.triu_indices(len(E), 1)
            e1 = E[a]
            e2 = E[b]
            T = len(e1)
            step = max(1, 1500000 // cols.size)
            for st in range(0, T, step):
                A = G[e1[st:st + step]].astype(np.float64)
                B = G[e2[st:st + step]].astype(np.float64)
                det = (B - A) @ CT + np.cross(B, A) @ DT
                good = np.flatnonzero(~(det == 0).any(1))
                if good.size:
                    g = st + int(good[rng.randrange(good.size)])
                    return p, int(e1[g]), int(e2[g])
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

    if len(sym_best) >= 3:
        rebuild(sym_best)
    else:
        rebuild([])
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
    stag = 600
    pforce = 0.08

    try:
        while time.time() < ils_deadline:
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
    except Exception:
        pass

    # ================= Phase 3: fixed-k min-conflict tabu search =================
    try:
        rebuild(best)
        fill()
        if len(S) > len(best):
            best = list(S)
        elites = [list(best)]
        eset = [frozenset(best)]

        def add_elite(sol):
            fs = frozenset(sol)
            if fs in eset:
                return
            if len(elites) < 8:
                elites.append(list(sol))
                eset.append(fs)
            else:
                r = rng.randrange(1, len(elites))
                elites[r] = list(sol)
                eset[r] = fs

        def grow():
            cand = np.flatnonzero(~occ)
            cs = count[cand]
            mn = cs.min()
            opts = cand[cs == mn]
            add(int(opts[rng.randrange(opts.size)]))

        if len(S) < cap - 1:
            grow()
        tb = np.zeros(N, np.int64)
        tin = np.zeros(N, np.int64)
        it2 = 0
        bestV = None
        lastimp = 0
        while time.time() < deadline:
            it2 += 1
            k = len(S)
            Sa = np.array(S, dtype=np.int64)
            conf = (count[Sa] - M[ar_k[:k], Sa]).astype(np.int64)
            V = int(conf.sum())
            if V == 0:
                best = list(S)
                fill()
                if len(S) > len(best):
                    best = list(S)
                elites = [list(best)]
                eset = [frozenset(best)]
                if len(S) >= cap - 1:
                    break
                grow()
                bestV = None
                lastimp = it2
                continue
            if bestV is None or V < bestV:
                bestV = V
                lastimp = it2
            elif it2 - lastimp > 120 + 8 * k:
                if rng.random() < 0.5 and len(elites) > 1:
                    src = elites[rng.randrange(len(elites))]
                else:
                    src = best
                rebuild(src)
                if rng.random() < 0.8:
                    nrem = rng.randint(1, 3)
                else:
                    nrem = rng.randint(3, max(3, len(S) // 5))
                for _ in range(nrem):
                    if len(S) > 4:
                        cell = remove(rng.randrange(len(S)))
                        tb[cell] = it2 + 10
                fill()
                if len(S) > len(best):
                    best = list(S)
                    elites = [list(best)]
                    eset = [frozenset(best)]
                elif len(S) == len(best):
                    add_elite(S)
                while len(S) < len(best) + 1 and len(S) < cap - 1:
                    grow()
                bestV = None
                lastimp = it2
                continue
            P = np.flatnonzero(conf > 0)
            if P.size > 1:
                Pin = P[tin[Sa[P]] <= it2]
                if Pin.size:
                    P = Pin
            if rng.random() < 0.03:
                P = P[rng.randrange(P.size):][:1]
            D = (count[None, :] - M[P]).astype(np.int64) - conf[P][:, None]
            D[:, occ] = BIG
            tm = tb > it2
            if tm.any():
                sub = D[:, tm]
                D[:, tm] = np.where(V + 4 * sub < bestV, sub, BIG)
            mn = D.min()
            if mn >= BIG:
                tb[:] = 0
                continue
            idx = np.flatnonzero(D.ravel() == mn)
            pi, x = divmod(int(idx[rng.randrange(idx.size)]), N)
            p = int(P[pi])
            y = remove(p)
            add(int(x))
            tb[y] = it2 + 4 + rng.randrange(max(2, k // 2))
            tin[int(x)] = it2 + 2 + rng.randrange(3)
    except Exception:
        pass

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
