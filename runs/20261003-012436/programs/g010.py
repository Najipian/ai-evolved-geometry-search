# g010 (parent g003): Cold restarts happen half the time and use a generic symmetry group (C2, S4 rotoreflection, C3 diagonal, D2 or none) for half the run; never return fewer points than `start`.
import random
import time
import itertools
import numpy as np


def construct(n, time_limit, seed, start=None):
    t0 = time.time()
    deadline = t0 + time_limit * 0.93
    rng = random.Random(seed)
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    if n == 2:
        return [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    start_pts = []
    if start:
        try:
            sp = [tuple(int(v) for v in p) for p in start
                  if all(0 <= int(v) < n for v in p)]
            sp = list(dict.fromkeys(sp))
            start_pts = _repair(sp)
        except Exception:
            start_pts = []
    use_start = len(start_pts) >= 4 and rng.random() < 0.5
    variant = rng.choice([0, 1, 2, 3, 4, 5, 0, 1, 2])
    start_cells = [(x * n + y) * n + z for (x, y, z) in start_pts] if use_start else []
    if start_cells:
        ils_deadline = t0 + time_limit * 0.35
        sym_deadline = t0
    else:
        sym_deadline = t0 + time_limit * 0.50
        ils_deadline = t0 + time_limit * 0.72

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

    # ================= Phase 1: symmetric orbit search =================
    X, Y, Z = G[:, 0], G[:, 1], G[:, 2]
    mm = n - 1

    def cidx(a, b, c):
        return (a * n + b) * n + c

    if variant == 0:
        gens = [cidx(Y, X, mm - Z)]
    elif variant == 1:
        gens = [cidx(mm - X, mm - Y, Z)]
    elif variant == 2:
        gens = [cidx(Y, mm - X, mm - Z), cidx(mm - X, mm - Y, Z), cidx(mm - Y, X, mm - Z)]
    elif variant == 3:
        gens = [cidx(Y, Z, X), cidx(Z, X, Y)]
    elif variant == 4:
        gens = [cidx(mm - X, mm - Y, Z), cidx(X, mm - Y, mm - Z), cidx(mm - X, Y, mm - Z)]
    else:
        gens = []
    ar_N = np.arange(N)
    GP = [ar_N] + gens
    oid = ar_N.copy()
    for g in gens:
        oid = np.minimum(oid, g)
    reps = np.flatnonzero(oid == ar_N)
    gsz = len(GP)

    def orb(o):
        return list(dict.fromkeys(int(g[o]) for g in GP))

    scount = np.zeros(N, np.int32)
    socc = np.zeros(N, bool)
    SS = []
    stabu = np.zeros(N, np.int64)

    def sadd(c):
        if len(SS) >= 2:
            nr, d, _, _ = nint(c, SS)
            _, cells = plane_cells(nr, d)
            scount[...] += np.bincount(cells, minlength=N)
        SS.append(c)
        socc[c] = True

    def sremove(c):
        SS.remove(c)
        socc[c] = False
        if len(SS) >= 2:
            nr, d, _, _ = nint(c, SS)
            _, cells = plane_cells(nr, d)
            scount[...] -= np.bincount(cells, minlength=N)

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
        while len(SS) < cap - gsz:
            f = (scount == 0) & ~socc
            ok = f.copy()
            for g in gens:
                ok &= f[g]
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
        return 4 + rng.randrange(max(2, len(SS) // (2 * gsz) + 2))

    def sforce(it):
        cand = reps[~socc[reps] & (stabu[reps] <= it)]
        if cand.size == 0:
            return
        ns = min(30, cand.size)
        sample = cand[rng.sample(range(cand.size), ns)]
        sc = np.zeros(ns, np.int64)
        for g in GP:
            sc += scount[g[sample]]
        mn = sc.min()
        opts = sample[sc == mn]
        o = int(opts[rng.randrange(len(opts))])
        pts = orb(o)
        if any(socc[p] for p in pts):
            return
        Sl = list(SS)
        removed = []
        guard = 0
        while guard < 200:
            guard += 1
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
        else:
            return
        for r in removed:
            for p in orb(r):
                if socc[p]:
                    sremove(p)
            stabu[r] = it + stenure()
        for p in pts:
            sadd(p)
        sfill()

    sym_best = []
    if not start_cells:
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
                elif len(SS) < len(sym_best) - 3 * gsz or sit - slast > sstag:
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
    else:
        sym_best = list(start_cells)

    # ================= Phase 2: asymmetric ILS =================
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

    def choose(F, ns):
        if F.size == 1:
            return int(F[0])
        if len(S) < 2:
            return int(F[rng.randrange(F.size)])
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
        BIG = 1 << 40
        ar_k = np.arange(cap)
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
    res = _repair(pts)
    if len(start_pts) > len(res):
        return start_pts
    return res


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
