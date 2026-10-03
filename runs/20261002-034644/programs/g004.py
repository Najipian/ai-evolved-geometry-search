# g004 (parent g003): Generalize symmetric orbit search to any rotation group, running body-diagonal C3, face-diagonal C2 and D3 phases, then seed the asymmetric tabu ILS with the best set.
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

    X, Y, Z = G[:, 0], G[:, 1], G[:, 2]

    def idx(x, y, z):
        return (x * n + y) * n + z

    P_C2 = idx(Y, X, n - 1 - Z)
    P_C2b = idx(n - 1 - Y, n - 1 - X, n - 1 - Z)
    P_C3 = idx(Y, Z, X)

    # ================= Phase 1: symmetric orbit search (generic group) =================
    def sym_search(perms, sdl):
        oid = np.full(N, -1, np.int64)
        orbs = {}
        for c0 in range(N):
            if oid[c0] >= 0:
                continue
            mem = {c0}
            st = [c0]
            while st:
                c = st.pop()
                for P in perms:
                    q = int(P[c])
                    if q not in mem:
                        mem.add(q)
                        st.append(q)
            lst = sorted(mem)
            r = lst[0]
            for q in lst:
                oid[q] = r
            orbs[r] = lst
        rl = []
        for r in sorted(orbs):
            lst = orbs[r]
            if len(lst) >= 3:
                pts = [tuple(int(v) for v in G[c]) for c in lst]
                if len(_repair(pts)) != len(pts):
                    continue
            rl.append(r)
        if not rl:
            return []
        reps = np.array(rl, dtype=np.int64)
        maxs = max(len(orbs[r]) for r in rl)
        orbM = np.array([orbs[r] + [orbs[r][0]] * (maxs - len(orbs[r])) for r in rl], dtype=np.int64)
        orbW = np.array([[1.0] * len(orbs[r]) + [0.0] * (maxs - len(orbs[r])) for r in rl], dtype=np.float32)
        rowof = np.full(N, -1, np.int64)
        rowof[reps] = np.arange(len(reps))

        def orb(o):
            return orbs[int(o)]

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
            while len(SS) <= cap - 2 - maxs:
                f = (scount == 0) & ~socc
                okr = f[orbM].all(1)
                cands = reps[okr]
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
                        s = int(blk.sum()) / len(orb(o))
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
                for ix in np.flatnonzero(colcnt):
                    q = int(oid[others[ix]])
                    if q == own:
                        continue
                    tally[q] = tally.get(q, 0) + int(colcnt[ix])
            return tally

        def stenure():
            return 4 + rng.randrange(max(2, len(SS) // (2 * maxs) + 2))

        def sforce(it):
            cand = reps[~socc[reps] & (stabu[reps] <= it)]
            if cand.size == 0:
                return
            ns = min(30, cand.size)
            sample = cand[rng.sample(range(cand.size), ns)]
            rows = rowof[sample]
            sc = (scount[orbM[rows]] * orbW[rows]).sum(1) / orbW[rows].sum(1)
            mn = sc.min()
            opts = sample[sc == mn]
            o = int(opts[rng.randrange(len(opts))])
            pts = orb(o)
            Sl = list(SS)
            removed = []
            guard = 0
            while True:
                guard += 1
                tally = conflicts(Sl, pts)
                if not tally or guard > 200:
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
            while time.time() < sdl:
                sit += 1
                sforce(sit)
                if len(SS) > len(sym_best):
                    sym_best = list(SS)
                    slast = sit
                elif len(SS) < len(sym_best) - 3 * maxs // 2 - 1 or sit - slast > sstag:
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
        return sym_best

    sym_best = []
    plan = [([P_C3], 0.17), ([P_C2], 0.20), ([P_C3, P_C2b], 0.08)]
    acc = 0.0
    for perms, frac in plan:
        acc += frac
        sdl = t0 + time_limit * acc
        if time.time() >= sdl:
            continue
        try:
            res = sym_search(perms, sdl)
        except Exception:
            res = []
        if len(res) > len(sym_best) or (len(res) == len(sym_best) and rng.random() < 0.5):
            sym_best = res

    # ================= Phase 2: asymmetric ILS =================
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

    if len(sym_best) >= 3:
        for c in sym_best[:cap - 2]:
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

    try:
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
                    ix = int(nz[rng.randrange(nz.size)])
                    p, c = divmod(ix, N)
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
