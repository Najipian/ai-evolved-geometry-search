# g005 (parent s0): Tabu search with numpy-maintained coplanarity counts per cell and per point. Every (remove point, add cell) swap is scored in one vectorized pass, warm-started from the best known set.
import math
import random
import time
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
    deadline = t0 + time_limit * 0.88
    rng = random.Random(seed)
    nc = n ** 3
    n2 = n * n
    KM = 3 * n + 2
    g = np.arange(n2, dtype=np.int64)
    G1 = g // n
    G2 = g % n
    AX = [(1, 2, n2, n, 1), (0, 2, n, n2, 1), (0, 1, 1, n2, n)]
    conf = np.zeros(nc, dtype=np.int64)
    D = np.zeros((KM, nc), dtype=np.int64)
    Dflat = D.reshape(-1)
    Du = np.zeros(KM, dtype=np.int64)
    U = [0]
    P = np.zeros((KM, 3), dtype=np.int64)
    cellof = [-1] * KM
    tri_cache = {}

    def tri(m):
        t = tri_cache.get(m)
        if t is None:
            t = np.triu_indices(m, 1)
            tri_cache[m] = t
        return t

    def update(slot, sign, kk):
        others = [s for s in range(kk) if s != slot]
        m = len(others)
        if m < 2:
            return
        oth = np.array(others, dtype=np.int64)
        i0, i1 = tri(m)
        oa = oth[i0]
        ob = oth[i1]
        q = P[slot]
        N = np.cross(P[oa] - q, P[ob] - q)
        d = N.dot(q)
        absN = np.abs(N)
        deg = absN.max(axis=1) == 0
        ax = absN.argmax(axis=1)
        cl = []
        rl = []
        for t, (i, j, mw, m1, m2) in enumerate(AX):
            sel = np.flatnonzero((ax == t) & ~deg)
            if sel.size == 0:
                continue
            Ns = N[sel]
            Nt = Ns[:, t:t + 1]
            num = d[sel, None] - Ns[:, i:i + 1] * G1 - Ns[:, j:j + 1] * G2
            w, r = np.divmod(num, Nt)
            ok = (r == 0) & (w >= 0) & (w < n)
            rr, cc = np.nonzero(ok)
            cl.append(w[rr, cc] * mw + G1[cc] * m1 + G2[cc] * m2)
            rl.append(sel[rr])
        if cl:
            cells = np.concatenate(cl)
            rows = np.concatenate(rl)
            conf[:] += sign * np.bincount(cells, minlength=nc)
            idx = np.concatenate((slot * nc + cells, oa[rows] * nc + cells, ob[rows] * nc + cells))
            Dflat[:] += sign * np.bincount(idx, minlength=KM * nc)
        nd = int(deg.sum())
        if nd:
            U[0] += sign * nd
            Du[slot] += sign * nd
            np.add.at(Du, oa[deg], sign)
            np.add.at(Du, ob[deg], sign)

    def set_slot(slot, c):
        cellof[slot] = c
        P[slot, 0] = c // n2
        P[slot, 1] = (c // n) % n
        P[slot, 2] = c % n

    def cur_pts(k):
        return [(int(P[s, 0]), int(P[s, 1]), int(P[s, 2])) for s in range(k)]

    k = 0
    best = []
    if start:
        sp = []
        seen = set()
        for p in start:
            p = tuple(int(v) for v in p)
            if p in seen or not all(0 <= v < n for v in p):
                continue
            seen.add(p)
            sp.append(p)
        for p in sp[:KM - 1]:
            set_slot(k, p[0] * n2 + p[1] * n + p[2])
            update(k, 1, k + 1)
            k += 1
        best = [tuple(p) for p in start]

    BIG = 1 << 40
    cell_tabu = np.zeros(nc, dtype=np.int64)
    slot_tabu = np.zeros(KM, dtype=np.int64)
    it = 0
    phase_best = BIG
    last_imp = 0

    def ctrue(k):
        idx = np.array(cellof[:k], dtype=np.int64)
        return idx, conf[idx] + U[0] - (k - 1) * (k - 2) // 2

    while time.time() < deadline and k < KM - 1:
        it += 1
        if k > 0:
            idx, ct = ctrue(k)
            cost = int(ct.sum()) // 4
        else:
            idx = np.zeros(0, dtype=np.int64)
            ct = np.zeros(0, dtype=np.int64)
            cost = 0
        if cost == 0:
            if k > len(best) or not best:
                cand_best = cur_pts(k)
                if k > len(best):
                    best = cand_best
            # grow
            vals = conf + U[0]
            if k > 0:
                vals = vals.copy()
                vals[idx] = BIG
            mn = vals.min()
            cs = np.flatnonzero(vals == mn)
            c = int(cs[rng.randrange(len(cs))])
            set_slot(k, c)
            update(k, 1, k + 1)
            k += 1
            phase_best = BIG
            last_imp = it
            cell_tabu[:] = 0
            slot_tabu[:] = 0
            continue
        if cost < phase_best:
            phase_best = cost
            last_imp = it
        if it - last_imp > 1500 + 50 * k:
            # perturb: a few random moves
            taken = set(cellof[:k])
            for _ in range(2 + rng.randrange(3)):
                s = rng.randrange(k)
                while True:
                    c = rng.randrange(nc)
                    if c not in taken:
                        break
                taken.discard(cellof[s])
                update(s, -1, k)
                set_slot(s, c)
                update(s, 1, k)
                taken.add(c)
            phase_best = BIG
            last_imp = it
            cell_tabu[:] = 0
            slot_tabu[:] = 0
            continue
        newc = (conf[None, :] + U[0]) - D[:k] - Du[:k, None]
        delta = newc - ct[:, None]
        delta[:, idx] = BIG
        tabu = (cell_tabu > it)[None, :] | (slot_tabu[:k] > it)[:, None]
        allowed = (~tabu) | (delta + cost < phase_best)
        dm = np.where(allowed, delta, BIG)
        mn = dm.min()
        if mn >= BIG:
            cell_tabu[:] = 0
            slot_tabu[:] = 0
            continue
        cs = np.flatnonzero(dm.ravel() == mn)
        pick = int(cs[rng.randrange(len(cs))])
        p = pick // nc
        c = pick % nc
        old = cellof[p]
        update(p, -1, k)
        set_slot(p, c)
        update(p, 1, k)
        cell_tabu[old] = it + 5 + rng.randrange(5 + k // 2)
        slot_tabu[p] = it + 1 + rng.randrange(3)

    if k > 0:
        idx, ct = ctrue(k)
        if int(ct.sum()) == 0 and k > len(best):
            best = cur_pts(k)

    best = [tuple(int(v) for v in p) for p in best]
    if best and not _verify(best):
        if start:
            return [tuple(p) for p in start]
    return best
