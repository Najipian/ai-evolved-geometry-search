# g015 (parent g012): Treat rot4/dia2 orbit solutions as labelled 2-factors on row classes, and run exact-delta tabu min-conflicts with 2-opt edge swaps after a shortened orbit DFS.
import time
import random
from math import gcd
import numpy as np

try:
    _pc = int.bit_count
    _pc(3)
except Exception:
    def _pc(x):
        return bin(x).count('1')


class _Abort(Exception):
    pass


def _maps(sym, n):
    m = n - 1
    r4 = [lambda r, c: (r, c), lambda r, c: (c, m - r),
          lambda r, c: (m - r, m - c), lambda r, c: (m - c, r)]
    refl = [lambda r, c: (c, r), lambda r, c: (m - c, m - r),
            lambda r, c: (m - r, c), lambda r, c: (r, m - c)]
    if sym == 'rot4':
        return r4
    if sym == 'dia2':
        return [r4[0], refl[0], refl[1], r4[2]]
    return r4 + refl


def _build(n, sym):
    maps = _maps(sym, n)
    N2 = n * n
    orb = [-1] * N2
    cellsof = []
    for p in range(N2):
        if orb[p] >= 0:
            continue
        r, c = divmod(p, n)
        s = set()
        for f in maps:
            a, b = f(r, c)
            s.add(a * n + b)
        k = len(cellsof)
        for q in s:
            orb[q] = k
        cellsof.append(sorted(s))
    norb = len(cellsof)

    def uimg(f, u):
        if u < n:
            a = f(u, 0)
            b = f(u, 1)
        else:
            a = f(0, u - n)
            b = f(1, u - n)
        return a[0] if a[0] == b[0] else n + a[1]

    ucl = [-1] * (2 * n)
    reps = []
    for u in range(2 * n):
        if ucl[u] >= 0:
            continue
        k = len(reps)
        reps.append(u)
        for f in maps:
            ucl[uimg(f, u)] = k
    H = len(reps)
    contrib = []
    invalid = 0
    for o in range(norb):
        cnt = {}
        for q in cellsof[o]:
            r, c = divmod(q, n)
            k = ucl[r]
            if reps[k] == r:
                cnt[k] = cnt.get(k, 0) + 1
            k = ucl[n + c]
            if reps[k] == n + c:
                cnt[k] = cnt.get(k, 0) + 1
        lst = list(cnt.items())
        if any(v > 2 for _, v in lst):
            invalid |= 1 << o
        contrib.append(lst)
    umask = [0] * H
    c2 = [0] * H
    for o, lst in enumerate(contrib):
        for k, v in lst:
            umask[k] |= 1 << o
            if v >= 2:
                c2[k] |= 1 << o
    S = max(len(x) for x in cellsof)
    Q = np.full((norb, S), -1, dtype=np.int64)
    for o in range(norb):
        for j, q in enumerate(cellsof[o]):
            Q[o, j] = q
    Vm = (Q >= 0)
    Vm8 = Vm.astype(np.int8)
    orbarr = np.array(orb, dtype=np.int64)
    idx = np.arange(N2, dtype=np.int64)
    R = idx // n
    Cc = idx % n
    M = [[0] * norb for _ in range(norb)]
    w0 = (norb + 7) // 8
    Pk = np.zeros((norb, norb, w0), dtype=np.uint8)
    jj = np.arange(S)[None, :]
    for o1 in range(norb):
        p1 = cellsof[o1][0]
        r1, c1 = divmod(p1, n)
        DX = R - r1
        DY = Cc - c1
        G = np.gcd(DX, DY)
        G[p1] = 1
        A = DX // G
        B = DY // G
        neg = (A < 0) | ((A == 0) & (B < 0))
        A = np.where(neg, -A, A)
        B = np.where(neg, -B, B)
        key = A * (2 * n + 1) + (B + n)
        key[p1] = -1
        uk, kidx = np.unique(key, return_inverse=True)
        kidx = kidx.ravel()
        nk = len(uk)
        Cnt = np.bincount(kidx * norb + orbarr, minlength=nk * norb).astype(np.int8).reshape(nk, norb)
        Cnt[kidx[p1]] = 0
        sub = Q[o1:]
        m = sub.shape[0]
        Qc = np.where(sub >= 0, sub, p1)
        kq = kidx[Qc]
        Rr = Cnt[kq]
        ii = np.arange(m)[:, None]
        kk = (o1 + np.arange(m))[:, None]
        Rr[ii, jj, kk] -= Vm8[o1:]
        Rr[0, 0, o1] += 1
        Bm = (Rr > 0).any(axis=1)
        pk = np.packbits(Bm, axis=1, bitorder='little')
        Pk[o1, o1:o1 + m] = pk
        Pk[o1:, o1] = pk
        w = pk.shape[1]
        bs = pk.tobytes()
        row = M[o1]
        for t in range(m):
            v = int.from_bytes(bs[t * w:(t + 1) * w], 'little')
            o2 = o1 + t
            row[o2] = v
            M[o2][o1] = v
    selfhit = [0] * norb
    for X in range(norb):
        mm = M[X][X]
        if (mm >> X) & 1:
            invalid |= 1 << X
        while mm:
            l = mm & -mm
            b = l.bit_length() - 1
            mm ^= l
            selfhit[b] |= 1 << X
    ALL = (1 << norb) - 1
    return {'M': M, 'selfhit': selfhit, 'contrib': contrib,
            'size': [len(x) for x in cellsof], 'cells': cellsof,
            'umask': umask, 'c2': c2, 'H': H, 'free0': ALL & ~invalid,
            'sym': sym, 'Pk': Pk}


def _dfs(D, rng, budget, deadline, best, noise):
    M = D['M']
    selfhit = D['selfhit']
    contrib = D['contrib']
    size = D['size']
    cells = D['cells']
    umask = D['umask']
    c2 = D['c2']
    H = D['H']
    deficit = [2] * H
    used = []
    cnt = [0]
    rnd = rng.random
    pc = _pc
    tm = time.time

    def rec(free, pts):
        cnt[0] += 1
        if cnt[0] > budget:
            raise _Abort()
        if not (cnt[0] & 511) and tm() > deadline:
            raise _Abort()
        if pts > best[0]:
            best[0] = pts
            fl = []
            for o in used:
                fl.extend(cells[o])
            best[1] = fl
        bu = -1
        bc = 1 << 30
        ba = 0
        for u in range(H):
            d = deficit[u]
            if d:
                a = umask[u] & free
                if not a:
                    return None
                c = pc(a)
                if d == 2:
                    if c == 1 and not (a & c2[u]):
                        return None
                    c = 2 * c - 1
                else:
                    c = 2 * c
                if c < bc or (c == bc and rnd() < 0.5):
                    bc = c
                    bu = u
                    ba = a
        if bu < 0:
            return list(used)
        cands = []
        a = ba
        while a:
            l = a & -a
            O = l.bit_length() - 1
            a ^= l
            Mo = M[O]
            G = Mo[O] | selfhit[O] | l
            for U in used:
                G |= Mo[U]
            nf = free & ~G
            for (u, k) in contrib[O]:
                dn = deficit[u] - k
                if dn == 0:
                    nf &= ~umask[u]
                elif dn == 1:
                    nf &= ~c2[u]
            cands.append((pc(nf) + rnd() * noise, O, nf))
        cands.sort(reverse=True)
        excl = 0
        for sc, O, nf in cands:
            cl = contrib[O]
            for (u, k) in cl:
                deficit[u] -= k
            used.append(O)
            r = rec(nf & ~excl, pts + size[O])
            used.pop()
            for (u, k) in cl:
                deficit[u] += k
            if r is not None:
                return r
            excl |= 1 << O
        return None

    return rec(D['free0'], 0)


def _luby(i):
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i -= (1 << (k - 1)) - 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1


def _prep(D):
    if 'ends' in D:
        return
    cells = D['cells']
    norb = len(cells)
    H = D['H']
    W = H
    free0 = D['free0']
    ends = [None] * norb
    pl = {}
    for o in range(norb):
        if not (free0 >> o) & 1:
            continue
        cl = D['contrib'][o]
        if len(cl) == 2:
            if cl[0][1] != 1 or cl[1][1] != 1:
                continue
            u = cl[0][0]
            v = cl[1][0]
            key = (u, v) if u < v else (v, u)
        elif len(cl) == 1:
            u, k = cl[0]
            if k == 2:
                key = (u, u)
            elif k == 1:
                key = (u, W)
            else:
                continue
        else:
            continue
        ends[o] = key
        pl.setdefault(key, []).append(o)
    Pk = D['Pk']
    ar = np.arange(norb)
    Ss = np.unpackbits(Pk[ar, ar], axis=1, bitorder='little')[:, :norb].astype(np.int32)
    D['ends'] = ends
    D['pl'] = pl
    D['Ss'] = Ss
    D['SsT'] = np.ascontiguousarray(Ss.T)
    D['selfm'] = [D['M'][o][o] for o in range(norb)]
    D['hasw'] = any(k[1] == W for k in pl)


def _ls_run(D, rng, deadline, gb, stag, maxk):
    M = D['M']
    Pk = D['Pk']
    ends = D['ends']
    pl = D['pl']
    Ss = D['Ss']
    SsT = D['SsT']
    selfm = D['selfm']
    cells = D['cells']
    norb = len(cells)
    H = D['H']
    W = H
    pc = _pc
    rnd = rng.random

    def delta(A, Sl):
        if Sl:
            ix = np.array(Sl, dtype=np.int64)
            Rm = np.unpackbits(Pk[A, ix], axis=1, bitorder='little')[:, :norb]
            dv = Rm.sum(axis=0, dtype=np.int32)
            dv[ix] -= Rm[np.arange(len(Sl)), ix]
            dv += Ss[A]
            dv += SsT[A]
        else:
            dv = Ss[A] + SsT[A]
        dv[A] = 0
        return dv

    S0 = None
    for _ in range(60):
        nodes = list(range(H))
        if D['hasw'] and rnd() < 0.5:
            nodes.append(W)
        rng.shuffle(nodes)
        lst = []
        ok = True
        L = len(nodes)
        for i in range(L):
            u = nodes[i]
            v = nodes[(i + 1) % L]
            key = (u, v) if u <= v else (v, u)
            ch = [o for o in pl.get(key, ()) if o not in lst]
            if not ch:
                ok = False
                break
            lst.append(ch[int(rnd() * len(ch))])
        if ok:
            S0 = lst
            break
    if S0 is None:
        return None

    S = []
    inS = [False] * norb
    F = np.zeros(norb, dtype=np.int32)
    Tot = 0
    smask = 0

    def add(A):
        nonlocal F, Tot, smask
        dv = delta(A, S)
        Tot += int(F[A])
        F += dv
        S.append(A)
        inS[A] = True
        smask |= 1 << A

    def rem(A):
        nonlocal F, Tot, smask
        S.remove(A)
        inS[A] = False
        smask ^= 1 << A
        dv = delta(A, S)
        Tot -= int(F[A])
        F -= dv

    def reset(lst):
        nonlocal F, Tot, smask
        for A in S:
            inS[A] = False
        del S[:]
        F = np.zeros(norb, dtype=np.int32)
        Tot = 0
        smask = 0
        for A in lst:
            add(A)

    def kick():
        for _ in range(40):
            X = S[int(rnd() * len(S))]
            Z = S[int(rnd() * len(S))]
            if X == Z:
                continue
            a, b = ends[X]
            c, e = ends[Z]
            if rnd() < 0.5:
                c, e = e, c
            k1 = (a, c) if a <= c else (c, a)
            k2 = (b, e) if b <= e else (e, b)
            L1 = [y for y in pl.get(k1, ()) if (not inS[y]) or y == X or y == Z]
            L2 = [y for y in pl.get(k2, ()) if (not inS[y]) or y == X or y == Z]
            if not L1 or not L2:
                continue
            Y1 = L1[int(rnd() * len(L1))]
            L2 = [y for y in L2 if y != Y1]
            if not L2:
                continue
            Y2 = L2[int(rnd() * len(L2))]
            if (Y1 == X and Y2 == Z) or (Y1 == Z and Y2 == X):
                continue
            rem(X)
            rem(Z)
            add(Y1)
            add(Y2)
            return

    reset(S0)
    best_run = Tot
    bestS = list(S)
    if Tot < gb[0]:
        gb[0] = Tot
        gb[1] = [q for o in S for q in cells[o]]
    tabu = [0] * norb
    it = 0
    lastimp = 0
    kicks = 0
    while True:
        if Tot == 0:
            return list(S)
        it += 1
        if (it & 7) == 0 and time.time() > deadline:
            return None
        Fl = F.tolist()
        conf = [o for o in S if Fl[o] > 0]
        if not conf:
            return list(S)
        X = conf[int(rnd() * len(conf))]
        Sl = [o for o in S if o != X]
        F1 = F - delta(X, Sl)
        F1l = F1.tolist()
        s1 = smask ^ (1 << X)
        a, b = ends[X]
        base = -Fl[X]
        asp = best_run - Tot
        rw = rnd() < 0.03
        bd = 1 << 30
        bm = []
        for Y in pl[ends[X]]:
            if Y == X or inS[Y]:
                continue
            dl = base + F1l[Y]
            if rw:
                dl = rnd()
            elif tabu[Y] > it and dl >= asp:
                continue
            if dl < bd:
                bd = dl
                bm = [(-1, Y, -1)]
            elif dl == bd:
                bm.append((-1, Y, -1))
        for Z in Sl:
            c, e = ends[Z]
            s1z = s1 ^ (1 << Z)
            MZ = M[Z]
            smZ = selfm[Z]
            base2 = base - F1l[Z]
            for opt in (0, 1):
                if opt == 0:
                    u1, v1, u2, v2 = a, c, b, e
                else:
                    if c == e or a == b:
                        break
                    u1, v1, u2, v2 = a, e, b, c
                L1 = pl.get((u1, v1) if u1 <= v1 else (v1, u1))
                if not L1:
                    continue
                L2 = pl.get((u2, v2) if u2 <= v2 else (v2, u2))
                if not L2:
                    continue
                for Y1 in L1:
                    if inS[Y1] and Y1 != X and Y1 != Z:
                        continue
                    if Y1 == Z:
                        cc1 = F1l[Z]
                    else:
                        cc1 = F1l[Y1] - pc(MZ[Y1] & s1z) - ((smZ >> Y1) & 1) - ((selfm[Y1] >> Z) & 1)
                    M1 = M[Y1]
                    sm1 = selfm[Y1]
                    tb1 = tabu[Y1] > it
                    for Y2 in L2:
                        if Y2 == Y1:
                            continue
                        if inS[Y2] and Y2 != X and Y2 != Z:
                            continue
                        if (Y1 == X and Y2 == Z) or (Y1 == Z and Y2 == X):
                            continue
                        if Y2 == Z:
                            cc2 = F1l[Z]
                        else:
                            cc2 = F1l[Y2] - pc(MZ[Y2] & s1z) - ((smZ >> Y2) & 1) - ((selfm[Y2] >> Z) & 1)
                        cc2 += pc(M1[Y2] & s1z) + ((sm1 >> Y2) & 1) + ((selfm[Y2] >> Y1) & 1)
                        dl = base2 + cc1 + cc2
                        if rw:
                            dl = rnd()
                        elif (tb1 or tabu[Y2] > it) and dl >= asp:
                            continue
                        if dl < bd:
                            bd = dl
                            bm = [(Z, Y1, Y2)]
                        elif dl == bd:
                            bm.append((Z, Y1, Y2))
        if not bm:
            continue
        Zm, Y1, Y2 = bm[int(rnd() * len(bm))]
        S.remove(X)
        inS[X] = False
        smask = s1
        Tot -= Fl[X]
        F = F1
        tabu[X] = it + 3 + int(rnd() * 8)
        if Zm >= 0:
            rem(Zm)
            tabu[Zm] = it + 3 + int(rnd() * 8)
            add(Y1)
            add(Y2)
        else:
            add(Y1)
        if Tot < best_run:
            best_run = Tot
            bestS = list(S)
            lastimp = it
            if Tot < gb[0]:
                gb[0] = Tot
                gb[1] = [q for o in S for q in cells[o]]
        elif it - lastimp > stag:
            kicks += 1
            if kicks > maxk:
                return None
            if rnd() < 0.5:
                reset(bestS)
            for _ in range(2 + int(rnd() * 3)):
                kick()
            lastimp = it


def _tri(n, pts):
    res = []
    for i, p in enumerate(pts):
        pr, pq = divmod(p, n)
        dd = {}
        for j, q in enumerate(pts):
            if j == i:
                continue
            dx = q // n - pr
            dy = q % n - pq
            g = gcd(dx, dy)
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            k = dx * 4096 + dy
            dd[k] = dd.get(k, 0) + 1
        res.append(sum(v * (v - 1) // 2 for v in dd.values()))
    return res


def _repair(n, pts, rng):
    pts = list(dict.fromkeys(pts))
    while len(pts) > 2:
        t = _tri(n, pts)
        mx = max(t)
        if mx == 0:
            break
        w = [i for i, v in enumerate(t) if v == mx]
        pts.pop(w[rng.randrange(len(w))])
    return pts


def _fill(n, pts, rng, deadline):
    pts = list(pts)
    occ = set(pts)
    cand = [p for p in range(n * n) if p not in occ]
    rng.shuffle(cand)
    rc = [0] * n
    cc = [0] * n
    for p in pts:
        rc[p // n] += 1
        cc[p % n] += 1
    for z in cand:
        if time.time() > deadline:
            break
        zr, zc = divmod(z, n)
        if rc[zr] >= 2 or cc[zc] >= 2:
            continue
        seen = set()
        ok = True
        for q in pts:
            dx = q // n - zr
            dy = q % n - zc
            g = gcd(dx, dy)
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            kk = dx * 4096 + dy
            if kk in seen:
                ok = False
                break
            seen.add(kk)
        if ok:
            pts.append(z)
            rc[zr] += 1
            cc[zc] += 1
    return pts


def _solve(n, t0, T, rng, best):
    end = t0 + T - 1.0
    t_dfs = t0 + 0.3 * T
    classes = []
    for sym in ('rot4', 'dia2'):
        if time.time() > t0 + 0.25 * T:
            break
        classes.append(_build(n, sym))
    active = list(classes)
    i = 0
    base = 150
    while active and time.time() < t_dfs:
        D = active[i % len(active)]
        budget = base * _luby(i // len(active) + 1)
        noise = rng.choice([1.0, 3.0, 6.0, 12.0])
        i += 1
        try:
            r = _dfs(D, rng, budget, t_dfs, best, noise)
            if r is not None:
                return [q for o in r for q in D['cells'][o]]
            active.remove(D)
        except _Abort:
            pass
    gb = [1 << 30, None]
    k = 0
    H = classes[0]['H'] if classes else 1
    stag = 400 + 25 * H
    while classes and time.time() < end - 0.4:
        D = classes[k % len(classes)]
        k += 1
        try:
            _prep(D)
            r = _ls_run(D, rng, end - 0.4, gb, stag, 8)
        except Exception:
            r = None
        if r is not None:
            pts = [q for o in r for q in D['cells'][o]]
            if pts and max(_tri(n, pts)) == 0:
                return pts
    cands = []
    seed = list(best[1])
    if gb[1]:
        try:
            V = _repair(n, gb[1], rng)
            V = _fill(n, V, rng, t0 + T - 0.45)
            cands.append(V)
        except Exception:
            pass
    cands.append(_fill(n, seed, rng, t0 + T - 0.3))
    return max(cands, key=len)


def construct(n, time_limit, seed):
    t0 = time.time()
    rng = random.Random(seed)
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if n == 3:
        return [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
    best = [0, []]
    res = None
    if n <= 64:
        try:
            res = _solve(n, t0, time_limit, rng, best)
        except Exception:
            res = None
    if res is None:
        base = best[1] if best[1] else []
        try:
            res = _fill(n, base, rng, t0 + time_limit - 0.25)
        except Exception:
            res = base
    seen = set()
    out = []
    for c in res:
        c = int(c)
        if c in seen:
            continue
        seen.add(c)
        out.append((c // n, c % n))
    return out
