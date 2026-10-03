# g003 (parent g001): Replace backtracking with a fast full-grid annealer. It keeps two points per row and column via rook swaps, scores moves in O(1) from cached tables, then repairs greedily.
# Full-grid simulated annealing over configurations with exactly two points per row and column.
# Moves are rook swaps (r1,c1),(r2,c2) -> (r1,c2),(r2,c1), which keep the row/column counts.
# Exact O(1) move deltas come from a per-cell array V. For an empty cell, V is twice the number
# of triples created by adding it; for a placed point, V is twice the number of triples it is in.
# Pair counts per line are also cached. After each accepted move, V is updated vectorised in numpy.
import time
import random
from math import gcd, exp
from array import array
import numpy as np


def _build_lid(n):
    N2 = n * n
    idx = np.arange(N2, dtype=np.int64)
    X = idx // n
    Y = idx % n
    DX = X[None, :] - X[:, None]
    DY = Y[None, :] - Y[:, None]
    G = np.gcd(DX, DY)
    G[G == 0] = 1
    A = DX // G
    B = DY // G
    del DX, DY, G
    neg = (A < 0) | ((A == 0) & (B < 0))
    A = np.where(neg, -A, A)
    B = np.where(neg, -B, B)
    del neg
    C = B * X[:, None] - A * Y[:, None]
    key = (A * (2 * n + 1) + (B + n)) * (4 * n * n + 1) + (C + 2 * n * n)
    del A, B, C
    _, inv = np.unique(key.ravel(), return_inverse=True)
    del key
    NL = int(inv.max()) + 1
    LID = np.ascontiguousarray(inv.reshape(N2, N2).astype(np.int32))
    return LID, NL


class _St(object):
    def __init__(self, n, LID, NL):
        N2 = n * n
        self.n = n
        self.LID = LID
        self.pc = [0] * NL
        self.V = np.zeros(N2, dtype=np.int64)
        self.inS = np.zeros(N2, dtype=np.int64)
        self.occ = [False] * N2
        self.S = []
        self.pos = [-1] * N2
        self.T2 = 0
        self.scr = np.zeros(NL, dtype=np.int64)

    def _upd(self, p, sign):
        S = self.S
        if not S:
            return
        row = self.LID[p]
        Lq = row[np.array(S, dtype=np.intp)]
        scr = self.scr
        np.add.at(scr, Lq, 1)
        c = scr[row]
        c -= self.inS
        c[p] = 0
        if sign > 0:
            self.V += 2 * c
        else:
            self.V -= 2 * c
        scr[Lq] = 0
        pc = self.pc
        if sign > 0:
            for L in Lq.tolist():
                pc[L] += 1
        else:
            for L in Lq.tolist():
                pc[L] -= 1

    def add(self, p):
        self.T2 += int(self.V[p])
        self._upd(p, 1)
        self.inS[p] = 1
        self.occ[p] = True
        self.pos[p] = len(self.S)
        self.S.append(p)

    def remove(self, p):
        self.T2 -= int(self.V[p])
        S = self.S
        i = self.pos[p]
        last = S.pop()
        if last != p:
            S[i] = last
            self.pos[last] = i
        self.pos[p] = -1
        self.inS[p] = 0
        self.occ[p] = False
        self._upd(p, -1)


def _anneal(st, n, lidf, kf, rng, t_start, t_end, tau0, tau1, best):
    N2 = n * n
    S = st.S
    occ = st.occ
    pc = st.pc
    Vl = st.V.tolist()
    rnd = rng.random
    rr = rng.randrange
    m = len(S)
    conf = [p for p in S if Vl[p] > 0]
    if not conf:
        if st.T2 < best[0]:
            best[0] = st.T2
            best[1] = list(S)
        return
    tau = tau0
    acc = [exp(-d / tau) for d in range(80)]
    span = max(1e-6, t_end - t_start)
    it = 0
    while True:
        it += 1
        if not (it & 255):
            now = time.time()
            if now >= t_end:
                break
            fr = (now - t_start) / span
            if fr > 1.0:
                fr = 1.0
            tau = tau0 * (tau1 / tau0) ** fr
            acc = [exp(-d / tau) for d in range(80)]
        if conf and rnd() < 0.7:
            a = conf[rr(len(conf))]
        else:
            a = S[rr(m)]
        b = S[rr(m)]
        r1, c1 = divmod(a, n)
        r2, c2 = divmod(b, n)
        if r1 == r2 or c1 == c2:
            continue
        ap = r1 * n + c2
        bp = r2 * n + c1
        if occ[ap] or occ[bp]:
            continue
        o = ap * N2
        La1 = lidf[o + a]
        Lb1 = lidf[o + b]
        if La1 != Lb1:
            d = Vl[ap] - 2 * (kf[pc[La1]] + kf[pc[Lb1]] - 2)
        else:
            d = Vl[ap] - (4 * kf[pc[La1]] - 6)
        o2 = bp * N2
        La2 = lidf[o2 + a]
        Lb2 = lidf[o2 + b]
        if La2 != Lb2:
            d += Vl[bp] - 2 * (kf[pc[La2]] + kf[pc[Lb2]] - 2)
        else:
            d += Vl[bp] - (4 * kf[pc[La2]] - 6)
        d += 2 * (kf[pc[lidf[a * N2 + b]]] - 2) - Vl[a] - Vl[b]
        L0 = lidf[o + bp]
        p0 = pc[L0]
        if p0:
            k0 = kf[p0] - (La1 == L0) - (Lb1 == L0)
        elif La1 == L0 or Lb1 == L0:
            k0 = 0
        else:
            dx = r2 - r1
            dy = c1 - c2
            g = gcd(dx, dy)
            sx = dx // g
            sy = dy // g
            k0 = 0
            x = r1 + sx
            y = c2 + sy
            while 0 <= x < n and 0 <= y < n:
                if occ[x * n + y]:
                    k0 = 1
                    break
                x += sx
                y += sy
            if not k0:
                x = r1 - sx
                y = c2 - sy
                while 0 <= x < n and 0 <= y < n:
                    if occ[x * n + y]:
                        k0 = 1
                        break
                    x -= sx
                    y -= sy
        d += 2 * k0
        if d > 0:
            if d >= 80 or rnd() >= acc[d]:
                continue
        st.remove(a)
        st.remove(b)
        st.add(ap)
        st.add(bp)
        Vl = st.V.tolist()
        conf = [p for p in S if Vl[p] > 0]
        if st.T2 < best[0]:
            best[0] = st.T2
            best[1] = list(S)
            if st.T2 == 0:
                return


def _repair(n, LID, NL, cells, rng, deadline, tries):
    best = []
    for t in range(tries):
        if t > 0 and time.time() > deadline:
            break
        st = _St(n, LID, NL)
        for c in cells:
            st.add(c)
        while st.T2 > 0:
            sa = np.array(st.S, dtype=np.intp)
            vals = st.V[sa]
            mx = vals.max()
            ch = sa[vals == mx]
            st.remove(int(ch[rng.randrange(len(ch))]))
        while True:
            free = np.nonzero((st.V == 0) & (st.inS == 0))[0]
            if len(free) == 0:
                break
            st.add(int(free[rng.randrange(len(free))]))
        if len(st.S) > len(best):
            best = list(st.S)
        if len(best) >= 2 * n:
            break
    return best


def _random_init(n, rng):
    perm = list(range(n))
    rng.shuffle(perm)
    pts = []
    for i in range(n):
        pts.append(i * n + perm[i])
        pts.append(i * n + perm[(i + 1) % n])
    return pts


def _solve(n, t0, deadline, rng, holder):
    LID, NL = _build_lid(n)
    N2 = n * n
    lidf = array('i')
    lidf.frombytes(LID.tobytes())
    kf = [0] * (n * (n - 1) // 2 + 2)
    for k in range(2, n + 1):
        kf[k * (k - 1) // 2] = k
    kf[0] = 1
    sa_end = deadline - 0.35
    st = _St(n, LID, NL)
    for c in _random_init(n, rng):
        st.add(c)
    best = [st.T2, list(st.S)]
    holder[0] = list(st.S)
    plan = [(0.45, 4.0, 0.6, False), (0.2, 2.0, 0.5, True),
            (0.2, 2.0, 0.5, False), (0.15, 1.5, 0.4, True)]
    start = time.time()
    total = max(0.0, sa_end - start)
    tcur = start
    for frac, ta, tb, rest in plan:
        if best[0] == 0:
            break
        tend = tcur + frac * total
        if tend > sa_end:
            tend = sa_end
        if time.time() >= tend:
            continue
        if rest:
            st = _St(n, LID, NL)
            for c in best[1]:
                st.add(c)
        _anneal(st, n, lidf, kf, rng, time.time(), tend, ta, tb, best)
        holder[0] = list(best[1])
        tcur = tend
    if best[0] == 0:
        return best[1]
    res = _repair(n, LID, NL, best[1], rng, deadline - 0.1, 30)
    return res


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit - 0.5
    rng = random.Random(seed)
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if n == 3:
        return [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
    holder = [None]
    try:
        res = _solve(n, t0, deadline, rng, holder)
    except Exception:
        res = holder[0]
    if not res:
        res = _random_init(n, rng)
    seen = set()
    out = []
    for c in res:
        c = int(c)
        if c in seen:
            continue
        seen.add(c)
        out.append((c // n, c % n))
    return out
