# g002 (parent g001): Replace backtracking with exact-delta numpy tabu search over symmetric orbits (quarter turn and half turn), then an unrestricted polish phase and a greedy repair.
import time
import random
import numpy as np


def _group_maps(n, name):
    m = n - 1
    idm = lambda x, y: (x, y)
    r90 = lambda x, y: (y, m - x)
    r180 = lambda x, y: (m - x, m - y)
    r270 = lambda x, y: (m - y, x)
    d1 = lambda x, y: (y, x)
    d2 = lambda x, y: (m - y, m - x)
    return {
        'rot4': [idm, r90, r180, r270],
        'dia2': [idm, r180, d1, d2],
        'rot2': [idm, r180],
        'iden': [idm],
    }[name]


def _valid(cells):
    rc, cc = {}, {}
    for (a, b) in cells:
        rc[a] = rc.get(a, 0) + 1
        cc[b] = cc.get(b, 0) + 1
        if rc[a] > 2 or cc[b] > 2:
            return False
    L = len(cells)
    for i in range(L):
        ax, ay = cells[i]
        for j in range(i + 1, L):
            bx, by = cells[j]
            for k in range(j + 1, L):
                cx, cy = cells[k]
                if (bx - ax) * (cy - ay) == (by - ay) * (cx - ax):
                    return False
    return True


def _orbits(n, name):
    maps = _group_maps(n, name)
    order = len(maps)
    seen = [False] * (n * n)
    out = []
    for x in range(n):
        for y in range(n):
            if seen[x * n + y]:
                continue
            cells = sorted(set(f(x, y) for f in maps))
            for (a, b) in cells:
                seen[a * n + b] = True
            if len(cells) == order and _valid(cells):
                out.append([a * n + b for (a, b) in cells])
    return out


def _build_lid(n):
    N2 = n * n
    idx = np.arange(N2, dtype=np.int32)
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
    C = B.astype(np.int64) * X[:, None] - A.astype(np.int64) * Y[:, None]
    key = (A.astype(np.int64) * (2 * n + 1) + (B + n)) * (4 * n * n + 1) + (C + 2 * n * n)
    del A, B, C
    _, inv = np.unique(key.ravel(), return_inverse=True)
    del key
    LID = inv.reshape(N2, N2).astype(np.int32)
    del inv
    NL = int(LID.max()) + 1
    ar = np.arange(N2)
    LID[ar, ar] = NL
    NLtot = NL + 1
    PL = [np.unique(LID[p])[:-1].astype(np.intp) for p in range(N2)]
    return LID, PL, NLtot


class _Search(object):
    def __init__(self, n, LID, PL, NLtot, orbits, rng, full=False):
        self.n = n
        self.PL = PL
        self.rng = rng
        self.nrng = np.random.default_rng(rng.randrange(1 << 30))
        self.cl = orbits
        self.cells = np.array(orbits, dtype=np.intp)
        self.N, self.s = self.cells.shape
        self.k = (2 * n) // self.s
        self.LR = LID if full else LID[self.cells[:, 0]]
        IL, IA, IO = [], [], []
        if self.s > 1:
            for o, cs in enumerate(orbits):
                d = {}
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        l = int(LID[cs[i], cs[j]])
                        d.setdefault(l, set()).update((i, j))
                for l, st in d.items():
                    IL.append(l)
                    IA.append(len(st))
                    IO.append(o)
        self.IL = np.array(IL, dtype=np.intp)
        self.IA = np.array(IA, dtype=np.int64)
        self.IO = np.array(IO, dtype=np.intp)
        self.cnt = np.zeros(NLtot, dtype=np.int32)
        L = 2 * n + 4
        F = np.zeros(L)
        Gt = np.zeros(L)
        for c in range(2, L):
            F[c] = 1.0 / c
            if c >= 3:
                Gt[c] = 1.0 / c
        self.F, self.G = F, Gt
        self.E = 0
        self.sel = []
        self.selm = np.zeros(self.N, dtype=bool)
        self.tabu = np.zeros(self.N, dtype=np.int64)
        self.it = 0
        self.bestE = 1 << 30
        self.best_sel = None
        self.lastimp = 0
        self.empty = np.zeros(0, dtype=np.intp)

    def addo(self, o):
        cnt = self.cnt
        PL = self.PL
        e = 0
        for p in self.cl[o]:
            idx = PL[p]
            v = cnt[idx]
            e += int(np.count_nonzero(v >= 2))
            cnt[idx] = v + 1
        self.E += e

    def remo(self, o):
        cnt = self.cnt
        PL = self.PL
        e = 0
        for p in self.cl[o]:
            idx = PL[p]
            v = cnt[idx]
            e += int(np.count_nonzero(v >= 3))
            cnt[idx] = v - 1
        self.E -= e

    def pts_arr(self, sel):
        if not sel:
            return self.empty
        return self.cells[np.array(sel, dtype=np.intp)].ravel()

    def batch(self, S):
        if len(S):
            M = self.cnt[self.LR[:, S]]
            d = np.rint(self.F[M].sum(axis=1)) * self.s
        else:
            d = np.zeros(self.N)
        if len(self.IL):
            c = self.cnt[self.IL]
            corr = np.where(c < 2, np.maximum(c + self.IA - 2, 0), 0)
            d = d + np.bincount(self.IO, weights=corr, minlength=self.N)
        return d

    def _rec(self):
        if self.E < self.bestE:
            self.bestE = self.E
            self.best_sel = list(self.sel)
            self.lastimp = self.it

    def init_greedy(self):
        for _ in range(self.k):
            S = self.pts_arr(self.sel)
            sc = self.batch(S) + self.nrng.random(self.N) * 0.9
            sc[self.selm] = np.inf
            o = int(np.argmin(sc))
            self.addo(o)
            self.sel.append(o)
            self.selm[o] = True
        self._rec()

    def set_state(self, sel_list):
        self.cnt[:] = 0
        self.E = 0
        self.selm[:] = False
        self.sel = []
        for o in sel_list:
            self.addo(o)
            self.sel.append(o)
            self.selm[o] = True

    def init_from(self, cells_list, n2):
        orb_of = np.full(n2, -1, dtype=np.int64)
        for o, cs in enumerate(self.cl):
            for c in cs:
                orb_of[c] = o
        sl = []
        seen = set()
        for c in cells_list:
            o = int(orb_of[c])
            if o >= 0 and o not in seen:
                seen.add(o)
                sl.append(o)
        self.set_state(sl)
        self._rec()

    def best_points(self):
        return [c for o in self.best_sel for c in self.cl[o]]

    def perturb(self):
        rng = self.rng
        if self.best_sel is not None and rng.random() < 0.5:
            self.set_state(self.best_sel)
        r = 1 + rng.randrange(2) if self.s > 1 else 2 + rng.randrange(3)
        for _ in range(r):
            if not self.sel:
                break
            j = rng.randrange(len(self.sel))
            o = self.sel[j]
            self.remo(o)
            self.selm[o] = False
            for _t in range(50):
                c = rng.randrange(self.N)
                if not self.selm[c] and c != o:
                    break
            else:
                c = o
            self.addo(c)
            self.sel[j] = c
            self.selm[c] = True
        self.lastimp = self.it
        self._rec()

    def run(self, deadline):
        rng = self.rng
        nrng = self.nrng
        cnt = self.cnt
        LR = self.LR
        G = self.G
        N = self.N
        k = self.k
        SL = max(300, 30 * k)
        tl = time.time
        while True:
            if self.E == 0:
                self._rec()
                return True
            if tl() > deadline:
                return False
            self.it += 1
            it = self.it
            sel = self.sel
            selarr = np.array(sel, dtype=np.intp)
            S = self.cells[selarr].ravel()
            M = cnt[LR[np.ix_(selarr, S)]]
            h = G[M].sum(axis=1)
            conf = np.flatnonzero(h > 1e-9)
            if len(conf) == 0:
                j = rng.randrange(len(sel))
            elif rng.random() < 0.3:
                j = int(np.argmax(h))
            else:
                j = int(conf[rng.randrange(len(conf))])
            o = sel[j]
            self.remo(o)
            sel[j] = sel[-1]
            sel.pop()
            self.selm[o] = False
            S2 = self.pts_arr(sel)
            sc = self.batch(S2)
            E0 = self.E
            notsel = ~self.selm
            ok = notsel & ((self.tabu <= it) | ((sc + E0) < self.bestE))
            ok[o] = False
            sc2 = sc + nrng.random(N) * 0.5
            sc2[~ok] = np.inf
            c = int(np.argmin(sc2))
            if not np.isfinite(sc2[c]):
                c = o
            self.addo(c)
            sel.append(c)
            self.selm[c] = True
            self.tabu[o] = it + 4 + rng.randrange(max(2, k // 2))
            if self.E < self.bestE:
                self._rec()
            elif it - self.lastimp > SL:
                self.perturb()


def _repair(n, LID, PL, NLtot, pts, rng):
    N2 = n * n
    cnt = np.zeros(NLtot, dtype=np.int32)
    S = list(dict.fromkeys(int(p) for p in pts))
    for p in S:
        cnt[PL[p]] += 1
    while len(S) > 2:
        Sa = np.array(S, dtype=np.intp)
        M = cnt[LID[np.ix_(Sa, Sa)]]
        bad = (M >= 3).sum(axis=1)
        mx = int(bad.max())
        if mx == 0:
            break
        ch = np.flatnonzero(bad == mx)
        i = int(ch[rng.randrange(len(ch))])
        p = S[i]
        cnt[PL[p]] -= 1
        S.pop(i)
    inS = np.zeros(N2, dtype=bool)
    if S:
        inS[np.array(S, dtype=np.intp)] = True
    while len(S) < 2 * n:
        if S:
            Sa = np.array(S, dtype=np.intp)
            M = cnt[LID[:, Sa]]
            blocked = (M >= 2).any(axis=1) | inS
        else:
            blocked = inS.copy()
        free = np.flatnonzero(~blocked)
        if len(free) == 0:
            break
        c = int(free[rng.randrange(len(free))])
        S.append(c)
        inS[c] = True
        cnt[PL[c]] += 1
    return S


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
    final_deadline = t0 + time_limit - 0.45
    tot = time_limit - 0.45
    gbestE = 1 << 30
    gbest_pts = None
    to_xy = lambda cl: [(int(c // n), int(c % n)) for c in cl]
    try:
        LID, PL, NLtot = _build_lid(n)
        groups = ['rot4', 'rot2'] if n % 2 == 0 else ['rot2']
        searches = {}
        sym_end = t0 + tot * 0.75
        slice_len = max(1.0, 0.12 * time_limit)
        gi = 0
        while time.time() < sym_end:
            nm = groups[gi % len(groups)]
            gi += 1
            if nm not in searches:
                orbs = _orbits(n, nm)
                if not orbs:
                    if len(groups) == 1:
                        break
                    continue
                s = _Search(n, LID, PL, NLtot, orbs, rng)
                s.init_greedy()
                searches[nm] = s
            s = searches[nm]
            s.run(min(sym_end, time.time() + slice_len))
            if s.bestE < gbestE:
                gbestE = s.bestE
                gbest_pts = s.best_points()
            if gbestE == 0:
                return to_xy(gbest_pts)
        s = _Search(n, LID, PL, NLtot, [[c] for c in range(n * n)], rng, full=True)
        if gbest_pts is not None:
            s.init_from(gbest_pts, n * n)
        else:
            s.init_greedy()
        s.run(final_deadline - 0.3)
        if s.bestE < gbestE or gbest_pts is None:
            gbestE = s.bestE
            gbest_pts = s.best_points()
        if gbestE == 0:
            return to_xy(gbest_pts)
        rep = _repair(n, LID, PL, NLtot, gbest_pts, rng)
        return to_xy(rep)
    except Exception:
        if gbest_pts is not None:
            return to_xy(gbest_pts)
        return [(i, 0) for i in range(min(n, 2))]
