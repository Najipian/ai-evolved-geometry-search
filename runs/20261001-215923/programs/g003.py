# g003 (parent g002): Cache orbit-to-orbit compatibility masks so each placement does one lookup per placed orbit instead of one per placed point, and use more of the time budget.
# g003 (parent g002): orbit-pair mask cache (one lookup per placed orbit instead of per point), slightly larger time usage.
import random
import time
from math import gcd
from itertools import combinations


class _Out(Exception):
    pass


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit * 0.94 - 0.06
    rng = random.Random(seed)
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    NN = n * n
    c = n - 1
    XS = [i % n for i in range(NN)]
    YS = [i // n for i in range(NN)]
    try:
        (5).bit_count()
        popcount = int.bit_count
    except AttributeError:
        def popcount(v):
            return bin(v).count("1")

    fullc = {}

    def linekey(p, q):
        px, py = XS[p], YS[p]
        dx = XS[q] - px
        dy = YS[q] - py
        g = gcd(dx, dy)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        x, y = px, py
        while 0 <= x - dx < n and 0 <= y - dy < n:
            x -= dx
            y -= dy
        return (dx, dy, x, y)

    def fullline(key):
        v = fullc.get(key)
        if v is None:
            dx, dy, x, y = key
            m = 0
            while 0 <= x < n and 0 <= y < n:
                m |= 1 << (y * n + x)
                x += dx
                y += dy
            fullc[key] = m
            v = m
        return v

    maps = {
        'rot4': [lambda x, y: (x, y), lambda x, y: (c - y, x),
                 lambda x, y: (c - x, c - y), lambda x, y: (y, c - x)],
        'dia2': [lambda x, y: (x, y), lambda x, y: (y, x),
                 lambda x, y: (c - y, c - x), lambda x, y: (c - x, c - y)],
        'rot2': [lambda x, y: (x, y), lambda x, y: (c - x, c - y)],
    }

    def collinear3(O):
        if len(O) < 3:
            return False
        for a, b, d in combinations(O, 3):
            ax, ay = XS[a], YS[a]
            if (XS[b] - ax) * (YS[d] - ay) - (YS[b] - ay) * (XS[d] - ax) == 0:
                return True
        return False

    def build(sym):
        fs = maps[sym]
        use_cols = (sym == 'rot2')
        Kr = len([k for k in range(n) if 2 * k <= c])
        K = Kr * (2 if use_cols else 1)
        seen = [False] * NN
        rid = [-1] * NN
        repcell = []
        members = []
        contrib = []
        orbfull = [0] * NN
        for i in range(NN):
            if seen[i]:
                continue
            s = set()
            for f in fs:
                u, v = f(XS[i], YS[i])
                s.add(v * n + u)
            O = sorted(s)
            for j in O:
                seen[j] = True
            cd = {}
            for j in O:
                y = YS[j]
                if 2 * y <= c:
                    cd[y] = cd.get(y, 0) + 1
                if use_cols:
                    x = XS[j]
                    if 2 * x <= c:
                        cd[Kr + x] = cd.get(Kr + x, 0) + 1
            if any(v > 2 for v in cd.values()):
                continue
            if collinear3(O):
                continue
            r = len(repcell)
            repcell.append(O[0])
            members.append(O)
            contrib.append(list(cd.items()))
            m = 0
            for j in O:
                rid[j] = r
                m |= 1 << j
            orbfull[O[0]] = m
        R = len(repcell)
        cross = [0] * K
        dbl = [0] * K
        for r in range(R):
            for idx, cn in contrib[r]:
                cross[idx] |= 1 << r
                if cn == 2:
                    dbl[idx] |= 1 << r
        projc = {}

        def proj(key):
            v = projc.get(key)
            if v is None:
                fm = fullline(key)
                m = 0
                while fm:
                    low = fm & -fm
                    j = low.bit_length() - 1
                    fm ^= low
                    rr = rid[j]
                    if rr >= 0:
                        m |= 1 << rr
                v = ~m
                projc[key] = v
            return v

        selfm = []
        for r in range(R):
            O = members[r]
            t = O[0]
            v = ~(1 << r)
            for q in O[1:]:
                v &= proj(linekey(t, q))
            selfm.append(v)
        pc = {}

        def mk(t, p):
            key = linekey(t, p)
            fm = fullline(key)
            if popcount(fm & orbfull[t]) > 1:
                v = 0
            else:
                v = proj(key)
            return v

        pm = {}

        def pmk(r, q, k):
            t = repcell[r]
            tN = t * NN
            v = -1
            for p in members[q]:
                kk = tN + p
                w = pc.get(kk)
                if w is None:
                    w = mk(t, p)
                    pc[kk] = w
                if not w:
                    v = 0
                    break
                v &= w
            pm[k] = v
            return v

        return {
            'K': K, 'R': R, 'cross': cross, 'dbl': dbl, 'ncross': [~m for m in cross],
            'ndbl': [~m for m in dbl], 'contrib': contrib, 'members': members,
            'repcell': repcell, 'selfm': selfm, 'pm': pm, 'pmk': pmk,
            'init': (1 << R) - 1,
        }

    bestp = [[]]

    def run(S, budget):
        K = S['K']
        R = S['R']
        cross = S['cross']
        dbl = S['dbl']
        ncross = S['ncross']
        ndbl = S['ndbl']
        contrib = S['contrib']
        members = S['members']
        selfm = S['selfm']
        pm = S['pm']
        pmg = pm.get
        pmk = S['pmk']
        need = [2] * K
        pts = []
        placed = []
        cnt = [0]

        def place(r, a):
            rR = r * R
            for q in placed:
                k = rR + q
                v = pmg(k)
                if v is None:
                    v = pmk(r, q, k)
                if not v:
                    return None
                a &= v
            a &= selfm[r]
            for idx, cn in contrib[r]:
                nd = need[idx] - cn
                need[idx] = nd
                if nd == 0:
                    a &= ncross[idx]
                elif nd == 1:
                    a &= ndbl[idx]
            pts.extend(members[r])
            placed.append(r)
            if len(pts) > len(bestp[0]):
                bestp[0] = list(pts)
            return a

        def undo(L):
            while len(placed) > L:
                r = placed.pop()
                for idx, cn in contrib[r]:
                    need[idx] += cn
                del pts[-len(members[r]):]

        def pick(a):
            bst = None
            bs = 1 << 30
            for idx in range(K):
                nd = need[idx]
                if nd:
                    m = a & cross[idx]
                    if nd == 1:
                        if not m:
                            return None
                        s = popcount(m)
                    else:
                        s = popcount(m)
                        if s < 2 and not (m & dbl[idx]):
                            return None
                        s = s * (s - 1) // 2 + 1
                    if s < bs:
                        bs = s
                        bst = (idx, nd, m)
            if bst is None:
                return ()
            return bst

        def bits(m):
            out = []
            while m:
                low = m & -m
                out.append(low.bit_length() - 1)
                m ^= low
            return out

        def solve(a):
            cnt[0] += 1
            if cnt[0] > budget or ((cnt[0] & 63) == 0 and time.time() > deadline):
                raise _Out
            r = pick(a)
            if r is None:
                return False
            if not r:
                return True
            idx, nd, m = r
            cand = bits(m)
            rng.shuffle(cand)
            L = len(placed)
            if nd == 1:
                for rr in cand:
                    a2 = place(rr, a)
                    if a2 is not None and solve(a2):
                        return True
                    undo(L)
                return False
            for i, rr in enumerate(cand):
                a1 = place(rr, a)
                if a1 is not None:
                    if need[idx] == 0:
                        if solve(a1):
                            return True
                    elif pick(a1) is not None:
                        L1 = len(placed)
                        for r2 in cand[i + 1:]:
                            if not (a1 >> r2) & 1:
                                continue
                            a2 = place(r2, a1)
                            if a2 is not None and solve(a2):
                                return True
                            undo(L1)
                undo(L)
            return False

        try:
            ok = solve(S['init'])
        except _Out:
            ok = False
        return ok, pts

    if n % 2 == 0:
        schedule = ['rot4', 'dia2']
    else:
        schedule = ['rot2', 'dia2', 'rot4']
    states = {}
    budgets = {s: 200 for s in schedule}
    attempt = 0
    while time.time() < deadline:
        sym = schedule[attempt % len(schedule)]
        attempt += 1
        if sym not in states:
            states[sym] = build(sym)
        ok, pts = run(states[sym], budgets[sym])
        if ok and len(pts) == 2 * n:
            return [(XS[j], YS[j]) for j in pts]
        budgets[sym] = int(budgets[sym] * 1.3) + 30

    # Fallback: deepest valid partial + greedy extension without symmetry.
    pts = list(bestp[0])
    a = (1 << NN) - 1
    for i in range(len(pts)):
        a &= ~(1 << pts[i])
        for j in range(i):
            a &= ~fullline(linekey(pts[i], pts[j]))
    order = list(range(NN))
    rng.shuffle(order)
    for cell in order:
        if (a >> cell) & 1:
            for p in pts:
                a &= ~fullline(linekey(cell, p))
            a &= ~(1 << cell)
            pts.append(cell)
    return [(XS[j], YS[j]) for j in pts]
