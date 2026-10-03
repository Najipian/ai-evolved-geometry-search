# g004 (parent g002): Least-constraining value ordering with noise, plus a pruning check that enough orbits remain. Alternates with the plain shuffled-order restarts as a hedge.
# g004 (parent g002): least-constraining-value ordering + orbit-count pruning, alternated with random ordering.
import random
import time
from math import gcd
from itertools import combinations


class _Out(Exception):
    pass


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit * 0.94 - 0.07
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

        def mk(t, p, k):
            key = linekey(t, p)
            fm = fullline(key)
            if popcount(fm & orbfull[t]) > 1:
                v = 0
            else:
                v = proj(key)
            pc[k] = v
            return v

        csum = [sum(cn for _, cn in contrib[r]) for r in range(R)]
        maxc = max(csum) if csum else 1
        return {
            'K': K, 'cross': cross, 'dbl': dbl, 'ncross': [~m for m in cross],
            'ndbl': [~m for m in dbl], 'contrib': contrib, 'members': members,
            'repcell': repcell, 'selfm': selfm, 'pc': pc, 'mk': mk,
            'init': (1 << R) - 1, 'csum': csum, 'maxc': max(1, maxc),
        }

    bestp = [[]]

    def run(S, budget, lcv, noise):
        K = S['K']
        cross = S['cross']
        dbl = S['dbl']
        ncross = S['ncross']
        ndbl = S['ndbl']
        contrib = S['contrib']
        members = S['members']
        repcell = S['repcell']
        selfm = S['selfm']
        csum = S['csum']
        maxc = S['maxc']
        pc = S['pc']
        pg = pc.get
        mk = S['mk']
        need = [2] * K
        needsum = [2 * K]
        pts = []
        placed = []
        cnt = [0]
        rnd = rng.random

        def place(r, a):
            t = repcell[r]
            tN = t * NN
            for p in pts:
                k = tN + p
                v = pg(k)
                if v is None:
                    v = mk(t, p, k)
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
            needsum[0] -= csum[r]
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
                needsum[0] += csum[r]
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

        def order(cand, a, L):
            sc = []
            for rr in cand:
                a2 = place(rr, a)
                if a2 is not None:
                    ns = needsum[0]
                    pcnt = popcount(a2)
                    if pcnt * maxc >= ns:
                        sc.append((pcnt + rnd() * noise, rr))
                undo(L)
            sc.sort(reverse=True)
            return [rr for _, rr in sc]

        def solve(a):
            cnt[0] += 1
            if cnt[0] > budget or ((cnt[0] & 63) == 0 and time.time() > deadline):
                raise _Out
            if needsum[0] and popcount(a) * maxc < needsum[0]:
                return False
            r = pick(a)
            if r is None:
                return False
            if not r:
                return True
            idx, nd, m = r
            cand = bits(m)
            L = len(placed)
            if lcv:
                cand = order(cand, a, L)
            else:
                rng.shuffle(cand)
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
                        rest = [r2 for r2 in cand[i + 1:] if (a1 >> r2) & 1]
                        if lcv and len(rest) > 1:
                            rest = order(rest, a1, L1)
                        for r2 in rest:
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
        schedule = [('rot4', True), ('rot4', False), ('dia2', True), ('dia2', False)]
    else:
        schedule = [('rot2', True), ('rot2', False), ('dia2', True), ('rot4', True)]
    states = {}
    budgets = {s: 150 for s in schedule}
    attempt = 0
    while time.time() < deadline:
        key = schedule[attempt % len(schedule)]
        sym, lcv = key
        attempt += 1
        if sym not in states:
            states[sym] = build(sym)
        noise = rng.choice([0.5, 2.0, 4.0, 8.0])
        ok, pts = run(states[sym], budgets[key], lcv, noise)
        if ok and len(pts) == 2 * n:
            return [(XS[j], YS[j]) for j in pts]
        budgets[key] = int(budgets[key] * 1.3) + 30

    # Fallback: deepest valid partial + greedy extension without symmetry.
    pts = list(bestp[0])
    a = (1 << NN) - 1
    for i in range(len(pts)):
        a &= ~(1 << pts[i])
        for j in range(i):
            a &= ~fullline(linekey(pts[i], pts[j]))
    order_ = list(range(NN))
    rng.shuffle(order_)
    for cell in order_:
        if (a >> cell) & 1:
            for p in pts:
                a &= ~fullline(linekey(cell, p))
            a &= ~(1 << cell)
            pts.append(cell)
    return [(XS[j], YS[j]) for j in pts]
