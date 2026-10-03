# g001 (parent s1): Use bitboard backtracking with quarter-turn and diagonal-mirror symmetry for even n, cached line masks, and random restarts with growing budgets.
import random
import time
from math import gcd


class _Out(Exception):
    pass


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + time_limit * 0.9 - 0.05
    rng = random.Random(seed)
    NN = n * n
    c = n - 1
    RM = (1 << n) - 1
    XS = [i % n for i in range(NN)]
    YS = [i // n for i in range(NN)]
    bitv = [1 << i for i in range(NN)]
    negbit = [~(1 << i) for i in range(NN)]
    negrow = [~(RM << (y * n)) for y in range(n)]
    colm = [sum(1 << (y * n + x) for y in range(n)) for x in range(n)]
    negcol = [~m for m in colm]
    FULL = (1 << NN) - 1
    try:
        (5).bit_count()
        popcount = int.bit_count
    except AttributeError:
        def popcount(v):
            return bin(v).count("1")

    pc = {}
    lc = {}

    def mkline(p, q, k):
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
        key = (dx, dy, x, y)
        v = lc.get(key)
        if v is None:
            m = 0
            while 0 <= x < n and 0 <= y < n:
                m |= bitv[y * n + x]
                x += dx
                y += dy
            v = ~m
            lc[key] = v
        pc[k] = v
        return v

    maps = {
        'rot4': [lambda x, y: (x, y), lambda x, y: (c - y, x),
                 lambda x, y: (c - x, c - y), lambda x, y: (y, c - x)],
        'dia2': [lambda x, y: (x, y), lambda x, y: (y, x),
                 lambda x, y: (c - y, c - x), lambda x, y: (c - x, c - y)],
        'rot2': [lambda x, y: (x, y), lambda x, y: (c - x, c - y)],
        'none': [lambda x, y: (x, y)],
    }
    orbcache = {}

    def get_orbits(name):
        if name in orbcache:
            return orbcache[name]
        fs = maps[name]
        orb = []
        for i in range(NN):
            x, y = XS[i], YS[i]
            s = set()
            for f in fs:
                u, v = f(x, y)
                s.add(v * n + u)
            orb.append(sorted(s))
        orbcache[name] = orb
        return orb

    pts = []
    rc = [0] * n
    cc = [0] * n
    occ = bytearray(NN)
    cur = {'orb': None, 'cols': False}
    best = [[]]
    cnt = [0, 0]  # nodes, budget
    DONE = ()

    def add(j, a):
        pg = pc.get
        for q in pts:
            k = q * NN + j if q < j else j * NN + q
            v = pg(k)
            if v is None:
                v = mkline(q, j, k)
            a &= v
        a &= negbit[j]
        y = YS[j]
        x = XS[j]
        rc[y] += 1
        if rc[y] == 2:
            a &= negrow[y]
        cc[x] += 1
        if cc[x] == 2:
            a &= negcol[x]
        pts.append(j)
        occ[j] = 1
        return a

    def place(i, a):
        if occ[i]:
            return a
        for j in cur['orb'][i]:
            if occ[j]:
                continue
            if not (a & bitv[j]):
                return None
            a = add(j, a)
        if len(pts) > len(best[0]):
            best[0] = list(pts)
        return a

    def undo(L):
        while len(pts) > L:
            j = pts.pop()
            rc[YS[j]] -= 1
            cc[XS[j]] -= 1
            occ[j] = 0

    def pick(a):
        bst = None
        bs = 1 << 30
        for y in range(n):
            need = 2 - rc[y]
            if need:
                bits = (a >> (y * n)) & RM
                f = popcount(bits)
                if f < need:
                    return None
                s = f if need == 1 else f * (f - 1) // 2
                if s < bs:
                    bs = s
                    bst = (y, need, bits)
        if bst is None:
            return DONE
        if cur['cols']:
            for x in range(n):
                need = 2 - cc[x]
                if need and popcount(a & colm[x]) < need:
                    return None
        return bst

    def solve(a):
        cnt[0] += 1
        if cnt[0] > cnt[1] or ((cnt[0] & 63) == 0 and time.time() > deadline):
            raise _Out
        r = pick(a)
        if r is None:
            return False
        if r is DONE:
            return True
        y, need, bits = r
        cand = [x for x in range(n) if (bits >> x) & 1]
        rng.shuffle(cand)
        L = len(pts)
        base = y * n
        if need == 1:
            for x in cand:
                a2 = place(base + x, a)
                if a2 is not None and solve(a2):
                    return True
                undo(L)
            return False
        for i, x in enumerate(cand):
            a1 = place(base + x, a)
            if a1 is not None:
                if rc[y] >= 2:
                    if solve(a1):
                        return True
                elif pick(a1) is not None:
                    L1 = len(pts)
                    for x2 in cand[i + 1:]:
                        j = base + x2
                        if not (a1 & bitv[j]):
                            continue
                        a2 = place(j, a1)
                        if a2 is not None and solve(a2):
                            return True
                        undo(L1)
            undo(L)
        return False

    def reset():
        del pts[:]
        for i in range(n):
            rc[i] = 0
            cc[i] = 0
        for i in range(NN):
            occ[i] = 0

    if n % 2 == 0:
        schedule = ['rot4', 'dia2', 'rot4', 'dia2', 'rot2']
    else:
        schedule = ['rot2', 'dia2', 'rot2', 'rot4', 'dia2']
    budgets = {'rot4': 150, 'dia2': 150, 'rot2': 150, 'none': 150}
    attempt = 0
    while time.time() < deadline:
        sym = schedule[attempt % len(schedule)]
        attempt += 1
        cur['orb'] = get_orbits(sym)
        cur['cols'] = sym in ('rot2', 'none')
        reset()
        cnt[0] = 0
        cnt[1] = budgets[sym]
        try:
            if solve(FULL):
                return [(XS[j], YS[j]) for j in pts]
        except _Out:
            pass
        budgets[sym] = int(budgets[sym] * 1.25) + 50

    # Fallback: rebuild deepest partial, then greedily extend without symmetry.
    reset()
    a = FULL
    for j in best[0]:
        if a & bitv[j]:
            a = add(j, a)
    order = list(range(NN))
    rng.shuffle(order)
    changed = True
    while changed:
        changed = False
        for j in order:
            if a & bitv[j]:
                a = add(j, a)
                changed = True
    return [(XS[j], YS[j]) for j in pts]
