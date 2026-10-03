# g007 (parent g001): Backtrack over symmetry orbits (quarter-turn, diagonal pair, half-turn) with cached orbit-pair line masks, row/column forward checking, sibling exclusion, and Luby restarts.
import random
import time
from math import gcd
from itertools import combinations


class _Out(Exception):
    pass


def _luby(i):
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    if i == (1 << k) - 1:
        return 1 << (k - 1)
    return _luby(i - (1 << (k - 1)) + 1)


def construct(n, time_limit, seed):
    t0 = time.time()
    deadline = t0 + max(0.05, time_limit * 0.92 - 0.1)
    rng = random.Random(seed)
    if n <= 2:
        return [(x, y) for y in range(n) for x in range(n)]
    c = n - 1
    NN = n * n
    try:
        (5).bit_count()
        popcount = int.bit_count
    except AttributeError:
        def popcount(v):
            return bin(v).count("1")

    maps = {
        'rot4': [lambda x, y: (x, y), lambda x, y: (c - y, x),
                 lambda x, y: (c - x, c - y), lambda x, y: (y, c - x)],
        'dia2': [lambda x, y: (x, y), lambda x, y: (y, x),
                 lambda x, y: (c - y, c - x), lambda x, y: (c - x, c - y)],
        'rot2': [lambda x, y: (x, y), lambda x, y: (c - x, c - y)],
    }

    def coll(a, b, d):
        return ((b % n - a % n) * (d // n - a // n)
                - (b // n - a // n) * (d % n - a % n)) == 0

    structs = {}

    def build(name):
        fs = maps[name]
        cell_orb = [-1] * NN
        orbs = []
        for i in range(NN):
            if cell_orb[i] >= 0:
                continue
            x, y = i % n, i // n
            s = set()
            for f in fs:
                u, v = f(x, y)
                s.add(v * n + u)
            o = len(orbs)
            for j in s:
                cell_orb[j] = o
            orbs.append(sorted(s))
        NO = len(orbs)
        valid = 0
        for o, cells in enumerate(orbs):
            rcnt = {}
            ccnt = {}
            for j in cells:
                rcnt[j // n] = rcnt.get(j // n, 0) + 1
                ccnt[j % n] = ccnt.get(j % n, 0) + 1
            ok = max(rcnt.values()) <= 2 and max(ccnt.values()) <= 2
            if ok:
                for a, b, d in combinations(cells, 3):
                    if coll(a, b, d):
                        ok = False
                        break
            if ok:
                valid |= 1 << o
        sig_index = {}
        m1 = []
        m2 = []
        contrib = [[] for _ in range(NO)]
        for kind in (0, 1):
            for t in range(n):
                cnt = {}
                for s in range(n):
                    j = t * n + s if kind == 0 else s * n + t
                    o = cell_orb[j]
                    cnt[o] = cnt.get(o, 0) + 1
                sig = tuple(sorted(cnt.items()))
                if sig in sig_index:
                    continue
                li = len(m1)
                sig_index[sig] = li
                a = b = 0
                for o, k in cnt.items():
                    if not (valid >> o) & 1:
                        continue
                    if k == 1:
                        a |= 1 << o
                    else:
                        b |= 1 << o
                    contrib[o].append((li, k))
                m1.append(a)
                m2.append(b)
        mall = [m1[i] | m2[i] for i in range(len(m1))]
        lcache = {}

        def lmask(a, b):
            ax = a % n
            ay = a // n
            dx = b % n - ax
            dy = b // n - ay
            g = gcd(dx, dy)
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            x, y = ax, ay
            while 0 <= x - dx < n and 0 <= y - dy < n:
                x -= dx
                y -= dy
            key = (dx, dy, x, y)
            v = lcache.get(key)
            if v is None:
                v = 0
                while 0 <= x < n and 0 <= y < n:
                    v |= 1 << cell_orb[y * n + x]
                    x += dx
                    y += dy
                lcache[key] = v
            return v

        Fc = {}

        def F(o, q):
            if o > q:
                o, q = q, o
            k = o * NO + q
            v = Fc.get(k)
            if v is None:
                a0 = orbs[o][0]
                v = 1 << o
                for b in orbs[q]:
                    if b != a0:
                        v |= lmask(a0, b)
                Fc[k] = v
            return v

        S = [0] * NO
        vv = valid
        while vv:
            low = vv & -vv
            o = low.bit_length() - 1
            vv ^= low
            f = F(o, o)
            while f:
                lw = f & -f
                q = lw.bit_length() - 1
                f ^= lw
                S[q] |= 1 << o
        return {'orbs': orbs, 'valid': valid, 'm1': m1, 'm2': m2,
                'mall': mall, 'contrib': contrib, 'F': F, 'S': S,
                'L': len(m1)}

    best = [[]]

    def run(st, budget):
        orbs = st['orbs']
        m1 = st['m1']
        m2 = st['m2']
        mall = st['mall']
        contrib = st['contrib']
        F = st['F']
        S = st['S']
        L = st['L']
        rc = [0] * L
        P = []
        cnt = [0]
        rlist = list(range(L))

        def solve(A, npts):
            cnt[0] += 1
            if cnt[0] > budget:
                raise _Out
            if (cnt[0] & 127) == 0 and time.time() > deadline:
                raise _Out
            if npts > len(best[0]):
                best[0] = [j for o in P for j in orbs[o]]
            bl = -1
            bc = 1 << 30
            for li in rlist:
                need = 2 - rc[li]
                if need == 0:
                    continue
                a1 = A & m1[li]
                if need == 1:
                    if not a1:
                        return False
                    k = popcount(a1)
                else:
                    k2 = popcount(A & m2[li])
                    k1 = popcount(a1)
                    if k2 == 0 and k1 < 2:
                        return False
                    k = k1 + k2
                if k < bc:
                    bc = k
                    bl = li
            if bl < 0:
                return True
            li = bl
            need = 2 - rc[li]
            cm = A & (mall[li] if need == 2 else m1[li])
            cand = []
            while cm:
                low = cm & -cm
                cand.append(low.bit_length() - 1)
                cm ^= low
            rng.shuffle(cand)
            for o in cand:
                forb = S[o] | F(o, o)
                for q in P:
                    forb |= F(o, q)
                A2 = A & ~forb
                for lj, k in contrib[o]:
                    r = rc[lj] + k
                    rc[lj] = r
                    if r == 2:
                        A2 &= ~mall[lj]
                    elif r == 1:
                        A2 &= ~m2[lj]
                P.append(o)
                if solve(A2, npts + len(orbs[o])):
                    return True
                P.pop()
                for lj, k in contrib[o]:
                    rc[lj] -= k
                A &= ~(1 << o)
                a1 = A & m1[li]
                if need == 1:
                    if not a1:
                        return False
                else:
                    if not (A & m2[li]) and popcount(a1) < 2:
                        return False
            return False

        try:
            if solve(st['valid'], 0):
                return [j for o in P for j in orbs[o]]
        except _Out:
            pass
        except RecursionError:
            pass
        return None

    if n % 2 == 0:
        schedule = ['rot4', 'dia2', 'rot4', 'dia2', 'rot2']
    else:
        schedule = ['rot2', 'dia2', 'rot2', 'rot4']
    lidx = {}
    unit = max(40, 8 * n)
    attempt = 0
    while time.time() < deadline:
        sym = schedule[attempt % len(schedule)]
        attempt += 1
        if sym not in structs:
            structs[sym] = build(sym)
            if time.time() >= deadline:
                break
        lidx[sym] = lidx.get(sym, 0) + 1
        budget = unit * _luby(lidx[sym])
        res = run(structs[sym], budget)
        if res is not None:
            return [(j % n, j // n) for j in res]

    # Fallback: greedy extension of best partial without symmetry.
    pts = list(best[0])
    forb = bytearray(NN)
    rowc = [0] * n
    colc = [0] * n

    def mark(p, q):
        px, py = p % n, p // n
        dx = q % n - px
        dy = q // n - py
        g = gcd(dx, dy)
        dx //= g
        dy //= g
        x, y = px, py
        while 0 <= x < n and 0 <= y < n:
            forb[y * n + x] = 1
            x += dx
            y += dy
        x, y = px, py
        while 0 <= x < n and 0 <= y < n:
            forb[y * n + x] = 1
            x -= dx
            y -= dy

    for i, p in enumerate(pts):
        forb[p] = 1
        rowc[p // n] += 1
        colc[p % n] += 1
        for q in pts[:i]:
            mark(p, q)
    order = list(range(NN))
    rng.shuffle(order)
    for j in order:
        if time.time() > t0 + time_limit * 0.97:
            break
        if forb[j] or rowc[j // n] >= 2 or colc[j % n] >= 2:
            continue
        for q in pts:
            mark(j, q)
        forb[j] = 1
        pts.append(j)
        rowc[j // n] += 1
        colc[j % n] += 1
    return [(j % n, j // n) for j in pts]
