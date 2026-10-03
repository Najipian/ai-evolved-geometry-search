# g009 (parent s0): Min-conflicts tabu search at k=best+1 with exact numpy counts of triple-planes through every cell, updated per swap, plus kicks that rebuild from the best set.
import math
import random
import time
import numpy as np


def construct(n, time_limit, seed, start=None):
    t0 = time.time()
    deadline = t0 + time_limit * 0.9
    rng = random.Random(seed)
    M = n ** 3
    coords = np.array([(x, y, z) for x in range(n) for y in range(n) for z in range(n)], dtype=np.float64)
    PT = coords.T.copy()
    tri_cache = {}

    def tri(m):
        t = tri_cache.get(m)
        if t is None:
            t = np.triu_indices(m, 1)
            tri_cache[m] = t
        return t

    def cid(p):
        return int(p[0]) * n * n + int(p[1]) * n + int(p[2])

    def Dvec(c, others):
        m = len(others)
        if m < 2:
            return np.zeros(M, dtype=np.int64)
        p = coords[c]
        U = coords[others] - p
        iu, ju = tri(m)
        N = np.cross(U[iu], U[ju])
        d = N @ p
        return ((N @ PT) == d[:, None]).sum(axis=0).astype(np.int64)

    state = {}

    def reset(cells):
        C = np.zeros(M, dtype=np.int64)
        S = []
        for c in cells:
            C += Dvec(c, S)
            S.append(c)
        inS = np.zeros(M, dtype=bool)
        inS[S] = True
        state['C'] = C
        state['S'] = S
        state['inS'] = inS

    def add_cell(c):
        state['C'] += Dvec(c, state['S'])
        state['S'].append(c)
        state['inS'][c] = True

    def remove_pos(pos):
        S = state['S']
        c = S.pop(pos)
        state['C'] -= Dvec(c, S)
        state['inS'][c] = False

    def greedy_cell():
        C = state['C']
        F = C.copy()
        F[state['inS']] = 1 << 40
        mn = F.min()
        cs = np.flatnonzero(F == mn)
        return int(cs[rng.randrange(len(cs))]), int(mn)

    def to_tuples(S):
        return [(c // (n * n), (c // n) % n, c % n) for c in S]

    best = []
    if start:
        cells = list(dict.fromkeys(cid(p) for p in start))
        reset(cells)
        K = len(cells)
        base = (K - 1) * (K - 2) // 2
        conf = state['C'][state['S']] - base
        if int(conf.sum()) == 0:
            best = list(cells)
    else:
        reset([rng.randrange(M)])
        while time.time() < deadline:
            c, mn = greedy_cell()
            if mn > 0:
                break
            add_cell(c)
        best = list(state['S'])

    if best:
        target = len(best) + 1
    else:
        target = len(state['S'])
    while len(state['S']) < target:
        c, _ = greedy_cell()
        add_cell(c)

    tabu_until = np.zeros(M, dtype=np.int64)
    step = 0
    bestQ = 1 << 60
    last = 0
    BIG = 1 << 40

    while time.time() < deadline:
        step += 1
        S = state['S']
        C = state['C']
        inS = state['inS']
        K = len(S)
        base = (K - 1) * (K - 2) // 2
        conf = C[S] - base
        Q4 = int(conf.sum())
        if Q4 == 0:
            if K > len(best):
                best = list(S)
            c, _ = greedy_cell()
            add_cell(c)
            bestQ = 1 << 60
            last = step
            continue
        if Q4 < bestQ:
            bestQ = Q4
            last = step
        elif step - last > 60 * K:
            # perturbation
            if best and rng.random() < 0.6:
                cells = list(best)
                r = rng.randint(1, max(2, K // 6))
                for _ in range(min(r, len(cells) - 1)):
                    cells.pop(rng.randrange(len(cells)))
                reset(cells)
                while len(state['S']) < K:
                    c, _ = greedy_cell()
                    add_cell(c)
            else:
                for _ in range(rng.randint(1, 2)):
                    pos = rng.randrange(len(state['S']))
                    remove_pos(pos)
                    while True:
                        c = rng.randrange(M)
                        if not state['inS'][c]:
                            break
                    add_cell(c)
            bestQ = 1 << 60
            last = step
            continue
        Q = Q4 // 4
        confl = np.flatnonzero(conf > 0)
        ncand = 3 if n <= 11 else 2
        if len(confl) <= ncand:
            cands = list(confl)
        else:
            w = conf[confl].tolist()
            cands = list(set(rng.choices(confl.tolist(), weights=w, k=ncand)))
        tabu = tabu_until > step
        bestmove = None
        for pos in cands:
            pos = int(pos)
            oldc = S[pos]
            old = int(conf[pos])
            others = S[:pos] + S[pos + 1:]
            Dp = Dvec(oldc, others)
            F = C - Dp
            Fm = F.copy()
            Fm[inS] = BIG
            Fm[tabu & (F != old - Q)] = BIG
            Fm[oldc] = BIG
            mn = int(Fm.min())
            if mn >= BIG:
                continue
            cs = np.flatnonzero(Fm == mn)
            newc = int(cs[rng.randrange(len(cs))])
            delta = mn - old
            if bestmove is None or delta < bestmove[0] or (delta == bestmove[0] and rng.random() < 0.5):
                bestmove = (delta, pos, newc, Dp, others)
        if bestmove is None:
            continue
        delta, pos, newc, Dp, others = bestmove
        oldc = S[pos]
        C -= Dp
        C += Dvec(newc, others)
        S[pos] = newc
        inS[oldc] = False
        inS[newc] = True
        tabu_until[oldc] = step + 5 + rng.randrange(10)

    if best:
        return to_tuples(best)
    if start:
        return [tuple(p) for p in start]
    return to_tuples(state['S'])
