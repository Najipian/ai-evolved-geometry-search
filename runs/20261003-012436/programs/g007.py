# g007 (parent s0): Exact numpy plane-cover counts give, for each point, the cells it blocks alone. A gcd-normalised duplicate-normal test then searches (1,2) and (2,3) swaps on a tabu plateau with kicks, warm-started from `start`.
import math
import random
import time
import numpy as np
from math import gcd


def _plane_key(q, a, b):
    ux, uy, uz = a[0] - q[0], a[1] - q[1], a[2] - q[2]
    vx, vy, vz = b[0] - q[0], b[1] - q[1], b[2] - q[2]
    x, y, z = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    if x == 0 and y == 0 and z == 0:
        return None
    g = gcd(gcd(abs(x), abs(y)), abs(z))
    x, y, z = x // g, y // g, z // g
    if x < 0 or (x == 0 and (y < 0 or (y == 0 and z < 0))):
        x, y, z = -x, -y, -z
    return (x, y, z, x * q[0] + y * q[1] + z * q[2])


def _verify(pts):
    k = len(pts)
    if len(set(pts)) != k:
        return False
    seen = set()
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                key = _plane_key(pts[i], pts[j], pts[l])
                if key is None or key in seen:
                    return False
                seen.add(key)
    return True


def construct(n, time_limit, seed, start=None):
    T0 = time.time()
    deadline = T0 + time_limit * 0.86
    rng = random.Random(seed)
    N = n ** 3
    ar = np.arange(N)
    coords = np.stack([ar // (n * n), (ar // n) % n, ar % n], 1).astype(np.int64)
    cT = coords.T.astype(np.float64)
    B = 2 * n * n + 2
    W = 2 * B + 1
    FCAP = 150

    def cid(p):
        return (p[0] * n + p[1]) * n + p[2]

    def ctup(c):
        return (int(c // (n * n)), int((c // n) % n), int(c % n))

    inc_cache = {}

    def pairs_inc(m):
        if m not in inc_cache:
            a, b = np.triu_indices(m, 1)
            P = len(a)
            Inc = np.zeros((m, P), np.float32)
            if P:
                Inc[a, np.arange(P)] = 1
                Inc[b, np.arange(P)] = 1
            inc_cache[m] = (a, b, Inc)
        return inc_cache[m]

    def plane_masks(p, R):
        m = len(R)
        a, b, Inc = pairs_inc(m)
        if len(a) == 0:
            return np.zeros((0, N), np.float32), Inc
        u = R[a] - p
        v = R[b] - p
        nr = np.cross(u, v)
        d = (nr @ p).astype(np.float64)
        masks = ((nr.astype(np.float64) @ cT) == d[:, None]).astype(np.float32)
        return masks, Inc

    st = {}

    def reset():
        st['cells'] = []
        st['cp'] = np.zeros((0, N), np.float32)
        st['cov'] = np.zeros(N, np.float32)
        st['occ'] = np.zeros(N, bool)

    def add(c):
        c = int(c)
        cells = st['cells']
        R = coords[cells] if cells else np.zeros((0, 3), np.int64)
        masks, Inc = plane_masks(coords[c], R)
        s = masks.sum(0)
        cp = st['cp']
        if len(cells) >= 2:
            cp = cp + Inc @ masks
        st['cp'] = np.vstack([cp, s[None, :]])
        st['cov'] = st['cov'] + s
        st['occ'][c] = True
        cells.append(c)

    def remove(i):
        cells = st['cells']
        c = cells[i]
        others = cells[:i] + cells[i + 1:]
        R = coords[others] if others else np.zeros((0, 3), np.int64)
        cp = np.delete(st['cp'], i, 0)
        if len(others) >= 2:
            masks, Inc = plane_masks(coords[c], R)
            cp -= Inc @ masks
        st['cov'] = st['cov'] - st['cp'][i]
        st['cp'] = cp
        st['occ'][c] = False
        cells.pop(i)
        return c

    def build(cells):
        reset()
        for c in cells:
            add(c)

    def free():
        return np.nonzero((st['cov'] == 0) & ~st['occ'])[0]

    def compat(F, Rc):
        A = coords[F]
        R = coords[Rc] if len(Rc) else np.zeros((0, 3), np.int64)
        D = A[None, :, :] - A[:, None, :]
        E = R[None, :, :] - A[:, None, :]
        Dx, Dy, Dz = D[..., 0][:, :, None], D[..., 1][:, :, None], D[..., 2][:, :, None]
        Ex, Ey, Ez = E[..., 0][:, None, :], E[..., 1][:, None, :], E[..., 2][:, None, :]
        x = Dy * Ez - Dz * Ey
        y = Dz * Ex - Dx * Ez
        z = Dx * Ey - Dy * Ex
        g = np.gcd(np.gcd(x, y), z)
        zero = g == 0
        g[zero] = 1
        x //= g
        y //= g
        z //= g
        s = np.where(x != 0, np.sign(x), np.where(y != 0, np.sign(y), np.sign(z)))
        code = ((x * s + B) * W + (y * s + B)) * W + (z * s + B)
        code.sort(axis=2)
        dup = (code[:, :, 1:] == code[:, :, :-1]).sum(2) + zero.sum(2)
        np.fill_diagonal(dup, 99)
        return dup

    def tri_ok(a, b, c, Rc):
        pa, pb, pc_ = coords[a], coords[b], coords[c]
        nr = np.cross(pb - pa, pc_ - pa)
        if not nr.any():
            return False
        if len(Rc) == 0:
            return True
        vals = (coords[Rc] - pa) @ nr
        return not (vals == 0).any()

    def sample_cap(F):
        if len(F) > FCAP:
            return np.array(rng.sample(list(F), FCAP))
        return F

    def greedy_fill(excl=None):
        while time.time() < deadline:
            F = free()
            if excl:
                F = F[~np.isin(F, np.array(list(excl)))]
            if len(F) == 0:
                return
            Fs = sample_cap(F)
            conf = compat(Fs, st['cells'])
            sc = (conf == 0).sum(1).astype(float) + np.array([rng.random() * 0.5 for _ in range(len(Fs))])
            add(int(Fs[int(np.argmax(sc))]))

    # ---------- initialisation ----------
    start_pts = None
    if start:
        sp = list(dict.fromkeys(tuple(int(v) for v in p) for p in start))
        if all(0 <= v < n for p in sp for v in p) and _verify(sp):
            start_pts = sp
    reset()
    if start_pts and len(start_pts) >= 3:
        build([cid(p) for p in start_pts])
    else:
        while True:
            a, b, c = rng.sample(range(N), 3)
            if np.cross(coords[b] - coords[a], coords[c] - coords[a]).any():
                break
        add(a); add(b); add(c)
    greedy_fill()
    best = list(st['cells'])

    def record():
        nonlocal best
        if len(st['cells']) > len(best):
            best = list(st['cells'])

    step = 0
    last_imp = 0
    tabu_add = {}
    added_at = {}
    sweep_time = 0.0
    seen_pairs_state = 0

    def sweep23():
        cells = st['cells']
        k = len(cells)
        covm = st['cov']
        cp = st['cp']
        occ = st['occ']
        order = list(range(k))
        rng.shuffle(order)
        done = set()
        for i in order:
            if time.time() > deadline:
                return False
            others = [j for j in range(k) if j != i]
            R = coords[[cells[j] for j in others]]
            masks, Inc = plane_masks(coords[cells[i]], R)
            pcov = Inc @ masks
            Fm = ((covm[None, :] - cp[i][None, :] - cp[others] + pcov) == 0) & ~occ[None, :]
            cnt = Fm.sum(1)
            for jj in np.nonzero(cnt >= 3)[0]:
                j = others[jj]
                key = (min(i, j), max(i, j))
                if key in done:
                    continue
                done.add(key)
                F = sample_cap(np.nonzero(Fm[jj])[0])
                R2 = [cells[t] for t in range(k) if t != i and t != j]
                conf = compat(F, R2)
                adj = conf == 0
                f = len(F)
                tries = 0
                for a in range(f):
                    na = np.nonzero(adj[a, a + 1:])[0] + a + 1
                    if len(na) < 2:
                        continue
                    for b in na:
                        common = na[(na > b) & adj[b, na]]
                        for c in common:
                            tries += 1
                            if tri_ok(int(F[a]), int(F[b]), int(F[c]), R2):
                                hi, lo = max(i, j), min(i, j)
                                remove(hi)
                                remove(lo)
                                add(int(F[a])); add(int(F[b])); add(int(F[c]))
                                return True
                            if tries > 400:
                                break
                        if tries > 400:
                            break
                    if tries > 400:
                        break
        return False

    while time.time() < deadline:
        step += 1
        F = free()
        if len(F):
            greedy_fill()
            record()
            last_imp = step
            continue
        cells = st['cells']
        k = len(cells)
        diff = st['cov'][None, :] - st['cp']
        Fmask = (diff == 0) & ~st['occ'][None, :]
        improved = False
        cand = []
        order = list(range(k))
        rng.shuffle(order)
        for i in order:
            Fi = np.nonzero(Fmask[i])[0]
            if len(Fi) == 0:
                continue
            Fi = sample_cap(Fi)
            Rc = cells[:i] + cells[i + 1:]
            if len(Fi) >= 2:
                conf = compat(Fi, Rc)
                ok = np.argwhere(conf == 0)
                if len(ok):
                    a, b = ok[rng.randrange(len(ok))]
                    remove(i)
                    add(int(Fi[a])); add(int(Fi[b]))
                    improved = True
                    break
                near = (conf <= 1).sum(1)
            else:
                near = [0]
            for a in range(len(Fi)):
                cand.append((i, int(Fi[a]), int(near[a])))
        if improved:
            greedy_fill()
            record()
            last_imp = step
            continue
        if time.time() > deadline:
            break
        # occasional (2,3) sweep with a time budget
        elapsed = time.time() - T0
        if sweep_time < 0.45 * elapsed + 0.5 and step % 3 == 1:
            ts = time.time()
            ok = sweep23()
            sweep_time += time.time() - ts
            if ok:
                greedy_fill()
                record()
                last_imp = step
                continue
        cells = st['cells']
        k = len(cells)
        stall = 40 + 2 * k
        if not cand or step - last_imp > stall:
            # kick
            if len(cells) < len(best) - 1:
                build(best)
            r = rng.choice([1, 2, 2, 3])
            removed = []
            for _ in range(min(r, len(st['cells']) - 3)):
                i = rng.randrange(len(st['cells']))
                removed.append(remove(i))
            for c in removed:
                tabu_add[c] = step + 10
            greedy_fill(set(removed))
            record()
            last_imp = step
            continue
        tenure = 6 + rng.randrange(6)
        cand2 = [t for t in cand if tabu_add.get(t[1], -1) < step and
                 step - added_at.get(cells[t[0]], -999) > 3]
        if not cand2:
            cand2 = cand
        if rng.random() < 0.25:
            i, c, _ = cand2[rng.randrange(len(cand2))]
        else:
            i, c, _ = max(cand2, key=lambda t: t[2] + rng.random())
        old = remove(i)
        add(c)
        tabu_add[old] = step + tenure
        added_at[c] = step

    res = [ctup(c) for c in best]
    if _verify(res):
        if start_pts and len(start_pts) > len(res):
            return start_pts
        return res
    if start_pts:
        return start_pts
    return res
