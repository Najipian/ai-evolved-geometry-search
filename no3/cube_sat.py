"""Exact SAT search for no-four-coplanar sets in the n x n x n grid with 3-fold symmetry.

Every published record set for n = 9..18 (OEIS A280537, file a280537_4.txt) is invariant under the
3-fold rotation R(x, y, z) = (y, z, x) about the cube's main diagonal. So we only look for sets made of
whole R-orbits: 3-point orbits plus at most 2 points on the axis x = y = z.

Lazy constraint generation (counterexample-guided):
  1. Ask the SAT solver for a set of orbits with at least k points, subject to the rules found so far.
  2. Check it with the exact verifier (no3.cube4). If it is valid, done.
  3. Otherwise, for every plane holding 4+ chosen points (or line holding 3+), add the rule
     "at most 3 points on this plane" (2 on this line) over ALL grid cells on it, and go to 1.
UNSAT means no R-symmetric set with k points exists at all (the rules added are all true rules).

Learned rules are saved to data/sat_rules/n<n>_k<k>.json after every round and loaded on start, so a
long search can be stopped and continued later without starting from zero.

Usage: python -m no3.cube_sat --n 9 --k 24 [--minutes 60]
"""
import argparse
import json
import time
from math import gcd
from pathlib import Path

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Cadical153

from no3.cube4 import RECORDS, is_valid, plane_key, violations

ROOT = Path(__file__).resolve().parent.parent


def rotate(p):
    return p[1], p[2], p[0]


def orbits(n):
    seen, out = set(), []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                p = (x, y, z)
                if p in seen:
                    continue
                orb = sorted({p, rotate(p), rotate(rotate(p))})
                seen.update(orb)
                out.append(orb)
    return out


def cells_on_plane(key, n):
    a, b, c, d = key
    return [(x, y, z) for x in range(n) for y in range(n) for z in range(n) if a * x + b * y + c * z == d]


def cells_on_line(p, q, n):
    dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    g = gcd(gcd(dx, dy), dz)
    dx, dy, dz = dx // g, dy // g, dz // g
    x, y, z = p
    while 0 <= x - dx < n and 0 <= y - dy < n and 0 <= z - dz < n:
        x, y, z = x - dx, y - dy, z - dz
    out = []
    while 0 <= x < n and 0 <= y < n and 0 <= z < n:
        out.append((x, y, z))
        x, y, z = x + dx, y + dy, z + dz
    return out


def deep_tuple(x):
    return tuple(deep_tuple(i) for i in x) if isinstance(x, list) else x


def search(n, k, minutes, log=print, rules_file=None):
    orbs = orbits(n)
    pool = IDPool()
    var_of_cell = {}
    for i, orb in enumerate(orbs):
        v = pool.id(("orbit", i))
        for p in orb:
            var_of_cell[p] = v
    lits = lambda cells: [var_of_cell[p] for p in cells]

    solver = Cadical153()
    added = set()
    saved = []  # learned rules in a JSON-friendly form, for --resume

    def add_atmost(cells, bound, tag, keep=False):
        if tag in added:
            return
        added.add(tag)
        if keep:
            saved.append({"bound": bound, "tag": tag,
                          "cells": [list(c) for c in cells]})
        enc = CardEnc.atmost(lits=lits(cells), bound=bound, vpool=pool, encoding=EncType.totalizer)
        solver.append_formula(enc.clauses)

    # at least k points: each orbit's literal appears once per point it contributes
    enc = CardEnc.atleast(lits=[var_of_cell[p] for orb in orbs for p in orb], bound=k, vpool=pool,
                          encoding=EncType.totalizer)
    solver.append_formula(enc.clauses)
    # cheap rules up front: the 3n axis-parallel layers, and the symmetry axis (a line)
    for c in range(n):
        for axis in range(3):
            add_atmost([p for p in var_of_cell if p[axis] == c], 3, ("layer", axis, c))
    add_atmost([(i, i, i) for i in range(n)], 2, ("axis",))
    if rules_file and rules_file.exists():
        for r in json.loads(rules_file.read_text(encoding="utf-8")):
            tag = deep_tuple(r["tag"])
            add_atmost([tuple(c) for c in r["cells"]], r["bound"], tag, keep=True)
        log(f"  resumed with {len(saved)} saved rules from {rules_file}", flush=True)

    def save():
        if rules_file:
            rules_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = rules_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved), encoding="utf-8")
            tmp.replace(rules_file)

    started = time.time()
    rounds = 0
    while True:
        rounds += 1
        remaining = minutes * 60 - (time.time() - started)
        if remaining <= 0:
            return "timeout", None, rounds, len(added)
        # CaDiCaL cannot be interrupted, so the limit is checked between rounds. Rules are saved every
        # round, so stopping the process at any moment loses nothing.
        if not solver.solve():
            return "unsat", None, rounds, len(added)
        model = set(l for l in solver.get_model() if l > 0)
        chosen = [p for p, v in var_of_cell.items() if v in model]
        bad = violations(chosen)
        if not bad:
            return "found", chosen, rounds, len(added)
        for s in bad:
            pts = [chosen[i] for i in sorted(s)]
            if len(s) == 3 and plane_key(*pts) is None:  # collinear triple: at most 2 on that line
                line = cells_on_line(pts[0], pts[1], n)
                add_atmost(line, 2, ("line", tuple(line[:2])), keep=True)
                continue
            key = None
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    for m in range(j + 1, len(pts)):
                        key = plane_key(pts[i], pts[j], pts[m])
                        if key:
                            break
                    if key:
                        break
                if key:
                    break
            add_atmost(cells_on_plane(key, n), 3, ("plane",) + key, keep=True)
        save()
        if rounds % 50 == 0:
            log(f"  round {rounds}: {len(added)} rules, {time.time() - started:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, help="points to find (default: best known + 1)")
    ap.add_argument("--minutes", type=float, default=60)
    args = ap.parse_args()
    k = args.k or RECORDS.get(args.n, 0) + 1
    print(f"n={args.n}: searching for {k} points with 3-fold diagonal symmetry "
          f"(best known {RECORDS.get(args.n, '?')}), up to {args.minutes:g} min", flush=True)
    rules_file = ROOT / "data" / "sat_rules" / f"n{args.n}_k{k}.json"
    status, pts, rounds, rules = search(args.n, k, args.minutes, log=print, rules_file=rules_file)
    print(f"result: {status} after {rounds} rounds and {rules} rules", flush=True)
    if status == "found":
        ok = is_valid(pts, args.n) and len(pts) >= k
        out = ROOT / "data" / "found"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"cube4_n{args.n}_{len(pts)}.json"
        path.write_text(json.dumps({"n": args.n, "size": len(pts), "points": sorted(pts)}), encoding="utf-8")
        print(f"{len(pts)} points, independently valid: {ok}; saved {path}", flush=True)
        if ok and len(pts) > RECORDS.get(args.n, 10 ** 9):
            print("NEW RECORD candidate - check the latest OEIS entry before claiming it", flush=True)


if __name__ == "__main__":
    main()
