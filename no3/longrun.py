"""Run one program for a long time per size (no API calls), keeping the best result in the store.

Usage:
  python -m no3.longrun --problem cube4 --program runs/<run>/programs/g012.py --minutes 20
  python -m no3.longrun --problem plane --program programs/evolved_r2_g013.py --sizes 34,38 --minutes 30 --rounds 2

Each round runs every size in parallel with a new seed. Results are verified, offered to the
cross-run store (best/<problem>.json), and compared with the target (2n, or the best known value).
"""
import argparse
import json
import time
from pathlib import Path

from no3 import store
from no3.evaluate import evaluate
from no3.problems import PROBLEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    ap.add_argument("--program", required=True, help="a program file, or a run folder (uses its best program)")
    ap.add_argument("--sizes", help="comma-separated sizes (default: the problem's benchmark sizes)")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--rounds", type=int, default=1, help="repeat with new seeds")
    ap.add_argument("--workers", type=int, help="parallel sizes (default: half the CPU threads)")
    args = ap.parse_args()

    sizes = [int(s) for s in (args.sizes or PROBLEMS[args.problem]["sizes"]).split(",")]
    program = Path(args.program)
    if program.is_dir():  # a run folder: use its best program (most sizes solved, then fitness)
        archive = json.loads((program / "archive.json").read_text(encoding="utf-8"))
        top = max((e for e in archive if e["results"]), key=lambda e: (e["solved"], e["fitness"]))
        print(f"Best program of {program.name}: {top['id']} (fitness {top['fitness']:.4f})", flush=True)
        program = Path(top["path"])
    label = f"longrun:{program.parent.parent.name}/{program.name}" if program.parent.name == "programs" else program.name
    print(f"Long run: {args.problem}, {program}, sizes {sizes}, {args.minutes:g} min per size, {args.rounds} round(s)",
          flush=True)
    for rnd in range(args.rounds):
        started = time.time()
        result = evaluate(program, sizes, args.minutes * 60, seed=1000 * (rnd + 1), workers=args.workers,
                          problem=args.problem, warm=True, source=label)
        if result["error"] and not result["results"]:
            print(f"round {rnd + 1}: program rejected: {result['error']}", flush=True)
            return
        best = store.load(args.problem)
        print(f"round {rnd + 1} ({(time.time() - started) / 60:.1f} min):", flush=True)
        for r in result["results"]:
            stored = best.get(str(r["n"]), {}).get("size", 0)
            if args.problem == "cube4":
                verdict = "NEW RECORD" if stored > r["target"] else ("matches record" if stored == r["target"] else "")
            else:
                verdict = "SOLVED" if stored == r["target"] else ""
            print(f"  n={r['n']}: this round {r['valid_points']}, best ever {stored} / {r['target']} {verdict}"
                  + (f" [error: {r['error']}]" if r["error"] else ""), flush=True)


if __name__ == "__main__":
    main()
