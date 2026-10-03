"""Evaluates the hand-written programs and writes runs/baselines/report.html.

Usage: python -m no3.baselines [--sizes 10,14,18,22,26,30] [--time 4]
"""
import argparse
import json
from pathlib import Path

from no3.evaluate import evaluate
from no3.evolve import describe, update_best
from no3.report import write_report

ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = [
    ("parabola", "baseline_parabola.py", "Erdos construction (x, x^2 mod p)"),
    ("greedy", "baseline_greedy.py", "random greedy insertion"),
    ("annealing", "seed_annealing.py", "simulated annealing, two per row/column (evolution seed)"),
    ("sym-anneal", "symmetry_annealing.py", "annealing restricted to half-turn / diagonal symmetry"),
    ("sym-backtrack", "symmetry_backtrack.py", "half-turn symmetric backtracking with line marking (evolution seed)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,14,18,22,26,30")
    ap.add_argument("--time", type=float, default=6.0)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    run = ROOT / "runs" / "baselines"
    run.mkdir(parents=True, exist_ok=True)
    (run / "best.json").unlink(missing_ok=True)
    archive = []
    for name, filename, idea in PROGRAMS:
        path = ROOT / "programs" / filename
        result = evaluate(path, sizes, args.time)
        archive.append({"id": name, "parent": None, "idea": idea, "path": str(path),
                        **{k: result[k] for k in ("fitness", "solved", "error")},
                        "results": [{k: v for k, v in r.items() if k != "points"} for r in result["results"]],
                        "usage": None})
        update_best(run, result, name)
        print(f"{name:10s} {describe(archive[-1])}")
    (run / "archive.json").write_text(json.dumps(archive, indent=1), encoding="utf-8")
    print(f"Report: {write_report(run)}")


if __name__ == "__main__":
    main()
