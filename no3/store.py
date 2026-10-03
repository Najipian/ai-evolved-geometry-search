"""Best verified configuration per problem and size, kept across all runs (best/<problem>.json).

Programs can be warm-started from it, so progress accumulates instead of being rediscovered.
Only configurations that pass the problem's verifier unchanged are ever stored.
"""
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = ROOT / "best"
_lock = threading.Lock()


def path(problem):
    return STORE_DIR / f"{problem}.json"


def load(problem):
    p = path(problem)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def offer(problem, n, points, source, score):
    """Store points for size n if they are valid and larger than what is stored. Returns True if stored."""
    result = score(points, n)
    if result["triples"] or result["valid_points"] != len(points) or not points:
        return False  # only configurations that need no repair
    with _lock:
        data = load(problem)
        old = data.get(str(n))
        if old and len(old["points"]) >= len(points):
            return False
        data[str(n)] = {"points": [list(p) for p in points], "size": len(points),
                        "target": result["target"], "source": source}
        STORE_DIR.mkdir(exist_ok=True)
        tmp = path(problem).with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path(problem))
    return True


def import_runs(runs_dir=ROOT / "runs"):
    """Offer every run's best.json to the store (python -m no3.store)."""
    from no3.problems import PROBLEMS
    for run in sorted(Path(runs_dir).iterdir()):
        best_file = run / "best.json"
        if not best_file.exists():
            continue
        config_file = run / "config.json"
        config = json.loads(config_file.read_text(encoding="utf-8-sig")) if config_file.exists() else {}
        problem = config.get("problem", "plane")
        for n, entry in json.loads(best_file.read_text(encoding="utf-8")).items():
            points = [tuple(p) for p in entry["points"]]
            if offer(problem, int(n), points, f"{run.name}/{entry['program']}", PROBLEMS[problem]["score"]):
                print(f"{problem} n={n}: stored {len(points)} points from {run.name}/{entry['program']}")


if __name__ == "__main__":
    import_runs()
    for problem in ("plane", "cube4"):
        data = load(problem)
        if data:
            print(problem, {int(k): (v["size"], v["target"]) for k, v in sorted(data.items(), key=lambda kv: int(kv[0]))})
