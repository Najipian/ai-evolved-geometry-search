"""Runs one candidate program in its own process and prints its points as JSON.

Usage: python -I runner.py <program.py> <n> <time_limit> <seed> [<warm-start store.json>]

With a store file, a program whose construct() accepts a `start` argument receives the best
verified configuration known for this n (or None), so it can continue from it.
"""
import importlib.util
import inspect
import json
import sys


def main():
    path, n, time_limit, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = {}
    if len(sys.argv) > 5 and "start" in inspect.signature(module.construct).parameters:
        try:
            with open(sys.argv[5], encoding="utf-8") as f:
                entry = json.load(f).get(str(n))
        except (OSError, ValueError):
            entry = None
        kwargs["start"] = [tuple(p) for p in entry["points"]] if entry else None
    points = module.construct(n, time_limit, seed, **kwargs)
    print(json.dumps([[int(c) for c in p] for p in points]))


if __name__ == "__main__":
    main()
