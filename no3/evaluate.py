"""Evaluates a program on several grid sizes, each in an isolated subprocess.

The evaluator never trusts a program's output: every configuration is
re-verified with the problem's own verifier before it is scored.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from no3 import store
from no3.problems import PROBLEMS

RUNNER = Path(__file__).with_name("runner.py")

ALLOWED_IMPORTS = {
    "math", "random", "time", "itertools", "functools", "collections",
    "heapq", "bisect", "array", "numpy", "fractions", "operator",
}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals"}


def static_check(code):
    """Reject programs that import outside the allowlist or call risky builtins.

    This is a guard against accidents, not a security sandbox.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            names = []
        for name in names:
            if name not in ALLOWED_IMPORTS:
                return f"import not allowed: {name}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return f"call not allowed: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            return f"dunder attribute not allowed: {node.attr}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "construct" for n in tree.body):
        return "missing top-level construct(n, time_limit, seed)"
    return None


def run_one(program, n, time_limit, seed, score, warm_file=None):
    started = time.time()
    extra = [str(warm_file)] if warm_file else []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(RUNNER), str(Path(program).resolve()), str(n), str(time_limit), str(seed)]
                + extra,
                capture_output=True, text=True, cwd=tmp, timeout=time_limit * 1.5 + 10,
            )
        except subprocess.TimeoutExpired:
            return {**score([], n), "error": "timeout", "seconds": time.time() - started}
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()[-1:] or ["unknown error"]
        return {**score([], n), "error": err[0][:300], "seconds": time.time() - started}
    try:
        points = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {**score([], n), "error": "could not parse output", "seconds": time.time() - started}
    return {**score(points, n), "error": None, "seconds": round(time.time() - started, 2)}


def evaluate(program, sizes, time_limit, seed=0, workers=None, problem="plane", warm=False, source=None):
    """Score a program on every n in sizes. Fitness = mean of valid_points / target.

    warm=True passes the best stored configuration to programs that accept `start`. Every valid
    result is offered to the cross-run store either way.
    """
    score = PROBLEMS[problem]["score"]
    code = Path(program).read_text(encoding="utf-8")
    issue = static_check(code)
    if issue:
        return {"fitness": 0.0, "solved": 0, "error": issue, "results": []}
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    warm_file = store.path(problem) if warm and store.path(problem).exists() else None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda n: run_one(program, n, time_limit, seed + n, score, warm_file), sizes))
    for r in results:
        if r["points"]:
            store.offer(problem, r["n"], r["points"], source or Path(program).name, score)
    errors = [r["error"] for r in results if r["error"]]
    return {
        "fitness": sum(r["ratio"] for r in results) / len(results),
        "solved": sum(r["solved"] for r in results),
        "error": errors[0] if errors else None,
        "results": results,
    }
