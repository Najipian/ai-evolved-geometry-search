"""LLM-guided program evolution for point-placement problems (a mini AlphaEvolve).

Each generation:
  1. pick a parent program from the archive (mostly the best one),
  2. ask Claude for an improved program, showing the parent's code, its
     per-size results and recent attempts,
  3. statically check the new program, run it in isolated subprocesses,
     verify every configuration, and add it to the archive.

Usage:
  python -m no3.evolve --generations 20
  python -m no3.evolve --problem cube4 --generations 20
  python -m no3.evolve --resume runs/20261001-120000 --generations 10
"""
import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import anthropic

from no3.evaluate import ALLOWED_IMPORTS, evaluate
from no3.problems import PROBLEMS
from no3.report import write_report

ROOT = Path(__file__).resolve().parent.parent
OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
PRICES = {OPUS: (4.0, 20.0), SONNET: (2.0, 10.0)}  # $ per million input / output tokens

WARM_NOTE = """
Warm start: construct may take a fourth argument, construct(n, time_limit, seed, start=None). If your
signature has `start`, it receives the best verified configuration ever found for this n (a list of
tuples), or None. Use it: continue the search from it (for example remove a few points and re-grow, or
local search around it) instead of starting from scratch. The best result is kept across runs, so
progress on top of `start` is what counts."""

SYSTEM_TEMPLATE = """You are a research engineer improving a search program for a combinatorial geometry problem.

{problem}

Rules for every program:
- It must return within time_limit seconds (leave a small margin).
- Pure Python. Allowed imports only: """ + ", ".join(sorted(ALLOWED_IMPORTS)) + """.
- No file, network, or process access. No printing.

{warm_note}
Reply with one line starting with "IDEA:" summarising your change in under 25 words, then the complete
program in a single ```python code block."""


DEFAULT_TIME = 6.0


def resolve_benchmark(run, archive, args):
    """Fix a run's problem, sizes and time limit when it starts, so a resumed run keeps the same benchmark.

    Scores are only comparable within one benchmark, so a resume that asks for a different one is refused.
    """
    path = run / "config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8-sig"))  # tolerate a BOM from hand edits
    elif archive:  # runs made before config.json existed: take the sizes from the first program's results
        saved = {"sizes": [r["n"] for r in archive[0]["results"]], "time": args.time or DEFAULT_TIME,
                 "problem": "plane"}
        path.write_text(json.dumps(saved), encoding="utf-8")
    else:
        saved = None
    asked_sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
    if saved:
        saved.setdefault("problem", "plane")  # runs made before other problems existed
        saved.setdefault("warm", False)
        saved.setdefault("models", "opus")
        if ((asked_sizes and asked_sizes != saved["sizes"]) or (args.time and args.time != saved["time"])
                or (args.problem and args.problem != saved["problem"])):
            raise SystemExit(f"This run is {saved['problem']} with sizes {saved['sizes']} at {saved['time']:g} s; "
                             "start a new run to change the benchmark.")
        return saved
    problem = args.problem or "plane"
    sizes = asked_sizes or [int(s) for s in PROBLEMS[problem]["sizes"].split(",")]
    time_limit = args.time or DEFAULT_TIME
    config = {"problem": problem, "sizes": sizes, "time": time_limit,
              "warm": not args.no_warm, "models": args.models or "mix"}
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


def load_archive(run):
    path = run / "archive.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_archive(run, archive):
    (run / "archive.json").write_text(json.dumps(archive, indent=1), encoding="utf-8")


def update_best(run, result, program_id):
    """Keep the best verified configuration found for each n across all programs."""
    path = run / "best.json"
    best = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for r in result["results"]:
        key = str(r["n"])
        if key not in best or r["valid_points"] > best[key]["valid_points"]:
            best[key] = {"valid_points": r["valid_points"], "target": r["target"],
                         "points": r["points"], "program": program_id}
    path.write_text(json.dumps(best), encoding="utf-8")


def describe(entry):
    per_n = ", ".join(f"n={r['n']}: {r['valid_points']}/{r['target']}" for r in entry["results"])
    return f"fitness {entry['fitness']:.4f}, solved {entry['solved']} sizes ({per_n})"


def pick_parent(archive, rng):
    ranked = sorted(archive, key=lambda e: (e["solved"], e["fitness"]), reverse=True)
    if rng.random() < 0.7 or len(ranked) == 1:
        return ranked[0]
    return rng.choice(ranked[1:6])


def build_prompt(parent, archive, sizes, time_limit):
    code = Path(parent["path"]).read_text(encoding="utf-8")
    recent = archive[-8:]
    attempts = "\n".join(
        f"- {e['id']}: {e['idea']} -> {describe(e)}" + (f" [error: {e['error']}]" if e["error"] else "")
        for e in recent
    )
    gaps = sorted(parent["results"], key=lambda r: r["target"] - r["valid_points"])
    focus = ", ".join(f"n={r['n']} ({r['valid_points']}/{r['target']})" for r in gaps[:3])
    return f"""Evaluation sizes: {sizes}, time limit {time_limit}s per size.
Closest to their target (focus here first): {focus}.

Parent program {parent['id']} ({describe(parent)}):
```python
{code}
```

Recent attempts (learn from failures; do not repeat ideas that did not help):
{attempts}

Write an improved program. Bold ideas are welcome if they can plausibly solve more sizes."""


def pick_model(archive, gen, models):
    """Opus every 4th generation and whenever the best has not improved for 3 generations; else Sonnet."""
    if models != "mix":
        return OPUS if models == "opus" else SONNET
    gens = [e for e in archive if e["id"].startswith("g")]
    best_ids = []
    running = None
    for e in archive:
        key = (e["solved"], e["fitness"])
        if running is None or key > running:
            running = key
            best_ids.append(e["id"])
    stalled = len(gens) >= 3 and all(e["id"] not in best_ids for e in gens[-3:])
    # A stall calls for Opus, but at most every other generation, so a long plateau stays affordable.
    last_was_opus = bool(gens) and (gens[-1].get("usage") or {}).get("model") == OPUS
    return OPUS if gen % 4 == 0 or (stalled and not last_was_opus) else SONNET


def cost(usage):
    if not usage:
        return 0.0
    pin, pout = PRICES.get(usage.get("model", OPUS), PRICES[OPUS])
    return usage["input_tokens"] * pin / 1e6 + usage["output_tokens"] * pout / 1e6


def ask_claude(client, prompt, system, model):
    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "high"},
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    text = "".join(b.text for b in message.content if b.type == "text")
    usage = message.usage
    return text, {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "model": model}


def parse_reply(text):
    idea = re.search(r"IDEA:\s*(.+)", text)
    code = re.findall(r"```python\s*\n(.*?)```", text, re.S)
    if not code:
        return None, None
    return (idea.group(1).strip() if idea else "(no idea given)"), max(code, key=len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--problem", choices=sorted(PROBLEMS), help="new runs default to plane")
    ap.add_argument("--sizes", help="comma-separated grid sizes (new runs use the problem's default)")
    ap.add_argument("--time", type=float, help=f"seconds per size (new runs default to {DEFAULT_TIME:g})")
    ap.add_argument("--resume", help="existing run directory to continue")
    ap.add_argument("--models", choices=["mix", "opus", "sonnet"], help="new runs default to mix")
    ap.add_argument("--no-warm", action="store_true", help="do not warm-start programs from the best store")
    ap.add_argument("--seed-programs", help="comma-separated files in programs/ that start the archive "
                    "(default: the problem's seeds)")
    args = ap.parse_args()

    run = Path(args.resume) if args.resume else ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    (run / "programs").mkdir(parents=True, exist_ok=True)
    archive = load_archive(run)
    config = resolve_benchmark(run, archive, args)
    problem, sizes, time_limit, warm = config["problem"], config["sizes"], config["time"], config["warm"]
    args.time = time_limit
    system = SYSTEM_TEMPLATE.format(problem=PROBLEMS[problem]["prompt"], warm_note=WARM_NOTE if warm else "")
    print(f"Benchmark: {problem}, sizes {sizes}, {time_limit:g} s per size, "
          f"warm start {'on' if warm else 'off'}, models {config['models']}")
    rng = random.Random()
    client = anthropic.Anthropic()

    if not archive:
        for i, name in enumerate((args.seed_programs or PROBLEMS[problem]["seeds"]).split(",")):
            sid = f"s{i}"
            seed_path = run / "programs" / f"{sid}_{name}"
            shutil.copy(ROOT / "programs" / name, seed_path)
            print(f"Evaluating seed program {name}...")
            result = evaluate(seed_path, sizes, args.time, problem=problem, warm=warm, source=f"{run.name}/{sid}")
            archive.append({"id": sid, "parent": None, "idea": f"seed: {name}",
                            "path": str(seed_path), **{k: result[k] for k in ("fitness", "solved", "error")},
                            "results": [{k: v for k, v in r.items() if k != "points"} for r in result["results"]],
                            "usage": None})
            update_best(run, result, sid)
            save_archive(run, archive)
            print(f"  {sid} {describe(archive[-1])}")

    start = sum(1 for e in archive if e["id"].startswith("g")) + 1
    for gen in range(start, start + args.generations):
        gid = f"g{gen:03d}"
        parent = pick_parent(archive, rng)
        model = pick_model(archive, gen, config["models"])
        print(f"\n[{gid}] parent {parent['id']} - asking {model}...")
        try:
            text, usage = ask_claude(client, build_prompt(parent, archive, sizes, args.time), system, model)
        except TypeError as e:
            if "authentication" not in str(e):  # a real bug, not missing credentials: show it
                raise
            print(f"  No Anthropic credentials found ({e}).\n  Set ANTHROPIC_API_KEY, then rerun with --resume {run}")
            break
        except (anthropic.APIError, RuntimeError) as e:
            print(f"  API problem: {e}; stopping. Rerun with --resume {run}")
            break
        idea, code = parse_reply(text)
        if code is None:
            print("  no code in reply; skipping")
            continue
        path = run / "programs" / f"{gid}.py"
        path.write_text(f"# {gid} (parent {parent['id']}): {idea}\n{code}", encoding="utf-8")
        print(f"  idea: {idea}")
        result = evaluate(path, sizes, args.time, problem=problem, warm=warm, source=f"{run.name}/{gid}")
        entry = {"id": gid, "parent": parent["id"], "idea": idea, "path": str(path),
                 **{k: result[k] for k in ("fitness", "solved", "error")},
                 "results": [{k: v for k, v in r.items() if k != "points"} for r in result["results"]],
                 "usage": usage}
        archive.append(entry)
        if result["results"]:
            update_best(run, result, gid)
        save_archive(run, archive)
        write_report(run)
        best = max(archive, key=lambda e: (e["solved"], e["fitness"]))
        print(f"  {describe(entry)}" + (f" [error: {entry['error']}]" if entry["error"] else ""))
        print(f"  best so far: {best['id']} {describe(best)}")
        print(f"  cost so far: ${sum(cost(e.get('usage')) for e in archive):.2f}")

    write_report(run)
    print(f"\nReport: {run / 'report.html'}")


if __name__ == "__main__":
    main()
