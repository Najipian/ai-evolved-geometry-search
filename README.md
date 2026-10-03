# Mini-AlphaEvolve: AI-evolved search for point-placement problems

An experiment in **LLM-guided program evolution**, in the spirit of DeepMind's FunSearch and AlphaEvolve.
Claude writes search programs, an exact verifier checks every answer they produce, and the best programs are
handed back to Claude to improve. It was pointed at two problems from combinatorial geometry:

1. **No-three-in-line** ([OEIS A000769](https://oeis.org/A000769)): place 2n points on an n x n grid so that
   no three lie on any straight line.
2. **No-four-on-a-plane** ([OEIS A280537](https://oeis.org/A280537)): choose as many points as possible from an
   n x n x n grid so that no four lie on one plane.

Built and run in two days (October 2026) for about **$38** of API usage plus local CPU time.

## Results

**No-three-in-line.** Perfect 2n solutions for every benchmark size from n = 10 to n = 40, plus 80/84 at n = 42.
The hand-written starting programs solved only n = 10 and 14. These sizes were already solved by others
(solutions are known up to n = 74 and n = 76; see
[Flammenkamp's record page](https://wwwhomes.uni-bielefeld.de/achim/no3in/readme.html)), so this is an
independent re-discovery, not new mathematics. Every grid is in `best/plane.json`.

**No-four-on-a-plane.** No new record yet; the best sets found are 0-3 points below the best known values:

| n | 9 | 10 | 11 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|
| Best found here | 23 | 24 | 26 | 30 | 32 | 34 |
| Best known (OEIS) | 23 | 26 | 28 | 32 | 35 | 36 |

The n = 9 set matching the record came from an exact SAT search, not from the evolved programs (see below).
SAT searches for 24 (n = 9), 27 (n = 10), 29 (n = 11) and 33 (n = 13) were inconclusive after 3 hours each.

## What we learned

- **The AI is very good at writing search code.** Within a few generations it combined symmetry, tabu search,
  iterated local search and exact backtracking with restarts, and it took the grid problem from 2 to 6 solved sizes
  in 6 generations.
- **Time can matter more than cleverness.** Giving the best grid program 30 minutes instead of 20 seconds per size
  solved n = 34, 38 and 40 at no API cost.
- **Every evolved cube program stalled 1-3 points below the records,** and 10 more generations with warm starts
  added nothing. Looking at the published record sets showed why: **every record for n = 9..18 has the same
  3-fold symmetry** (rotation about the cube's diagonal, (x, y, z) -> (y, z, x)). Restricting an exact SAT search to
  that symmetry matched the n = 9 record in 3 minutes. The breakthrough came from analysing the data and choosing
  the right tool, not from more generations.
- **Verification is everything.** The AI never grades its own answers; a bug that silently changed the benchmark
  on resume was caught only because each run records its benchmark (`config.json`).

## How it works

```
archive of programs ──> Claude writes an improved program ──> static safety check
        ^                                                            │
        │                                                            v
   scores + best sets <── exact verifier (recount, repair) <── run each size in an isolated process
```

- `no3/evolve.py`: the loop. Parent selection, prompt with the parent's code, results and recent attempts,
  Claude Sonnet 5.5 for most generations and Claude Opus 5.5 every 4th generation or on a stall.
- `no3/core.py`, `no3/cube4.py`: exact verifiers (line and plane hashing with gcd-normalised integer keys).
- `no3/evaluate.py`, `no3/runner.py`: one isolated subprocess per size with a time limit; imports outside a small
  allowlist and calls such as `open` and `eval` are rejected before running.
- `no3/store.py`: the best verified answer per size across all runs; programs that accept `start` are
  warm-started from it.
- `no3/longrun.py`: run one finished program for many minutes per size, no API calls.
- `no3/cube_sat.py`: exact SAT search (python-sat, CaDiCaL) for 3-fold-symmetric cube sets with lazy constraint
  generation: the solver proposes a set, the verifier finds a plane with 4+ points, that plane's rule is added,
  repeat. Learned rules are saved, so long searches can be resumed.
- `no3/report.py`: an HTML report per run (progress chart, best grid or cube per size, every program tried).

The algorithms the evolved programs use (simulated annealing, tabu search, backtracking with forward checking,
symmetry restriction, iterated local search, restarts) are explained in plain language in the project write-up.

## Reproduce

```bash
pip install -r requirements.txt
python -m no3.baselines                                   # hand-written baselines, no API
export ANTHROPIC_API_KEY=sk-ant-...                       # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."
python -m no3.evolve --generations 20                     # grid problem, sizes 10-30
python -m no3.evolve --problem cube4 --time 20 --generations 20
python -m no3.longrun --problem plane --program runs/<run> --sizes 34,38 --minutes 30
python -m no3.cube_sat --n 9 --k 24 --minutes 720         # exact search for a new record
```

Every run writes `runs/<timestamp>/` with the programs, `archive.json`, `best.json`, `config.json` and
`report.html`. `--resume runs/<folder>` continues a run on the same benchmark.

## Runs in this repository

| Run | Problem | Generations | Cost | Outcome |
|---|---|---|---|---|
| `20261001-215923` | grid, n = 10-30 | 20 | $3.51 | all 6 sizes solved by generation 6 |
| `20261002-004836` | grid | 20 | $2.56 | resume bug: scored on the wrong sizes; programs 2x faster |
| `20261002-020525` | grid, n = 32-42 | 20 | $10.87 | n = 32 and 36 solved |
| `20261002-014724` | grid, n = 32-42 | 16 | $10.75 | parallel run, same results |
| `20261002-034644` | cube, n = 9-15 | 20 | $7.40 | 1-3 points below every record |
| `20261003-012436` | cube, warm starts | 10 | $3.10 | no gain; paused |

`runs/best-plane/report.html` and `runs/best-cube4/report.html` show every best grid and cube found.

## Safety

Generated code runs locally. The static check and process isolation guard against accidents, not against a
determined attacker; use a container or a dedicated machine for long unattended runs.

## Credits and sources

- No-three-in-line records: Achim Flammenkamp's [record page](https://wwwhomes.uni-bielefeld.de/achim/no3in/readme.html)
  (2026 records by Marijn Heule and Thomas Prellberg).
- No-four-on-a-plane records and witness sets: [OEIS A280537](https://oeis.org/A280537), with configurations by
  Moritz Franckenstein, Aleksei Kudriashov and Hugo Pfoertner, and
  [Al Zimmermann's "Non-Coplanar Points" contest](http://azspcs.com/Contest/Tetrahedra).
- Method inspired by DeepMind's FunSearch (2023) and AlphaEvolve (2025).

## License

MIT, see `LICENSE`.
