"""The problems the evolution loop can work on: verifier, defaults, and the prompt text Claude sees."""
from no3 import core, cube4

PLANE_PROMPT = """Problem: place as many points as possible on an n x n integer grid so that no three points lie on a
common straight line (any slope). At most 2n points are possible (two per row). The goal is to reach
exactly 2n points for as many n as possible, as fast as possible.

Background you may use:
- Every 2n solution has exactly two points in every row and every column.
- Known record solutions for larger n are usually found by restricting the search to configurations
  with a symmetry (e.g. 180-degree rotation, 90-degree rotation, or reflection), which shrinks the
  search space dramatically. Different n favour different symmetries (odd vs even n matters).
- Incremental cost updates (only recompute lines through moved points) make local search much faster.
- Two seed approaches exist: simulated annealing with two points per row/column, and randomised
  backtracking over half-turn-symmetric configurations (fill rows outside-in, mark every cell on a line
  through two placed points as forbidden, pick the most constrained row, restart with growing budgets).
  Backtracking finds exact 2n solutions; annealing gives better partial results. Combining ideas,
  adding other symmetries (quarter turn for even n, diagonal mirrors), better pruning and faster
  bookkeeping are all promising directions.

Program contract:
- Define a top-level function construct(n: int, time_limit: float, seed: int) -> list of (x, y) tuples.
- Return the best configuration found even if it still has collinear triples; the evaluator repairs it
  by deleting the worst points.

Scoring: for each n, valid_points / (2n) after verification and repair; fitness is the mean across sizes.
Solving (reaching 2n) more sizes is what matters most."""

CUBE4_PROMPT = """Problem (OEIS A280537): choose as many points as possible from the n x n x n integer grid
{0..n-1}^3 so that no four of them lie on a common plane (any orientation). Three collinear points plus
any fourth point are coplanar, so a valid set also has no three points on a line. Upper bound: 3n (each
plane x = const holds at most 3 points).

Best known values (exact or long-standing lower bounds): a(9)>=23, a(10)>=26, a(11)>=28, a(12)>=31,
a(13)>=32, a(14)>=35, a(15)>=36, a(16)>=38, a(17)>=42. Most bounds for n >= 9 date from 2017. BEATING
any of them would be a new mathematical record - that is the goal.

Background you may use:
- A plane is ax+by+cz=d with integer (a,b,c) reduced by gcd; hashing planes through pairs of points gives
  O(N^2) incremental cost per point move.
- Good constructions in related problems often come from algebraic curves over finite fields (e.g. points
  (x, x^2, x^3) mod p: the moment curve has no four coplanar points over a prime field), then trimmed or
  adapted to the grid; or from symmetry (central symmetry, rotations of the cube) to shrink the search.
- The seed is plain simulated annealing on k points that grows k whenever it reaches zero violations.

Program contract:
- Define a top-level function construct(n: int, time_limit: float, seed: int) -> list of (x, y, z) tuples.
- Return the best configuration found; invalid sets are repaired by deleting the worst points.

Scoring: for each n, valid_points / best_known after verification and repair; fitness is the mean across
sizes. A size counts as "solved" only when valid_points EXCEEDS the best known value (a new record), so
fitness close to 1.0 means matching the records."""

PROBLEMS = {
    "plane": {"score": core.score, "sizes": "10,14,18,22,26,30", "prompt": PLANE_PROMPT,
              "seeds": "seed_annealing.py,symmetry_backtrack.py", "dims": 2,
              "goal": "reach 2n points", "title": "No-three-in-line search"},
    "cube4": {"score": cube4.score, "sizes": "9,10,11,13,14,15", "prompt": CUBE4_PROMPT,
              "seeds": "cube4_annealing.py", "dims": 3,
              "goal": "beat the best known value", "title": "No-four-coplanar cube search"},
}
