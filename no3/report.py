"""Builds a self-contained report.html for a run directory (archive.json + best.json)."""
import html
import json
from pathlib import Path

CSS = """
:root { --bg:#f7f7f5; --card:#ffffff; --text:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc;
        --accent:#2f6fde; --good:#1f9d55; --bad:#d64545; --dot:#1d1d1b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151514; --card:#1f1f1d; --text:#ececea; --muted:#9a9a94; --line:#33332f;
          --accent:#6b9bff; --good:#3ecf7a; --bad:#ff6b6b; --dot:#ececea; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 12px; }
.muted { color:var(--muted); }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-top:16px; }
.stat, .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.stat b { display:block; font-size:22px; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); font-size:14px; vertical-align:top; }
th { color:var(--muted); font-weight:600; }
.ok { color:var(--good); font-weight:600; } .err { color:var(--bad); }
.grids { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; }
.grids svg { width:100%; height:auto; display:block; }
.wrap { overflow-x:auto; }
"""


def grid_svg(n, points):
    size = 200
    cell = size / n
    lines = "".join(
        f'<line x1="{i * cell:.2f}" y1="0" x2="{i * cell:.2f}" y2="{size}" />'
        f'<line x1="0" y1="{i * cell:.2f}" x2="{size}" y2="{i * cell:.2f}" />'
        for i in range(n + 1)
    )
    dots = "".join(
        f'<circle cx="{(x + 0.5) * cell:.2f}" cy="{(y + 0.5) * cell:.2f}" r="{max(1.2, cell * 0.35):.2f}" />'
        for x, y in points
    )
    return (f'<svg viewBox="-1 -1 {size + 2} {size + 2}" role="img" aria-label="{len(points)} points on a {n} by {n} grid">'
            f'<g stroke="var(--line)" stroke-width="0.6">{lines}</g><g fill="var(--dot)">{dots}</g></svg>')


def cube_svg(n, points):
    """A 3D configuration drawn as its n horizontal layers (z = 0 .. n-1), side by side."""
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    cell = 30 / n * 2 if n <= 8 else 4
    side = cell * n
    gap = 10
    w = cols * (side + gap)
    h = rows * (side + gap + 12)
    parts = []
    for z in range(n):
        ox = (z % cols) * (side + gap)
        oy = (z // cols) * (side + gap + 12) + 12
        parts.append(f'<text x="{ox:.1f}" y="{oy - 3:.1f}" font-size="8" fill="var(--muted)">z={z}</text>')
        parts.append(f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{side:.1f}" height="{side:.1f}" fill="none" stroke="var(--line)"/>')
        for x, y, zz in points:
            if zz == z:
                parts.append(f'<circle cx="{ox + (x + 0.5) * cell:.1f}" cy="{oy + (y + 0.5) * cell:.1f}" r="{max(1.2, cell * 0.35):.1f}" fill="var(--dot)"/>')
    return (f'<svg viewBox="-1 -1 {w + 2:.0f} {h + 2:.0f}" role="img" aria-label="{len(points)} points in a {n}-cube, by layer">'
            + "".join(parts) + "</svg>")


def fitness_chart(archive):
    if not archive:
        return ""
    w, h, pad = 700, 220, 36
    vals = [e["fitness"] for e in archive]
    lo, hi = min(vals + [0.8]), 1.0
    def px(i):
        return pad + (w - 2 * pad) * (i / max(1, len(archive) - 1))
    def py(v):
        return h - pad - (h - 2 * pad) * ((v - lo) / (hi - lo or 1))
    best, running = [], 0
    for i, v in enumerate(vals):
        running = max(running, v)
        best.append(f"{px(i):.1f},{py(running):.1f}")
    dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.5" fill="var(--muted)"><title>{html.escape(archive[i]["id"])}: {v:.4f}</title></circle>'
                   for i, v in enumerate(vals))
    axis = (f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="var(--line)"/>'
            f'<text x="{pad}" y="{h - 10}" fill="var(--muted)" font-size="12">attempt 0</text>'
            f'<text x="{w - pad}" y="{h - 10}" fill="var(--muted)" font-size="12" text-anchor="end">attempt {len(vals) - 1}</text>'
            f'<text x="4" y="{py(hi) + 4:.1f}" fill="var(--muted)" font-size="12">{hi:.2f}</text>'
            f'<text x="4" y="{py(lo) + 4:.1f}" fill="var(--muted)" font-size="12">{lo:.2f}</text>')
    return (f'<div class="card wrap"><svg viewBox="0 0 {w} {h}" style="width:100%;min-width:480px">{axis}'
            f'<polyline points="{" ".join(best)}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>{dots}</svg>'
            f'<div class="muted">Blue line: best fitness so far. Grey dots: each program. Fitness = mean of points found / 2n.</div></div>')


def status(entry, cube):
    diff = entry["valid_points"] - entry["target"]
    if cube:
        return "<span class=ok>NEW RECORD +%d</span>" % diff if diff > 0 else ("matches record" if diff == 0 else str(diff))
    return "<span class=ok>solved</span>" if diff == 0 else str(diff)


def write_report(run):
    run = Path(run)
    archive = json.loads((run / "archive.json").read_text(encoding="utf-8")) if (run / "archive.json").exists() else []
    best = json.loads((run / "best.json").read_text(encoding="utf-8")) if (run / "best.json").exists() else {}
    config = json.loads((run / "config.json").read_text(encoding="utf-8-sig")) if (run / "config.json").exists() else {}
    cube = config.get("problem") == "cube4"
    title = "No-four-coplanar cube search" if cube else "No-three-in-line search"
    target_word = "Best known" if cube else "Target 2n"
    top = max(archive, key=lambda e: (e["solved"], e["fitness"]), default=None)
    tokens = sum((e.get("usage") or {}).get("output_tokens", 0) + (e.get("usage") or {}).get("input_tokens", 0) for e in archive)

    rows = "".join(
        f"<tr><td>{html.escape(e['id'])}</td><td>{html.escape(e['parent'] or '-')}</td>"
        f"<td>{html.escape(e['idea'])}</td><td>{e['fitness']:.4f}</td><td>{e['solved']}</td>"
        f"<td class='err'>{html.escape(e['error'] or '')}</td></tr>"
        for e in reversed(archive)
    )
    sizes = sorted(best, key=int)
    best_rows = "".join(
        f"<tr><td>{k}</td><td>{best[k]['valid_points']}</td><td>{best[k]['target']}</td>"
        f"<td>{status(best[k], cube)}</td>"
        f"<td>{html.escape(best[k]['program'])}</td></tr>"
        for k in sizes
    )
    grids = "".join(
        f'<div class="card">{(cube_svg if cube else grid_svg)(int(k), best[k]["points"])}<div><b>n = {k}</b> '
        f'<span class="muted">{best[k]["valid_points"]} / {best[k]["target"]} points</span></div></div>'
        for k in sizes
    )
    stats = ""
    if top:
        stats = (f'<div class="stats"><div class="stat"><span class="muted">Programs tried</span><b>{len(archive)}</b></div>'
                 f'<div class="stat"><span class="muted">Best fitness</span><b>{top["fitness"]:.4f}</b></div>'
                 f'<div class="stat"><span class="muted">{"Records beaten" if cube else "Sizes solved (2n)"}</span><b>{sum((best[k]["valid_points"] > best[k]["target"]) if cube else (best[k]["valid_points"] == best[k]["target"]) for k in sizes)} / {len(sizes)}</b></div>'
                 f'<div class="stat"><span class="muted">Best program</span><b>{html.escape(top["id"])}</b></div>'
                 f'<div class="stat"><span class="muted">API tokens</span><b>{tokens:,}</b></div></div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{CSS}</style></head><body><main>
<h1>{title}</h1>
<div class="muted">Run: {html.escape(run.name)}. {"Choose points in an n&times;n&times;n grid with no four on a plane; beat the best known value." if cube else "Place 2n points on an n&times;n grid with no three on a line."}</div>
{stats}
<h2>Progress</h2>{fitness_chart(archive)}
<h2>Best configuration per size</h2>
<div class="wrap"><table><tr><th>n</th><th>Points</th><th>{target_word}</th><th>Status</th><th>Found by</th></tr>{best_rows}</table></div>
<h2>Best grids</h2><div class="grids">{grids}</div>
<h2>All programs</h2>
<div class="wrap"><table><tr><th>Id</th><th>Parent</th><th>Idea</th><th>Fitness</th><th>Solved</th><th>Error</th></tr>{rows}</table></div>
</main></body></html>"""
    (run / "report.html").write_text(page, encoding="utf-8")
    return run / "report.html"
