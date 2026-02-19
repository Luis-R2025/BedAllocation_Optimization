"""
Heatmap report – Optimized Occupancy by Unit

Reads:
- outputs/report/optimized_plan_<solve_date>.xlsx   (latest file, after report1 enrichment)

Writes:
- outputs/report/occupancy_heatmap.html
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


# ------------------ CONFIG ------------------
ROOT = Path(".")
REPORT_DIR = ROOT / "outputs" / "report"
OUTDIR = ROOT / "outputs" / "report"


def _find_latest_xlsx(directory: Path) -> Path:
    """Return outputs/report/optimized_plan.xlsx, raising if absent."""
    p = directory / "optimized_plan.xlsx"
    if not p.exists():
        raise FileNotFoundError(
            f"'optimized_plan.xlsx' not found in {directory}. "
            "Run build_and_solve first."
        )
    return p


def _color(value: float) -> str:
    """Return a background colour string based on occupancy %."""
    if value >= 100:
        return "#d32f2f"       # red – at/over capacity
    if value >= 95:
        return "#f57c00"       # orange – near capacity
    if value >= 85:
        return "#fbc02d"       # yellow – moderate
    if value >= 50:
        return "#66bb6a"       # green – healthy
    return "#42a5f5"           # blue  – low occupancy


def _text_color(value: float) -> str:
    """White text on dark backgrounds, black elsewhere."""
    return "#ffffff" if value >= 95 else "#212121"


def generate_heatmap(project_root: Path | None = None) -> Path:
    """Build an HTML heatmap from the enriched optimized_plan xlsx.

    Parameters
    ----------
    project_root : Path, optional
        Overrides the module-level ROOT for path resolution.

    Returns
    -------
    Path to the generated HTML file.
    """
    global ROOT, REPORT_DIR, OUTDIR
    if project_root is not None:
        ROOT = Path(project_root)
        REPORT_DIR = ROOT / "outputs" / "report"
        OUTDIR = ROOT / "outputs" / "report"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    xlsx_path = _find_latest_xlsx(REPORT_DIR)
    print(f"[report2] Reading {xlsx_path.name}")

    df = pd.read_excel(xlsx_path, sheet_name="Optimization occupancy", engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns
    for col in ("unit", "unit_description", "beds_used", "beds", "Optimized Occupancy%", "Overflow"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in sheet 'Optimization occupancy' of {xlsx_path.name}")

    df = df.sort_values("Optimized Occupancy%", ascending=False).reset_index(drop=True)

    report_date = df["date"].iloc[0] if "date" in df.columns else "N/A"

    # ---- Build HTML ----
    rows_html = []
    for _, r in df.iterrows():
        occ = float(r["Optimized Occupancy%"])
        bg = _color(occ)
        fg = _text_color(occ)
        overflow_badge = (
            f'<span class="badge-overflow">{int(r["Overflow"])}</span>'
            if int(r.get("Overflow", 0)) > 0
            else str(int(r.get("Overflow", 0)))
        )
        rows_html.append(
            f"<tr>"
            f'<td class="unit-cell">{r["unit"]}</td>'
            f'<td class="desc-cell">{r["unit_description"]}</td>'
            f'<td class="num">{int(r["beds_used"])}</td>'
            f'<td class="num">{int(r["beds"])}</td>'
            f'<td class="occ-cell" style="background:{bg};color:{fg};">{occ:.1f}%</td>'
            f'<td class="num">{overflow_badge}</td>'
            f"</tr>"
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Occupancy Heatmap – {report_date}</title>
<style>
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: 2rem;
    background: #f5f5f5;
  }}
  h1 {{
    color: #333;
    margin-bottom: 0.2rem;
  }}
  .subtitle {{
    color: #777;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    max-width: 900px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    background: #fff;
  }}
  th {{
    background: #37474f;
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  td {{
    padding: 8px 14px;
    border-bottom: 1px solid #e0e0e0;
    font-size: 0.9rem;
  }}
  tr:hover td {{
    background: #f0f4f8;
  }}
  .unit-cell {{
    font-weight: 700;
  }}
  .desc-cell {{
    color: #555;
    font-size: 0.85rem;
  }}
  .num {{
    text-align: center;
  }}
  .occ-cell {{
    text-align: center;
    font-weight: 700;
    border-radius: 4px;
    min-width: 70px;
  }}
  .badge-overflow {{
    background: #d32f2f;
    color: #fff;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 700;
  }}
  .legend {{
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    font-size: 0.82rem;
    flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }}
  .legend-swatch {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
    display: inline-block;
  }}
</style>
</head>
<body>
<h1>Occupancy Heatmap</h1>
<p class="subtitle">Solve date: {report_date}</p>

<table>
<thead>
<tr>
  <th>Unit</th>
  <th>Description</th>
  <th>Beds Used</th>
  <th>Beds</th>
  <th>Occupancy %</th>
  <th>Overflow</th>
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>

<div class="legend">
  <div class="legend-item"><span class="legend-swatch" style="background:#d32f2f"></span> &ge;100 %</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#f57c00"></span> 95–99 %</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#fbc02d"></span> 85–94 %</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#66bb6a"></span> 50–84 %</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#42a5f5"></span> &lt; 50 %</div>
</div>

</body>
</html>
"""

    out_path = OUTDIR / "occupancy_heatmap.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[done] Wrote heatmap -> {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    generate_heatmap()
