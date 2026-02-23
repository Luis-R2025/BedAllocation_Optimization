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
    """Build a multi-day HTML matrix heatmap from the accumulated optimized_plan xlsx.

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
    for col in ("unit", "beds_used", "beds", "Optimized Occupancy%", "Overflow"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in sheet 'Optimization occupancy' of {xlsx_path.name}")

    if "unit_description" not in df.columns:
        df["unit_description"] = ""

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    dates = sorted(df["date"].unique())
    date_range = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else dates[0]

    # Pivot occupancy into unit × date matrix
    pct_df = df.pivot_table(index="unit", columns="date", values="Optimized Occupancy%", aggfunc="first")
    bu_df = df.pivot_table(index="unit", columns="date", values="beds_used", aggfunc="first").fillna(0).astype(int)
    beds_df = df.pivot_table(index="unit", columns="date", values="beds", aggfunc="first").fillna(0).astype(int)
    ov_df = df.pivot_table(index="unit", columns="date", values="Overflow", aggfunc="first").fillna(0).astype(int)
    for m in (pct_df, bu_df, beds_df, ov_df):
        m.sort_index(axis=1, inplace=True)

    # Unit descriptions lookup
    desc_map = dict(zip(df["unit"], df["unit_description"]))

    # ---- Build HTML matrix ----
    date_headers = "".join(f'<th class="date-col">{d}</th>' for d in dates)

    rows_html = []
    for unit in sorted(pct_df.index):
        cells = ""
        for d in dates:
            if d in pct_df.columns and not pd.isna(pct_df.loc[unit, d]):
                occ = float(pct_df.loc[unit, d])
                bg = _color(occ)
                fg = _text_color(occ)
                bu = int(bu_df.loc[unit, d])
                bd = int(beds_df.loc[unit, d])
                ov = int(ov_df.loc[unit, d])
                ov_str = f' <span class="badge-overflow">+{ov}</span>' if ov > 0 else ""
                cells += (
                    f'<td class="occ-cell" style="background:{bg};color:{fg};">'
                    f'{occ:.0f}%<br><small>{bu}/{bd}{ov_str}</small></td>'
                )
            else:
                cells += '<td class="occ-cell" style="background:#eee;color:#999;">—</td>'
        desc = desc_map.get(unit, "")
        rows_html.append(
            f'<tr><td class="unit-cell">{unit}</td>'
            f'<td class="desc-cell">{desc}</td>{cells}</tr>'
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Occupancy Heatmap – {date_range}</title>
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
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    background: #fff;
  }}
  th {{
    background: #37474f;
    color: #fff;
    padding: 8px 10px;
    text-align: center;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    white-space: nowrap;
  }}
  td {{
    padding: 6px 10px;
    border-bottom: 1px solid #e0e0e0;
    font-size: 0.85rem;
  }}
  tr:hover td {{
    filter: brightness(1.08);
  }}
  .unit-cell {{
    font-weight: 700;
    white-space: nowrap;
    text-align: left;
  }}
  .desc-cell {{
    color: #555;
    font-size: 0.78rem;
    text-align: left;
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .date-col {{
    writing-mode: vertical-rl;
    text-orientation: mixed;
    min-width: 55px;
  }}
  .occ-cell {{
    text-align: center;
    font-weight: 700;
    min-width: 55px;
  }}
  .occ-cell small {{
    font-weight: 400;
    font-size: 0.7rem;
  }}
  .badge-overflow {{
    background: #d32f2f;
    color: #fff;
    padding: 1px 4px;
    border-radius: 8px;
    font-size: 0.65rem;
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
<p class="subtitle">{date_range} &mdash; {len(dates)} day(s), {len(pct_df.index)} units</p>

<table>
<thead>
<tr>
  <th>Unit</th>
  <th>Description</th>
  {date_headers}
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
