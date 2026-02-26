"""
metrics_history – generates/updates outputs/inference/metrics_general_history.csv

Reads:
- data/processed/bedroster.csv
- data/processed/location_forecast.csv
- data/processed/compatibility_matrix.csv

Writes:
- outputs/inference/metrics_general_history.csv

Columns:
  date | metric_scope | unit | metric_name | metric_value | run_timestamp
"""

from __future__ import annotations

import datetime
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.model import ilp


# ------------------ CONFIG ------------------
DATA_DIR = Path("data") / "processed"
HISTORY_PATH = Path("outputs") / "inference" / "metrics_general_history.csv"


# ------------------ Helpers ------------------
def _status_flag(occ_pct: float, overflow: float) -> str:
    if overflow > 0:
        return "Overflow"
    if occ_pct >= 100:
        return "Critical (At Capacity)"
    if occ_pct >= 95:
        return "Near Capacity"
    if occ_pct >= 90:
        return "Danger"
    if occ_pct >= 85:
        return "Acceptable"
    return "Low Utilization"


def _append_history(history_path: Path, new_df: pd.DataFrame) -> None:
    """Append rows, replacing any existing rows that share the same natural key."""
    key_cols = ["date", "metric_scope", "unit", "metric_name"]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists() and history_path.stat().st_size > 0:
        old = pd.read_csv(history_path, encoding="utf-8-sig")
        if set(key_cols).issubset(old.columns):
            new_keys = new_df[key_cols].astype(str).drop_duplicates()
            merged = old.merge(new_keys, on=key_cols, how="left", indicator=True)
            old = old.loc[merged["_merge"] == "left_only"].copy()
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    combined.sort_values(["date", "metric_scope", "unit", "metric_name"], inplace=True)
    combined.to_csv(history_path, index=False, encoding="utf-8-sig")


# ------------------ Main ------------------
def generate_metrics(project_root: Path | None = None, solve_date: Optional[str] = None) -> Path:
    """Run the ILP and append per-unit + system KPIs to the metrics history CSV.

    Parameters
    ----------
    project_root : Path, optional
        Overrides default paths so the function works from any working directory.
    solve_date : str, optional
        Force a specific solve date (YYYY-MM-DD). Defaults to auto-detect.

    Returns
    -------
    Path to the updated metrics_general_history.csv file.
    """
    data_dir = (Path(project_root) / "data" / "processed") if project_root else DATA_DIR
    history_path = (Path(project_root) / "outputs" / "inference" / "metrics_general_history.csv") if project_root else HISTORY_PATH

    # Solve ILP into a temp file to avoid polluting real inference outputs
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as _tmp:
        tmp_csv = Path(_tmp.name)
    try:
        _, sol_df, transfers_simple, _ = ilp.build_and_solve(
            bedroster_csv=data_dir / "bedroster.csv",
            forecast_csv=data_dir / "location_forecast.csv",
            compat_csv=data_dir / "compatibility_matrix.csv",
            out_csv=tmp_csv,
            write_alex_excel=False,
            solve_date=solve_date,
        )
    finally:
        tmp_csv.unlink(missing_ok=True)

    solve_date = sol_df["date"].iloc[0] if solve_date is None else solve_date
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sol_idx = sol_df.set_index("unit")
    units: List[str] = sol_df["unit"].tolist()

    used_s = (sol_idx["final_census"] - sol_idx["overflow"]).clip(lower=0)
    overflow_s = sol_idx["overflow"]
    beds_s = sol_idx["beds"]

    total_beds = float(beds_s.sum())
    total_used = float(used_s.sum())
    total_overflow = float(overflow_s.sum())
    system_occ_pct = (100.0 * total_used / total_beds) if total_beds > 0 else 0.0
    total_transfers = float(transfers_simple["transfers"].sum()) if not transfers_simple.empty else 0.0

    rows = [
        {"date": solve_date, "metric_scope": "system", "unit": "", "metric_name": "total_transfers",         "metric_value": total_transfers,  "run_timestamp": run_ts},
        {"date": solve_date, "metric_scope": "system", "unit": "", "metric_name": "system_occupancy_percent","metric_value": system_occ_pct,   "run_timestamp": run_ts},
        {"date": solve_date, "metric_scope": "system", "unit": "", "metric_name": "total_overflow",          "metric_value": total_overflow,   "run_timestamp": run_ts},
    ]
    for u in units:
        beds_u = float(beds_s.get(u, 0))
        used_u = float(used_s.get(u, 0))
        overflow_u = float(overflow_s.get(u, 0))
        occ_u = (100.0 * used_u / beds_u) if beds_u > 0 else 0.0
        rows.append({"date": solve_date, "metric_scope": "unit", "unit": u, "metric_name": "occupancy_percent", "metric_value": occ_u,                                  "run_timestamp": run_ts})
        rows.append({"date": solve_date, "metric_scope": "unit", "unit": u, "metric_name": "overflow",          "metric_value": overflow_u,                              "run_timestamp": run_ts})
        rows.append({"date": solve_date, "metric_scope": "unit", "unit": u, "metric_name": "status_flag",       "metric_value": _status_flag(occ_u, overflow_u),         "run_timestamp": run_ts})

    _append_history(history_path, pd.DataFrame(rows))
    print(f"[metrics_history] Updated: {history_path}")
    return history_path


# ------------------ CLI ENTRYPOINT ------------------
if __name__ == "__main__":
    out = generate_metrics()
    print(f"[metrics_history] Done -> {out}")
