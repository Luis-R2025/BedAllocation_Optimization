
"""
OUTPUT FORMAT ALEX VERSION

Reads:
- outputs/inference/ilp_solution.csv          (Luis standard output)
- outputs/inference/ilp_transfers.csv         (optional but recommended)

Writes:
- outputs/reports/optimized_plan_mnemonic.csv

Final columns EXACTLY:
date, unit, unit_description, beds_used, beds, Optimized Occupancy%, Overflow, destination unit, number of transfer
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


# ------------------ CONFIG ------------------
ROOT = Path(".")  # project root (adjust if needed)

SOL_PATH = ROOT / "outputs" / "inference" / "ilp_solution.csv"
TRANS_PATH = ROOT / "outputs" / "inference" / "ilp_transfers.csv"   # create this in run 1 (recommended)

OUTDIR = ROOT / "outputs" / "report"


# ------------------ Helpers ------------------
def infer_solve_date_from_bedroster() -> str:
    """
    If your pipeline doesn't save solve_date anywhere, we can infer:
      solve_date = max(date in data/processed/bedroster.csv) + 1 day
    Falls back to today's date if files aren't found / parse fails.
    """
    bed_path = ROOT / "data" / "processed" / "bedroster.csv"
    if not bed_path.exists():
        return pd.Timestamp.today().strftime("%Y-%m-%d")

    bed = pd.read_csv(bed_path, sep=None, engine="python")
    bed.columns = [c.strip() for c in bed.columns]

    # tolerate header issues
    date_col = None
    for c in bed.columns:
        if str(c).strip().lower() == "date":
            date_col = c
            break
    if date_col is None:
        return pd.Timestamp.today().strftime("%Y-%m-%d")

    bed[date_col] = pd.to_datetime(bed[date_col], errors="coerce", dayfirst=True)
    census_dt = bed[date_col].dropna().max()
    if pd.isna(census_dt):
        return pd.Timestamp.today().strftime("%Y-%m-%d")

    solve_dt = census_dt + pd.Timedelta(days=1)
    return solve_dt.strftime("%Y-%m-%d")


def generate_report(project_root: Path | None = None) -> Path:
    """Build the optimized-plan report and return the path to the output CSV.

    Parameters
    ----------
    project_root : Path, optional
        If provided, overrides the module-level ROOT so that paths resolve
        correctly when called from another working directory.
    """
    global ROOT, SOL_PATH, TRANS_PATH, OUTDIR
    if project_root is not None:
        ROOT = Path(project_root)
        SOL_PATH = ROOT / "outputs" / "inference" / "ilp_solution.csv"
        TRANS_PATH = ROOT / "outputs" / "inference" / "ilp_transfers.csv"
        OUTDIR = ROOT / "outputs" / "report"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ------------------ Load Luis solution ------------------
    if not SOL_PATH.exists():
        raise FileNotFoundError(f"Missing Luis solution file: {SOL_PATH}")

    sol = pd.read_csv(SOL_PATH)
    sol.columns = [c.strip() for c in sol.columns]

    required = {"unit", "final_census", "overflow", "beds"}
    missing = required - set(sol.columns)
    if missing:
        raise ValueError(f"{SOL_PATH} missing required cols {missing}. Found: {list(sol.columns)}")

    sol["unit"] = sol["unit"].astype(str).str.strip()
    sol["final_census"] = pd.to_numeric(sol["final_census"], errors="coerce").fillna(0).astype(int)
    sol["overflow"] = pd.to_numeric(sol["overflow"], errors="coerce").fillna(0).astype(int)
    sol["beds"] = pd.to_numeric(sol["beds"], errors="coerce").fillna(0).astype(int)

    # beds_used = staffed in-beds portion (exclude overflow)
    sol["beds_used"] = sol[["final_census", "beds"]].min(axis=1).astype(int)

    # Optimized Occupancy% based on final census (includes overflow) / beds
    sol["Optimized Occupancy%"] = (
        100.0 * sol["final_census"] / sol["beds"].replace(0, np.nan)
    ).fillna(0.0).round(1)

    sol["Overflow"] = sol["overflow"].astype(int)

    # Solve date inference
    solve_date = infer_solve_date_from_bedroster()

    # ------------------ Unit description mapping ------------------
    loc_desc_path = ROOT / "data" / "raw" / "df_location_description.csv"
    if loc_desc_path.exists():
        loc_desc = pd.read_csv(loc_desc_path)
        loc_desc.columns = [c.strip() for c in loc_desc.columns]
        mnemonic_to_name = dict(zip(
            loc_desc["Mnemonic"].astype(str).str.strip(),
            loc_desc["Name"].astype(str).str.strip(),
        ))
        sol["unit_description"] = sol["unit"].map(mnemonic_to_name).fillna("")
    else:
        print(f"[warn] {loc_desc_path} not found – unit_description will be blank")
        sol["unit_description"] = ""

    # ------------------ Transfers (optional) ------------------
    sol["destination unit"] = ""
    sol["number of transfer"] = ""

    if TRANS_PATH.exists():
        tr = pd.read_csv(TRANS_PATH)
        tr.columns = [c.strip() for c in tr.columns]

        if {"origin", "destination", "transfers"}.issubset(tr.columns):
            tr["origin"] = tr["origin"].astype(str).str.strip()
            tr["destination"] = tr["destination"].astype(str).str.strip()
            tr["transfers"] = pd.to_numeric(tr["transfers"], errors="coerce").fillna(0).astype(int)

            tr = tr[tr["transfers"] > 0].copy()

            if not tr.empty:
                compact = (
                    tr.groupby("origin", sort=False)
                      .agg(
                          **{
                              "destination unit": ("destination", lambda s: " ; ".join(s.astype(str).tolist())),
                              "number of transfer": ("transfers", "sum"),
                          }
                      )
                      .reset_index()
                      .rename(columns={"origin": "unit"})
                )

                sol = sol.merge(compact, on="unit", how="left", suffixes=("", "_y"))
                sol["destination unit"] = sol["destination unit"].fillna("")
                sol["number of transfer"] = sol["number of transfer"].fillna("")

        else:
            print("[warn] ilp_transfers.csv exists but doesn't have columns: origin, destination, transfers. Leaving transfer columns blank.")

    # ------------------ Final table (exact columns) ------------------
    final = pd.DataFrame({
        "date": solve_date,
        "unit": sol["unit"].astype(str).str.strip(),
        "unit_description": sol["unit_description"].astype(str),
        "beds_used": sol["beds_used"].astype(int),
        "beds": sol["beds"].astype(int),
        "Optimized Occupancy%": sol["Optimized Occupancy%"],
        "Overflow": sol["Overflow"].astype(int),
        "destination unit": sol.get("destination unit", "").fillna("").astype(str),
        "number of transfer": sol.get("number of transfer", "").replace({np.nan: ""}),
    })

    # Optional: stable ordering
    final = final.sort_values("unit").reset_index(drop=True)

    out_csv = OUTDIR / "optimized_plan.csv"
    final.to_csv(out_csv, index=False)

    print("[done] Wrote  output report:", out_csv.resolve())
    return out_csv


if __name__ == "__main__":
    generate_report()
