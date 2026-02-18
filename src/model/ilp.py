"""
ILP bed transfers optimization model

This script builds and solves an ILP that attempts to keep unit census within
bed capacity by allowing transfers between compatible units. It uses the
following inputs:

Inputs:
- data/processed/bedroster.csv
    Required columns (case-insensitive):
      date, unit, beds, patient_days (or patient_day)

- data/processed/location_forecast.csv
    Required columns:
      location, admission_avg, discharge_avg

- data/processed/compatibility_matrix.csv
    Long format preferred:
      origin, destination, allowed (0/1)
    (Also accepts 'compatible' or 'count' if present)

Assumptions date logic convention:
- census_date = latest date in bedroster.csv (today-1 midnight census)
- solve_date  = census_date + 1 day
- C0 comes from patient_days for census_date only.

Original ILP logic reproduced:
- Occ[u] = C0[u] - D[u] + A[u] + inflow(u) - outflow(u)
- Occ[u] = In85[u] + Ov85[u] + StayOverflow[u]
- In85[u] <= floor(0.85 * beds[u])
- In85[u] + Ov85[u] <= beds[u]
- Near95[u] >= (In85[u] + Ov85[u]) - floor(0.95 * beds[u])

Objective (weighted objective):
min 1000*sum(StayOverflow) + 1*sum(Transfers) + 0.1*sum(Near95)

Output (matches expected ilp_solution.csv format):
unit, final_census, overflow, beds, transfers_out, transfers_in
"""

from pathlib import Path
import pandas as pd
import math

try:
    import pulp
except Exception:
    pulp = None


def _norm(x) -> str:
    return str(x).strip()


def _detect_col(df: pd.DataFrame, *cands: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def read_compatibility_matrix(path: Path) -> pd.DataFrame:
    """Return long-form compatibility with columns: origin, destination, compatible (0/1)."""
    df = pd.read_csv(path)
    ocol = _detect_col(df, "origin")
    dcol = _detect_col(df, "destination")
    if ocol and dcol:
        vcol = _detect_col(df, "allowed", "compatible", "count")
        out = pd.DataFrame(
            {
                "origin": df[ocol].map(_norm),
                "destination": df[dcol].map(_norm),
            }
        )
        if vcol is None:
            out["compatible"] = 1
            return out

        vals = pd.to_numeric(df[vcol], errors="coerce").fillna(0)
        if vcol.lower() == "count":
            out["compatible"] = (vals.astype(float) > 0).astype(int)
        else:
            out["compatible"] = (vals.astype(float) != 0).astype(int)
        return out

    # Optional wide-matrix fallback
    if df.shape[1] >= 2:
        origins = df.iloc[:, 0].map(_norm).tolist()
        dest_cols = [str(c) for c in df.columns[1:]]
        recs = []
        for i, orig in enumerate(origins):
            for j, col in enumerate(dest_cols):
                val = df.iloc[i, j + 1]
                try:
                    comp = 1 if float(val) != 0 else 0
                except Exception:
                    comp = 1 if str(val).strip() not in ("", "0", "False", "false") else 0
                recs.append({"origin": orig, "destination": _norm(col), "compatible": int(comp)})
        return pd.DataFrame.from_records(recs)

    raise ValueError("Could not parse compatibility_matrix.csv")


def build_and_solve(
    bedroster_csv: Path,
    forecast_csv: Path,
    compat_csv: Path,
    out_csv: Path,
    solver=None,
    transfer_cost: float = 1.0,
    overflow_cost: float = 1000.0,
    near95_cost: float = 0.1,
):
    if pulp is None:
        raise ImportError("pulp is required; please install with `pip install pulp`")

    df_bed = pd.read_csv(bedroster_csv)
    df_for = pd.read_csv(forecast_csv)
    df_compat = read_compatibility_matrix(compat_csv)

    # ---- Detect columns ----
    date_col = _detect_col(df_bed, "date")
    unit_col = _detect_col(df_bed, "unit") or df_bed.columns[0]
    beds_col = _detect_col(df_bed, "beds") or "beds"
    c0_col = _detect_col(df_bed, "patient_days", "patient_day")

    if date_col is None:
        raise ValueError("bedroster.csv must contain a 'date' column")
    if c0_col is None:
        raise ValueError("bedroster.csv must contain 'patient_days' or 'patient_day'")

    loc_col = _detect_col(df_for, "location") or "location"
    a_col = _detect_col(df_for, "admission_avg") or "admission_avg"
    d_col = _detect_col(df_for, "discharge_avg") or "discharge_avg"

    # ---- Census date selection (automation convention) ----
    df_bed = df_bed.copy()
    df_bed[date_col] = pd.to_datetime(df_bed[date_col], errors="coerce", format="%Y-%m-%d")
    census_date = df_bed[date_col].max()
    if pd.isna(census_date):
        raise ValueError("Could not parse any valid dates in bedroster.csv")
    solve_date = census_date + pd.Timedelta(days=1)

    df_day = df_bed[df_bed[date_col] == census_date].copy()
    if df_day.empty:
        raise ValueError(f"No rows found for census_date={census_date.date()} in bedroster.csv")

    # ---- Normalize & coerce ----
    df_day[unit_col] = df_day[unit_col].map(_norm)
    df_day[beds_col] = pd.to_numeric(df_day[beds_col], errors="coerce").fillna(0).astype(int)
    df_day[c0_col] = pd.to_numeric(df_day[c0_col], errors="coerce").fillna(0).astype(int)

    # Unit-level C0 and beds for census_date
    bed_summary = (
        df_day.groupby(unit_col)
        .agg(beds=(beds_col, "max"), C0=(c0_col, "sum"))
        .reset_index()
        .rename(columns={unit_col: "unit"})
    )

    # ---- Forecast ----
    df_for = df_for.copy()
    df_for[loc_col] = df_for[loc_col].map(_norm)
    df_for[a_col] = pd.to_numeric(df_for[a_col], errors="coerce").fillna(0.0)
    df_for[d_col] = pd.to_numeric(df_for[d_col], errors="coerce").fillna(0.0)

    for_summary = df_for[[loc_col, a_col, d_col]].rename(
        columns={loc_col: "unit", a_col: "admit_avg", d_col: "disch_avg"}
    )

    # Keep only units present in bed roster for the census day (faithful to original)
    merged = bed_summary.merge(for_summary, on="unit", how="left")
    merged["A"] = merged["admit_avg"].fillna(0.0).round().astype(int)
    merged["D"] = merged["disch_avg"].fillna(0.0).round().astype(int)

    units = merged["unit"].astype(str).tolist()
    unit_set = set(units)

    beds_dict = dict(zip(merged["unit"], merged["beds"].astype(int)))
    C0_dict = dict(zip(merged["unit"], merged["C0"].astype(int)))
    A_dict = dict(zip(merged["unit"], merged["A"].astype(int)))
    D_dict = dict(zip(merged["unit"], merged["D"].astype(int)))

    # ---- Compatibility arcs ----
    df_compat = df_compat.copy()
    df_compat["origin"] = df_compat["origin"].map(_norm)
    df_compat["destination"] = df_compat["destination"].map(_norm)

    compat_allowed = df_compat[df_compat["compatible"].astype(int) != 0].copy()
    compat_allowed = compat_allowed[compat_allowed["origin"] != compat_allowed["destination"]]

    allowed_arcs = [
        (o, d)
        for o, d in zip(compat_allowed["origin"], compat_allowed["destination"])
        if (o in unit_set and d in unit_set)
    ]

    # ---- Build ILP (original) ----
    prob = pulp.LpProblem("ipl_balance_one_day", pulp.LpMinimize)

    Trans = pulp.LpVariable.dicts("Trans", allowed_arcs, lowBound=0, cat="Integer")
    Occ = pulp.LpVariable.dicts("Occ", units, lowBound=0, cat="Integer")
    In85 = pulp.LpVariable.dicts("In85", units, lowBound=0, cat="Integer")
    Ov85 = pulp.LpVariable.dicts("Ov85", units, lowBound=0, cat="Integer")
    StayOverflow = pulp.LpVariable.dicts("StayOverflow", units, lowBound=0, cat="Integer")
    Near95 = pulp.LpVariable.dicts("Near95", units, lowBound=0, cat="Continuous")

    prob += (
        overflow_cost * pulp.lpSum(StayOverflow[u] for u in units)
        + transfer_cost * pulp.lpSum(Trans[a] for a in allowed_arcs)
        + near95_cost * pulp.lpSum(Near95[u] for u in units)
    )

    for u in units:
        inflow = pulp.lpSum(Trans[(i, u)] for (i, j) in allowed_arcs if j == u)
        outflow = pulp.lpSum(Trans[(u, j)] for (i, j) in allowed_arcs if i == u)

        prob += Occ[u] == C0_dict[u] - D_dict[u] + A_dict[u] + inflow - outflow, f"MassBal_{u}"
        prob += Occ[u] == In85[u] + Ov85[u] + StayOverflow[u], f"Decomp_{u}"

        b = int(beds_dict[u])
        prob += In85[u] <= math.floor(0.85 * b), f"In85Cap_{u}"
        prob += In85[u] + Ov85[u] <= b, f"InBedsCap_{u}"
        prob += Near95[u] >= (In85[u] + Ov85[u]) - math.floor(0.95 * b), f"Near95_{u}"

    solver_to_use = solver or pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver_to_use)

    # ---- Extract transfers ----
    transfer_rows = []
    for (i, j) in allowed_arcs:
        v = int(pulp.value(Trans[(i, j)]) or 0)
        if v > 0:
            transfer_rows.append({"origin": i, "destination": j, "transfers": v})
    transfers_df = pd.DataFrame(transfer_rows)

    # ---- Build expected ilp_solution output ----
    rows = []
    for u in units:
        final_census = int(pulp.value(Occ[u]) or 0)
        overflow = int(pulp.value(StayOverflow[u]) or 0)  # matches original meaning of overflow
        beds = int(beds_dict[u])
        rows.append({"unit": u, "final_census": final_census, "overflow": overflow, "beds": beds})

    sol_df = pd.DataFrame(rows)

    # transfers_in/out columns expected
    if not transfers_df.empty:
        out_sum = transfers_df.groupby("origin")["transfers"].sum().rename("transfers_out").reset_index()
        in_sum = transfers_df.groupby("destination")["transfers"].sum().rename("transfers_in").reset_index()
        sol_df = sol_df.merge(out_sum, left_on="unit", right_on="origin", how="left").drop(columns=["origin"])
        sol_df = sol_df.merge(in_sum, left_on="unit", right_on="destination", how="left").drop(columns=["destination"])
    else:
        sol_df["transfers_out"] = 0
        sol_df["transfers_in"] = 0

    sol_df = sol_df.fillna(0)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sol_df.to_csv(out_csv, index=False)

    return prob, sol_df, transfers_df, {"census_date": census_date, "solve_date": solve_date}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    bed = root / "data" / "processed" / "bedroster.csv"
    fore = root / "data" / "processed" / "location_forecast.csv"
    compat = root / "data" / "processed" / "compatibility_matrix.csv"
    out = root / "outputs" / "inference" / "ilp_solution.csv"

    prob, sol_df, transfers_df, meta = build_and_solve(bed, fore, compat, out)
    if pulp is not None:
        print("Status:", pulp.LpStatus[prob.status])
        print("Objective:", pulp.value(prob.objective))
        print("Census date used:", meta["census_date"].date(), "| Solve date:", meta["solve_date"].date())
    print("Solution written to", out)
