"""ILP bed transfers optimization model (Luis pipeline) rewritten to match Alex script EXACTLY.

This module plugs into Luis' VS Code pipeline structure while reproducing Alex's:
- date convention (solve_date inferred; census_date = solve_date-1)
- data cleaning (delimiter autodetect; unit aliases; dayfirst date parsing)
- ILP formulation and weights
- Alex-style artifacts (Excel + heatmap + transfer labeling backlog vs new transfer)

Pipeline compatibility:
- Public API stays: build_and_solve(bedroster_csv, forecast_csv, compat_csv, out_csv, ...)
- Still writes outputs/inference/ilp_solution.csv and ilp_transfers.csv for Luis reports.

Alex-style artifacts written (default ON):
- outputs/report/optimized_plan.xlsx
- outputs/report/Heatmap_OptimizedOcc.png

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

try:
    import pulp as pl
except Exception:  # pragma: no cover
    pl = None


# ------------------------- Unit mnemonic normalization (MATCH ALEX) -------------------------
UNIT_ALIASES: Dict[str, str] = {
    "4E-ORTHOPEDIE/ORL/PLASTIE": "4E",
    "3D-PEDIATRIE": "3D",
    "4A-CHIR.GENERALE/GYN-ONCO/URO": "4A",
    "4F-NEPHROLOGIE": "4F",
    "3F-EVAL./READAPTATION/AVC": "3F",
    "3FSP-SOINS PALLIATIFS": "3FSP",
    "3D PEDOPSYCHIATRIE": "3DPEDOPSY",
    "4C-MED GEN/INTERNE/TELEMETRIE": "4C",
    "4D ONCOLOGIE": "4D",
    "3B-MERE ENFANT": "3B",
    "3B-GYNECO/OBS": "3B",
    "SOINS INT MED CHIRURGICAUX": "SCHIR",
    "SOINS CORONARIENS": "SCOR",
    "SOINS INTERMEDIAIRES": "SINTER",
}


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Robust CSV reader: auto-detect delimiter (comma vs semicolon), strips whitespace from headers."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _clean_text(s: pd.Series) -> pd.Series:
    s = (
        s.astype(str)
        .str.replace("\u00A0", " ", regex=False)  # NBSP
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return s.map(lambda x: UNIT_ALIASES.get(x, x))


def _detect_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Find a column in df whose name matches any candidate (case-insensitive, strip)."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in cands:
        key = str(cand).strip().lower()
        if key in lower:
            return lower[key]
    return None


def _require_cols(where: str, **logical_to_col: Optional[str]) -> None:
    missing = [k for k, v in logical_to_col.items() if v is None]
    if missing:
        raise ValueError(f"{where}: missing required columns {missing}. Found: {list(logical_to_col.values())}")


# ------------------------- Compatibility loading (MATCH ALEX) -------------------------
def load_compat(path_like: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Read long-form compatibility. Requires origin, destination. Accepts allowed/compatible/count."""
    df = read_csv_auto(path_like)

    ocol = _detect_col(df, "origin")
    dcol = _detect_col(df, "destination")
    if ocol is None or dcol is None:
        raise ValueError(f"{path_like}: must include origin and destination columns. Found: {list(df.columns)}")

    vcol = _detect_col(df, "allowed", "compatible", "count")

    df = df.copy()
    df[ocol] = _clean_text(df[ocol])
    df[dcol] = _clean_text(df[dcol])

    if vcol is None:
        ok = pd.Series([True] * len(df))
    else:
        vals = pd.to_numeric(df[vcol], errors="coerce").fillna(0)
        ok = (vals > 0) if str(vcol).strip().lower() == "count" else (vals != 0)

    allowed_df = df[ok & (df[ocol] != df[dcol])].copy()  # removes self-loops
    allowed = list(allowed_df[[ocol, dcol]].itertuples(index=False, name=None))

    if not allowed:
        raise ValueError("No allowed arcs found (all flags are 0 or only self-loops).")

    compat_units = sorted(set(allowed_df[ocol]).union(set(allowed_df[dcol])))
    return compat_units, allowed


# ------------------------- Date inference (MATCH ALEX) -------------------------
def infer_solve_date_from_bedroster(bedroster_csv: Path) -> str:
    bed = read_csv_auto(bedroster_csv)

    date_col = _detect_col(bed, "date")
    if date_col is None:
        raise ValueError(f"{bedroster_csv} must have a 'date' column. Found: {list(bed.columns)}")

    dt = pd.to_datetime(bed[date_col], errors="coerce", dayfirst=True)
    census_date = dt.max()
    if pd.isna(census_date):
        raise ValueError(f"Could not parse any valid dates in {bedroster_csv} date column.")

    solve_date = census_date + pd.Timedelta(days=1)
    return solve_date.strftime("%Y-%m-%d")


def _census_str_from_solve(solve_date_str: str) -> str:
    solve_dt = pd.to_datetime(solve_date_str, errors="raise")
    census_dt = solve_dt - pd.Timedelta(days=1)
    return census_dt.strftime("%Y-%m-%d")


# ------------------------- Inputs (MATCH ALEX) -------------------------
@dataclass
class ModelInputs:
    solve_date: str
    census_date: str
    units: List[str]
    beds_dict: Dict[str, int]
    C0_dict: Dict[str, int]
    A_dict: Dict[str, int]
    D_dict: Dict[str, int]
    allowed_arcs: List[Tuple[str, str]]


def _load_beds_and_raw_census(bedroster_csv: Path, solve_date_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bed = read_csv_auto(bedroster_csv)

    date_col = _detect_col(bed, "date")
    unit_col = _detect_col(bed, "unit")
    beds_col = _detect_col(bed, "beds")
    pdays_col = _detect_col(bed, "patient_days", "patient_day")

    _require_cols(str(bedroster_csv), date=date_col, unit=unit_col, beds=beds_col, patient_days=pdays_col)

    census_str = _census_str_from_solve(solve_date_str)

    bed = bed.copy()
    bed[date_col] = pd.to_datetime(bed[date_col], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    bed[unit_col] = _clean_text(bed[unit_col])
    bed[beds_col] = pd.to_numeric(bed[beds_col], errors="coerce").fillna(0).astype(int)
    bed[pdays_col] = pd.to_numeric(bed[pdays_col], errors="coerce").fillna(0).astype(int)

    day = bed[bed[date_col] == census_str].copy()
    if day.empty:
        avail = sorted(bed[date_col].dropna().unique().tolist())
        raise ValueError(
            f"No rows in {bedroster_csv} for census_date={census_str} (solve_date={solve_date_str}). "
            f"Available dates (sample): {avail[:10]} ... (n={len(avail)})"
        )

    beds_df = (
        day[[unit_col, beds_col]]
        .rename(columns={unit_col: "Unit", beds_col: "Beds"})
        .groupby("Unit", as_index=False)["Beds"].max()
    )

    raw_df = (
        day[[unit_col, pdays_col]]
        .rename(columns={unit_col: "Unit", pdays_col: "Raw_Census"})
        .groupby("Unit", as_index=False)["Raw_Census"].sum()
    )

    return beds_df, raw_df


def _load_forecast(forecast_csv: Path) -> pd.DataFrame:
    f = read_csv_auto(forecast_csv)

    loc_col = _detect_col(f, "location", "loc", "unit", "unite", "location ")
    a_col = _detect_col(f, "admission_avg", "admit_avg", "admissions_avg", "admissions", "admission")
    d_col = _detect_col(f, "discharge_avg", "disch_avg", "discharges_avg", "discharges", "discharge")

    _require_cols(str(forecast_csv), location=loc_col, admission_avg=a_col, discharge_avg=d_col)

    return pd.DataFrame(
        {
            "Unit": _clean_text(f[loc_col]),
            "Admit_Avg": pd.to_numeric(f[a_col], errors="coerce").fillna(0.0),
            "Disch_Avg": pd.to_numeric(f[d_col], errors="coerce").fillna(0.0),
        }
    )


def build_inputs(
    bedroster_csv: Path,
    forecast_csv: Path,
    compat_csv: Path,
    solve_date: Optional[str] = None,
) -> ModelInputs:
    if solve_date is None:
        solve_date = infer_solve_date_from_bedroster(bedroster_csv)
    census_date = _census_str_from_solve(solve_date)

    beds_df, raw_df = _load_beds_and_raw_census(bedroster_csv, solve_date)
    fc_df = _load_forecast(forecast_csv)
    _, allowed = load_compat(compat_csv)

    df = beds_df.merge(raw_df, on="Unit", how="left").merge(fc_df, on="Unit", how="left")

    if df["Raw_Census"].isna().any():
        missing = df[df["Raw_Census"].isna()]["Unit"].tolist()
        raise ValueError(f"Units missing Raw_Census from bedroster for solve_date={solve_date}: {missing}")

    df["A"] = df["Admit_Avg"].fillna(0).round().astype(int)
    df["D"] = df["Disch_Avg"].fillna(0).round().astype(int)

    units = df["Unit"].astype(str).tolist()
    unit_set = set(units)
    allowed_arcs = [(i, j) for (i, j) in allowed if (i in unit_set and j in unit_set)]

    # Alex prints and keeps going even if 0 allowed arcs; but the model would be degenerate.
    if not allowed_arcs:
        raise ValueError("No allowed arcs after filtering compatibility to the modeled unit set.")

    beds_dict = dict(zip(df["Unit"], df["Beds"].astype(int)))
    C0_dict = dict(zip(df["Unit"], df["Raw_Census"].astype(int)))
    A_dict = dict(zip(df["Unit"], df["A"].astype(int)))
    D_dict = dict(zip(df["Unit"], df["D"].astype(int)))

    return ModelInputs(
        solve_date=solve_date,
        census_date=census_date,
        units=units,
        beds_dict=beds_dict,
        C0_dict=C0_dict,
        A_dict=A_dict,
        D_dict=D_dict,
        allowed_arcs=allowed_arcs,
    )


# ------------------------- Solve (MATCH ALEX) -------------------------
def _solve_ilp(
    inputs: ModelInputs,
    transfer_cost: float = 1.0,
    overflow_cost: float = 1000.0,
    near95_cost: float = 0.1,
    solver=None,
):
    if pl is None:
        raise ImportError("pulp is required; please install with `pip install 'pulp==2.8.0'`")

    locations = inputs.units
    beds_dict = inputs.beds_dict
    C0_dict = inputs.C0_dict
    A_dict = inputs.A_dict
    D_dict = inputs.D_dict
    allowed = inputs.allowed_arcs

    model = pl.LpProblem(f"OneDay_{inputs.solve_date}", pl.LpMinimize)

    Occ = pl.LpVariable.dicts("Occ", [(u, 1) for u in locations], lowBound=0, cat=pl.LpInteger)
    In85 = pl.LpVariable.dicts("In85", [(u, 1) for u in locations], lowBound=0, cat=pl.LpInteger)
    Ov85 = pl.LpVariable.dicts("Ov85", [(u, 1) for u in locations], lowBound=0, cat=pl.LpInteger)
    StayOverflow = pl.LpVariable.dicts("StayOverflow", [(u, 1) for u in locations], lowBound=0, cat=pl.LpInteger)
    Trans = pl.LpVariable.dicts("Trans", [(i, j, 1) for (i, j) in allowed], lowBound=0, cat=pl.LpInteger)
    Near95 = pl.LpVariable.dicts("Near95", [(u, 1) for u in locations], lowBound=0, cat=pl.LpContinuous)

    for u in locations:
        inflow = pl.lpSum(Trans[(i, u, 1)] for (i, j) in allowed if j == u)
        outflow = pl.lpSum(Trans[(u, j, 1)] for (i, j) in allowed if i == u)
        model += (Occ[(u, 1)] == C0_dict[u] - D_dict[u] + A_dict[u] + inflow - outflow, f"MassBal_{u}")

    for u in locations:
        b = int(beds_dict[u])
        model += (Occ[(u, 1)] == In85[(u, 1)] + Ov85[(u, 1)] + StayOverflow[(u, 1)], f"Decomp_{u}")
        model += (In85[(u, 1)] <= math.floor(0.85 * b), f"In85Cap_{u}")
        model += (In85[(u, 1)] + Ov85[(u, 1)] <= b, f"InBedsCap_{u}")
        cap95 = math.floor(0.95 * b)
        model += (Near95[(u, 1)] >= In85[(u, 1)] + Ov85[(u, 1)] - cap95, f"Near95Def_{u}")

    model += (
        overflow_cost * pl.lpSum(StayOverflow[(u, 1)] for u in locations)
        + transfer_cost * pl.lpSum(Trans[(i, j, 1)] for (i, j) in allowed)
        + near95_cost * pl.lpSum(Near95[(u, 1)] for u in locations)
    )

    solver_to_use = solver or pl.PULP_CBC_CMD(msg=False)
    model.solve(solver_to_use)

    return model, Occ, In85, Ov85, StayOverflow, Trans, Near95


def _build_alex_transfers_tab(inputs: ModelInputs, Trans) -> pd.DataFrame:
    """Exact Alex transfer labeling: split each origin's outflow into backlog vs new transfer."""
    day_str = inputs.solve_date
    locations = inputs.units
    allowed = inputs.allowed_arcs

    backlog0 = {u: max(0, int(inputs.C0_dict[u]) - int(inputs.beds_dict[u])) for u in locations}

    rows: List[List[object]] = []
    for u in locations:
        out_arcs = [(u, v) for (i, v) in allowed if i == u]
        # NOTE: Alex used int(round(value or 0))
        tcounts = {v: int(round(pl.value(Trans[(u, v, 1)]) or 0)) for v in [j for (_, j) in out_arcs]}
        tot_out = sum(tcounts.values())
        if tot_out == 0:
            continue

        bl_rem = min(backlog0.get(u, 0), tot_out)

        if bl_rem > 0:
            shares = {v: (tcounts[v] / tot_out) for v in tcounts}
            alloc = {v: int(shares[v] * bl_rem) for v in tcounts}
            rem = bl_rem - sum(alloc.values())
            if rem > 0:
                fracs = sorted([(shares[v] * bl_rem - alloc[v], v) for v in tcounts], reverse=True)
                for k in range(rem):
                    alloc[fracs[k][1]] += 1
        else:
            alloc = {v: 0 for v in tcounts}

        for v in tcounts:
            tij = tcounts[v]
            bl = min(alloc[v], tij)
            nw = tij - bl
            if bl > 0:
                rows.append([day_str, u, v, bl, "backlog"])
            if nw > 0:
                rows.append([day_str, u, v, nw, "new transfer"])

    return pd.DataFrame(
        rows,
        columns=["date", "origin unit", "destination unit", "patient (number)", "type (new transfer or backlog)"],
    )


def _extract_outputs(inputs: ModelInputs, Occ, In85, Ov85, StayOverflow, Trans, Near95):
    locations = inputs.units
    beds_dict = inputs.beds_dict

    beds_s = pd.Series({u: int(beds_dict[u]) for u in locations}, name="beds").astype(float)
    in85_s = pd.Series({u: pl.value(In85[(u, 1)]) for u in locations}, name="In85")
    ov85_s = pd.Series({u: pl.value(Ov85[(u, 1)]) for u in locations}, name="Ov85")
    stay_s = pd.Series({u: pl.value(StayOverflow[(u, 1)]) for u in locations}, name="Overflow")
    occ_s = pd.Series({u: pl.value(Occ[(u, 1)]) for u in locations}, name="TrueCensus")
    near95_s = pd.Series({u: pl.value(Near95[(u, 1)]) for u in locations}, name="PtLast5%")

    staffed_used = in85_s + ov85_s
    true_pct = (100.0 * occ_s / beds_s.replace(0, np.nan)).fillna(0.0)

    occ_tab = pd.DataFrame(
        {
            "date": inputs.solve_date,
            "unit": locations,
            "beds_used": staffed_used.fillna(0).round().astype(int).values,
            "beds": beds_s.fillna(0).round().astype(int).values,
            "Optimized Occupancy%": true_pct.round(1).values,
            "PtLast5%": near95_s.fillna(0).round().astype(int).values,
            "Overflow": stay_s.fillna(0).round().astype(int).values,
        }
    )

    transfers_tab = _build_alex_transfers_tab(inputs, Trans)

    # Compact transfer info for main sheet (EXACT Alex columns)
    if not transfers_tab.empty:
        transfers_tab["patient (number)"] = (
            pd.to_numeric(transfers_tab["patient (number)"], errors="coerce").fillna(0).astype(int)
        )
        transfers_compact = (
            transfers_tab.groupby(["date", "origin unit"], dropna=False, sort=False)
            .agg(
                **{
                    "destination unit": ("destination unit", lambda s: " ; ".join(s.astype(str).tolist())),
                    "number of transfer": ("patient (number)", lambda s: int(np.sum(s.values))),
                    "type of transfer": (
                        "type (new transfer or backlog)",
                        lambda s: ", ".join(sorted({t.strip() for t in s.dropna().astype(str)})),
                    ),
                }
            )
            .reset_index()
            .rename(columns={"origin unit": "unit"})
            [["date", "unit", "destination unit", "number of transfer", "type of transfer"]]
        )
        occ_tab = occ_tab.merge(transfers_compact, on=["date", "unit"], how="left")

    overflows_tab = pd.DataFrame(
        [[inputs.solve_date, u, int(round(stay_s[u] or 0))] for u in locations if int(round(stay_s[u] or 0)) > 0],
        columns=["date", "unit", "overflows number"],
    )

    # Pipeline solution output (Luis)
    rows = []
    for u in locations:
        rows.append(
            {
                "unit": u,
                "final_census": int(round(occ_s[u] or 0)),
                "overflow": int(round(stay_s[u] or 0)),
                "beds": int(beds_dict[u]),
            }
        )
    sol_df = pd.DataFrame(rows)

    if not transfers_tab.empty:
        # aggregate transfers out/in from Alex transfers_tab
        out_sum = (
            transfers_tab.groupby("origin unit")["patient (number)"].sum().rename("transfers_out").reset_index()
        )
        in_sum = (
            transfers_tab.groupby("destination unit")["patient (number)"].sum().rename("transfers_in").reset_index()
        )
        sol_df = sol_df.merge(out_sum, left_on="unit", right_on="origin unit", how="left").drop(columns=["origin unit"])
        sol_df = sol_df.merge(in_sum, left_on="unit", right_on="destination unit", how="left").drop(
            columns=["destination unit"]
        )
    else:
        sol_df["transfers_out"] = 0
        sol_df["transfers_in"] = 0

    sol_df = sol_df.fillna(0)
    sol_df["date"] = inputs.solve_date

    return sol_df, transfers_tab, overflows_tab, occ_tab


# ------------------------- Alex-style Heatmap + Excel (MATCH ALEX) -------------------------
def _write_heatmap_png(occ_tab: pd.DataFrame, out_png: Path, vmin: float = 0.0, vmax: float = 110.0) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    day_str = str(occ_tab["date"].iloc[0])
    locations = occ_tab["unit"].astype(str).tolist()

    pct_df = pd.DataFrame({day_str: occ_tab["Optimized Occupancy%"].astype(float).values}, index=locations)
    beds_used_mat = pd.DataFrame({day_str: occ_tab["beds_used"].astype(int).values}, index=locations)
    beds_mat = pd.DataFrame({day_str: occ_tab["beds"].astype(int).values}, index=locations)
    ov_mat = pd.DataFrame({day_str: occ_tab["Overflow"].astype(int).values}, index=locations)

    cmap = mpl.colormaps.get_cmap("RdYlGn_r") if hasattr(mpl, "colormaps") else plt.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(pct_df.index) + 1)))
    im = ax.imshow(pct_df.values.astype(float), aspect="auto", cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))

    ax.set_yticks(range(len(pct_df.index)))
    ax.set_yticklabels(pct_df.index)
    ax.set_xticks(range(len(pct_df.columns)))
    ax.set_xticklabels(pct_df.columns, rotation=0)

    for i in range(len(pct_df.index)):
        for j in range(len(pct_df.columns)):
            if pd.isna(pct_df.iloc[i, j]):
                continue
            valp = float(pct_df.iloc[i, j])
            bu_i = int(beds_used_mat.iloc[i, j])
            bd_i = int(beds_mat.iloc[i, j])
            ov_i = int(ov_mat.iloc[i, j])
            label = f"{valp:.0f}%\n{bu_i}/{bd_i}" + (f" +{ov_i}" if ov_i > 0 else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=("black" if valp < 80 else "white"))

    ax.set_title(f"Optimized occupancy (with beds_used/beds + overflow) — {day_str}")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Optimized occupancy %")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def _write_alex_excel(occ_tab: pd.DataFrame, overflows_tab: pd.DataFrame, transfers_tab: pd.DataFrame, out_xlsx: Path, heatmap_png: Path) -> None:
    try:
        import xlsxwriter  # noqa: F401
    except Exception:  # pragma: no cover
        import sys, subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "XlsxWriter", "-q"])

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        occ_tab.to_excel(writer, sheet_name="Optimization occupancy", index=False)
        overflows_tab.to_excel(writer, sheet_name="Overflows", index=False)
        transfers_tab.to_excel(writer, sheet_name="Transfers", index=False)

        ws_hm = writer.book.add_worksheet("Heatmap")
        writer.sheets["Heatmap"] = ws_hm
        day_str = str(occ_tab["date"].iloc[0])
        ws_hm.write(0, 0, f"Optimized occupancy heatmap — {day_str}")
        ws_hm.write(1, 0, "Cell label shows: % and occ/beds (+overflow if any)")
        if heatmap_png.exists():
            ws_hm.insert_image(3, 0, str(heatmap_png))

        ws = writer.sheets["Optimization occupancy"]
        cols = list(occ_tab.columns)
        fmt_nowrap = writer.book.add_format({"text_wrap": False})
        for name, width in [("destination unit", 34), ("type of transfer", 18)]:
            if name in cols:
                i = cols.index(name)
                ws.set_column(i, i, width, fmt_nowrap)


# ------------------------- Public API (pipeline) -------------------------
def build_and_solve(
    bedroster_csv: Path,
    forecast_csv: Path,
    compat_csv: Path,
    out_csv: Path,
    solver=None,
    transfer_cost: float = 1.0,
    overflow_cost: float = 1000.0,
    near95_cost: float = 0.1,
    solve_date: Optional[str] = None,
    write_alex_excel: bool = True,
    outdir_alex: Optional[Path] = None,
):
    """Pipeline entrypoint (keeps Luis signature, but matches Alex logic and artifacts)."""

    inputs = build_inputs(Path(bedroster_csv), Path(forecast_csv), Path(compat_csv), solve_date=solve_date)

    model, Occ, In85, Ov85, StayOverflow, Trans, Near95 = _solve_ilp(
        inputs,
        transfer_cost=transfer_cost,
        overflow_cost=overflow_cost,
        near95_cost=near95_cost,
        solver=solver,
    )

    sol_df, transfers_tab, overflows_tab, occ_tab = _extract_outputs(inputs, Occ, In85, Ov85, StayOverflow, Trans, Near95)

    # Luis pipeline CSV
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sol_df.to_csv(out_csv, index=False)

    # A simple transfers CSV for Luis (origin,destination,transfers) derived from Alex transfers_tab
    if transfers_tab.empty:
        transfers_simple = pd.DataFrame(columns=["origin", "destination", "transfers"])
    else:
        transfers_simple = (
            transfers_tab.groupby(["origin unit", "destination unit"], sort=False)["patient (number)"]
            .sum()
            .reset_index()
            .rename(columns={"origin unit": "origin", "destination unit": "destination", "patient (number)": "transfers"})
        )

    transfers_path = out_csv.parent / "ilp_transfers.csv"
    transfers_simple.to_csv(transfers_path, index=False)

    # Alex artifacts
    if write_alex_excel:
        base_outdir = Path(outdir_alex) if outdir_alex is not None else out_csv.parent.parent / "report"
        base_outdir.mkdir(parents=True, exist_ok=True)

        heatmap_png = base_outdir / "Heatmap_OptimizedOcc.png"
        _write_heatmap_png(occ_tab, heatmap_png)

        out_xlsx = base_outdir / "optimized_plan.xlsx"
        _write_alex_excel(occ_tab, overflows_tab, transfers_tab, out_xlsx, heatmap_png)

    meta = {
        "census_date": pd.to_datetime(inputs.census_date),
        "solve_date": pd.to_datetime(inputs.solve_date),
        "beds_dict": inputs.beds_dict,
        "C0_dict": inputs.C0_dict,
        "A_dict": inputs.A_dict,
        "D_dict": inputs.D_dict,
        "allowed_arcs": inputs.allowed_arcs,
        "transfers_csv": str(transfers_path),
    }

    return model, sol_df, transfers_simple, meta


__all__ = ["infer_solve_date_from_bedroster", "build_inputs", "build_and_solve"]

