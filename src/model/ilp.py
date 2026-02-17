"""Simple IPL integer program for bedroster balancing.

This script builds and solves an ILP that attempts to keep unit census within
bed capacity by allowing transfers between compatible units. It uses the
following inputs (CSV files produced by `run_pipeline.py`):

- data/processed/bedroster.csv        -> contains `Unit`, `Beds`, `Admissions`, etc.
- data/processed/location_forecast.csv -> contains `LOCATION`, `admission_avg`, `discharge_avg`, `los_avg`
- data/processed/compatibility_matrix.csv -> compatibility matrix (wide or long format)

The model variables:
- t[i,j] : integer transfers from unit i to unit j (>=0)
- overflow[i] : non-negative integer overflow at unit i

Constraints:
- final_census_i = init_census_i + admission_avg_i - discharge_avg_i - sum_j t[i,j] + sum_k t[k,i]
- final_census_i <= beds_i + overflow_i

Objective: minimize total overflow (primary) and transfers (secondary).

Requires `pulp` to be installed. If not available the module raises ImportError
when executed.
"""

from pathlib import Path
import pandas as pd
import math

try:
    import pulp
except Exception as e:
    pulp = None


def read_compatibility_matrix(path: Path) -> pd.DataFrame:
    """Return compatibility as a long-form DataFrame with columns ['origin','destination','compatible'].

    Accepts either a matrix format (rows index = origins, columns = destinations)
    or a long form with columns that include 'origin'/'destination' or similar.
    """
    df = pd.read_csv(path)

    # Try to detect long format
    lower = [c.lower() for c in df.columns]
    if 'origin' in lower and 'destination' in lower:
        # normalize
        cols = {k.lower(): v for k, v in zip(lower, df.columns)}
        return df.rename(columns={cols['origin']: 'origin', cols['destination']: 'destination'})[['origin','destination']].assign(compatible=1)

    # Otherwise assume matrix: index in first column, destinations in remaining columns
    if df.shape[1] >= 2:
        origins = df.iloc[:, 0].astype(str).tolist()
        dest_cols = list(df.columns[1:])
        records = []
        for i, orig in enumerate(origins):
            for j, col in enumerate(dest_cols):
                val = df.iloc[i, j+1]
                try:
                    compatible = 1 if float(val) != 0 else 0
                except Exception:
                    compatible = 1 if str(val).strip() not in ('', '0', 'False', 'false') else 0
                records.append({'origin': str(orig), 'destination': str(col), 'compatible': int(compatible)})
        return pd.DataFrame.from_records(records)

    raise ValueError('Could not parse compatibility matrix')


def build_and_solve(
    bedroster_csv: Path,
    forecast_csv: Path,
    compat_csv: Path,
    out_csv: Path,
    solver=None,
    transfer_cost: float = 0.1,
    overflow_cost: float = 100.0,
):
    """Build the ILP and write solution to `out_csv`.

    Returns the pulp problem and solution DataFrame.
    """
    if pulp is None:
        raise ImportError('pulp is required; please install with `pip install pulp`')

    df_occ = pd.read_csv(bedroster_csv)
    df_for = pd.read_csv(forecast_csv)
    df_compat = read_compatibility_matrix(compat_csv)

    # Normalize keys
    df_occ_cols = {c.lower(): c for c in df_occ.columns}
    df_for_cols = {c.lower(): c for c in df_for.columns}

    unit_col = df_occ_cols.get('unit', df_occ.columns[0])
    beds_col = df_occ_cols.get('beds', 'Beds')
    admissions_col = df_occ_cols.get('admissions', 'Admissions')

    loc_col = df_for_cols.get('location', 'LOCATION')
    admit_avg_col = df_for_cols.get('admission_avg', 'admission_avg')
    disch_avg_col = df_for_cols.get('discharge_avg', 'discharge_avg')

    # Aggregate bedroster to unit-level
    occ_summary = df_occ.groupby(unit_col).agg(
        beds=(beds_col, 'max'),
        init_admissions=(admissions_col, 'sum'),
    ).reset_index()

    # Forecast keyed by LOCATION
    for_summary = df_for[[loc_col, admit_avg_col, disch_avg_col]].rename(columns={loc_col: 'unit', admit_avg_col: 'admit_avg', disch_avg_col: 'disch_avg'})

    # Merge
    units = sorted(set(occ_summary['unit'].astype(str)).union(for_summary['unit'].astype(str)))
    occ_summary['unit'] = occ_summary['unit'].astype(str)
    for_summary['unit'] = for_summary['unit'].astype(str)
    merged = pd.DataFrame({'unit': units})
    merged = merged.merge(occ_summary, on='unit', how='left')
    merged = merged.merge(for_summary, on='unit', how='left')

    # Fill missing numeric with zeros
    merged['beds'] = merged['beds'].fillna(0).astype(int)
    merged['init_admissions'] = merged['init_admissions'].fillna(0).astype(int)
    merged['admit_avg'] = merged['admit_avg'].fillna(0.0).astype(float)
    merged['disch_avg'] = merged['disch_avg'].fillna(0.0).astype(float)

    # Compatibility: set of allowed pairs
    compat_df = df_compat[df_compat['compatible'].astype(int) != 0].copy()
    compat_df['origin'] = compat_df['origin'].astype(str)
    compat_df['destination'] = compat_df['destination'].astype(str)

    allowed_pairs = set(zip(compat_df['origin'], compat_df['destination']))

    # Build model
    prob = pulp.LpProblem('ipl_balance', pulp.LpMinimize)

    # Variables
    t = {}
    overflow = {}
    final_census = {}

    for u in units:
        overflow[u] = pulp.LpVariable(f'overflow__{u}', lowBound=0, cat='Integer')
        final_census[u] = pulp.LpVariable(f'final__{u}', lowBound=0, cat='Integer')

    for (i, j) in allowed_pairs:
        # Only create variables for known units
        if i in units and j in units:
            t[(i, j)] = pulp.LpVariable(f'transfer__{i}__{j}', lowBound=0, cat='Integer')

    # Objective: heavy penalty on overflow, small penalty on transfers
    prob += pulp.lpSum([overflow[u] * overflow_cost for u in units]) + pulp.lpSum([tvar * transfer_cost for tvar in t.values()])

    # Constraints
    for u in units:
        init = int(merged.loc[merged['unit'] == u, 'init_admissions'].iloc[0])
        admit = float(merged.loc[merged['unit'] == u, 'admit_avg'].iloc[0])
        disch = float(merged.loc[merged['unit'] == u, 'disch_avg'].iloc[0])
        beds = int(merged.loc[merged['unit'] == u, 'beds'].iloc[0])

        # Sum outgoing and incoming transfers
        outgoing = pulp.lpSum([t[(u, j)] for (i, j) in t.keys() if i == u and (u, j) in t]) if any(i == u for (i, j) in t.keys()) else 0
        incoming = pulp.lpSum([t[(i, u)] for (i, j) in t.keys() if j == u and (i, u) in t]) if any(j == u for (i, j) in t.keys()) else 0

        # final_census = init + admit - disch - outgoing + incoming
        # admit/disch may be fractional; round conservatively by taking ceil of admit and floor of disch
        prob += final_census[u] == init + pulp.lpSum([math.ceil(admit)]) - pulp.lpSum([math.floor(disch)]) - outgoing + incoming

        # capacity constraint
        prob += final_census[u] <= beds + overflow[u]

    # Solve
    solver_to_use = solver or pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver_to_use)

    # Extract solution
    rows = []
    for u in units:
        fin = int(pulp.value(final_census[u])) if pulp.value(final_census[u]) is not None else None
        of = int(pulp.value(overflow[u])) if pulp.value(overflow[u]) is not None else 0
        beds = int(merged.loc[merged['unit'] == u, 'beds'].iloc[0])
        rows.append({'unit': u, 'final_census': fin, 'overflow': of, 'beds': beds})

    sol_df = pd.DataFrame(rows)

    # Add transfers summary
    transfer_rows = []
    for (i, j), var in t.items():
        val = int(pulp.value(var)) if pulp.value(var) is not None else 0
        if val > 0:
            transfer_rows.append({'origin': i, 'destination': j, 'transfers': val})
    transfers_df = pd.DataFrame(transfer_rows)

    # Merge transfers totals per unit
    if not transfers_df.empty:
        out_sum = transfers_df.groupby('origin')['transfers'].sum().rename('transfers_out').reset_index()
        in_sum = transfers_df.groupby('destination')['transfers'].sum().rename('transfers_in').reset_index()
        sol_df = sol_df.merge(out_sum, left_on='unit', right_on='origin', how='left').drop(columns=['origin'])
        sol_df = sol_df.merge(in_sum, left_on='unit', right_on='destination', how='left').drop(columns=['destination'])
    else:
        sol_df['transfers_out'] = 0
        sol_df['transfers_in'] = 0

    sol_df = sol_df.fillna(0)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sol_df.to_csv(out_csv, index=False)

    return prob, sol_df, transfers_df


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    occ = root / 'data' / 'processed' / 'bedroster.csv'
    fore = root / 'data' / 'processed' / 'location_forecast.csv'
    compat = root / 'data' / 'processed' / 'compatibility_matrix.csv'
    out = root / 'outputs' / 'inference' / 'ilp_solution.csv'

    prob, sol_df, transfers_df = build_and_solve(occ, fore, compat, out)
    if pulp is not None:
        print('Objective:', pulp.value(prob.objective))
    print('Solution written to', out)
