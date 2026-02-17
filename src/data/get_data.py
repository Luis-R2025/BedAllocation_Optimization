"""Lightweight helpers for fetching bedroster data from the RDS_Provincial view.

This module provides a simple function `fetch_bedroster` that returns a
pandas.DataFrame for a given date range. By default no date filter is applied
("all rows in the table").

"""

from typing import Optional, Sequence
import os
import pyodbc
import pandas as pd


DEFAULT_CONN_STR = (
    "Driver={SQL Server};"
    "Server=FNBDBSQLPR15.rha-rrs.ca;"
    "Database=RDS_Provincial;"
    "Trusted_Connection=yes;"
)


def get_db_connection(conn_str: Optional[str] = None):
    """Return a pyodbc connection using the given connection string or the default."""
    return pyodbc.connect(conn_str or DEFAULT_CONN_STR)

# 1. Get bedroster data 
#--------------------------------------
def fetch_bedroster(
    conn: Optional[pyodbc.Connection] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch bedroster rows.

    Parameters
    - conn: optional pyodbc connection. If None, a connection will be created.
    - from_date, to_date: strings 'YYYY-MM-DD' to filter DATE FROM/TO. If both
      are None then no date filter is applied (return all rows).

    Returns a pandas.DataFrame with columns: Date, facility, Unit, Beds, Patient_days,
    Admissions, Discharges.
    """

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    where_clauses = []
    params = []

    if from_date is not None and to_date is not None:
        # DATEFROMPARTS constructed from LOCDATE is used (same as original notebook)
        where_clauses.append(
            "DATEFROMPARTS(SUBSTRING(CAST(LOCDATE AS VARCHAR(8)),1,4),"
            "SUBSTRING(CAST(LOCDATE AS VARCHAR(8)),5,2),"
            "SUBSTRING(CAST(LOCDATE AS VARCHAR(8)),7,2)) BETWEEN ? AND ?"
        )
        params.extend([from_date, to_date])

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    query = f"""
SELECT
    CONVERT(VARCHAR(10),
        DATEFROMPARTS(
            SUBSTRING(CAST(LOCDATE AS VARCHAR(8)), 1, 4),
            SUBSTRING(CAST(LOCDATE AS VARCHAR(8)), 5, 2),
            SUBSTRING(CAST(LOCDATE AS VARCHAR(8)), 7, 2)
        ),
        120
    ) AS [Date],
    [LOCFACILITY] AS facility,
    [NAME] AS Unit,
    [LOCBEDS] AS Beds,
    [LOCPATIENTDAYS] AS Patient_days,
    [LOCADMISSIONS] AS Admissions,
    [LOCDISCHARGES] AS Discharges
FROM RDS_Provincial.dbo.VIT_ADT_AdmissionStats_V
{where_sql}
"""

    df = pd.read_sql(query, conn, params=params)

    if close_conn:
        conn.close()

    return df


# 2. Get compatibility matrix from Excel file
#----------------------------------------------
def read_compatibility_matrix(path: Optional[str] = None, sheet_name: Optional[object] = 0) -> pd.DataFrame:
    """Read the df_compatibility_matrix.xlsx file and return a DataFrame.

    Parameters
    - path: explicit path to the Excel file. If omitted the function will try
      several sensible default locations inside the repository.
    - sheet_name: passed to `pandas.read_excel`; defaults to the first sheet.

    Returns a pandas.DataFrame.
    """

    if not path:
        raise ValueError("path parameter is required")
    candidates = [path]

    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            result = pd.read_excel(p, sheet_name=sheet_name)
            # pandas.read_excel returns a dict when sheet_name=None (all sheets).
            # Ensure we always return a single DataFrame (first sheet) to callers.
            if isinstance(result, dict):
                return next(iter(result.values()))
            return result

    raise FileNotFoundError(f"Could not find df_compatibility_matrix.xlsx. Tried: {candidates}")


__all__ = ["get_db_connection", "fetch_bedroster", "read_compatibility_matrix"]


# 3. Get forecast or expected admission, discharges, LOS average based on the las x months.
#---------------------------------------------------------------------- ----------------
def fetch_forecast(conn: Optional[pyodbc.Connection] = None, months: int = 3) -> pd.DataFrame:
        """Return per-location averages for admissions, discharges and LOS over the past `months` months.

        Returns: DataFrame with columns: LOCATION, admission_avg, discharge_avg, los_avg
        """
        close_conn = False
        if conn is None:
                conn = get_db_connection()
                close_conn = True

        query = rf"""
SELECT
    [LOCATION],
    AVG(TRY_CONVERT(float, [LOCADMISSIONS]))    AS admission_avg,
    AVG(TRY_CONVERT(float, [LOCDISCHARGES]))    AS discharge_avg,
    AVG(TRY_CONVERT(float, [LOCLENGTHOFSTAY]))  AS los_avg
FROM [RDS_Provincial].[dbo].[VIT_ADT_AdmissionStats_V]
WHERE
    [LOCATION] <> '(none)'
    AND TRY_CONVERT(date, [LOCDATE]) >= DATEADD(month, DATEDIFF(month, 0, GETDATE())-{months}, 0)
    AND TRY_CONVERT(date, [LOCDATE]) <  DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0)
GROUP BY [LOCATION];
"""

        df = pd.read_sql(query, conn)

        if close_conn:
                conn.close()

        return df


# append new helpers to __all__
__all__ += ["fetch_forecast", "fetch_location_description"]


# 4. Get LOCATION description data from the MIS.rep.R0043_Location table
# -------------------------------------------------------------------
def fetch_location_description(conn: Optional[pyodbc.Connection] = None) -> pd.DataFrame:
    """Read the `[MIS].[rep].[R0043_Location]` table (location descriptions).

    If `conn` is omitted this uses a Trusted Connection to
    `S99IWVSQMDBP04.rha-rrs.ca` (Database=`MIS`).
    """
    close_conn = False
    if conn is None:
        conn = pyodbc.connect(
            "Driver={SQL Server};Server=S99IWVSQMDBP04.rha-rrs.ca;Database=MIS;Trusted_Connection=yes;"
        )
        close_conn = True

    query = "SELECT * FROM [MIS].[rep].[R0043_Location]"
    df = pd.read_sql(query, conn)

    if close_conn:
        conn.close()

    return df

