
"""Data preparation utilities for occupancy data.

Provides a small helper to normalize column names to lowercase and to fill
missing values in admissions/discharge columns with zeros.
"""

from typing import MutableMapping
import pandas as pd


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
	"""Simple preparation: lowercase column names and fill nulls with 0.

	This function intentionally keeps behavior minimal: it lowercases all
	column names and replaces any missing values with integer 0 (using
	pandas' fillna). Callers can further coerce dtypes if necessary.
	"""
	out = df.copy()
	out.columns = [str(c).lower() for c in out.columns]
	out = out.fillna(0)
	return out


__all__ = ["prepare_df"]


def prepare_location_forecast(
	df_forecast: pd.DataFrame,
	df_location_description: pd.DataFrame,
	location_col: str = "LOCATION",
	mnemonic_col: str = "Mnemonic",
	name_col: str = "Name",
	allowed_locations: set | None = None,
) -> pd.DataFrame:
	"""Map `df_forecast[location_col]` values from mnemonics to descriptive names.

	- `df_location_description` is expected to contain columns `mnemonic_col`
	  (matching values in `df_forecast[location_col]`) and `name_col` (the
	  human-readable description).
	- If `allowed_locations` is provided, the returned DataFrame will be
	  filtered to only keep rows whose mapped location value is in that set.

	Returns a new DataFrame with the same columns as `df_forecast` but with
	the `location_col` values replaced by the description (or left unchanged
	when no mapping exists). Filtering (if any) is applied after mapping.
	"""
	if df_forecast is None or df_location_description is None:
		raise ValueError("Both df_forecast and df_location_description are required")

	out = df_forecast.copy()

	# Build mapping from mnemonic -> name
	if mnemonic_col not in df_location_description.columns or name_col not in df_location_description.columns:
		# be permissive: try case-insensitive lookup
		cols = {c.lower(): c for c in df_location_description.columns}
		mn_col = cols.get(mnemonic_col.lower())
		nm_col = cols.get(name_col.lower())
		if mn_col and nm_col:
			mapping = dict(zip(df_location_description[mn_col], df_location_description[nm_col]))
		else:
			raise ValueError(f"df_location_description must contain columns '{mnemonic_col}' and '{name_col}'")
	else:
		mapping = dict(zip(df_location_description[mnemonic_col], df_location_description[name_col]))

	# Replace mnemonics with descriptions where possible
	out[location_col] = out[location_col].map(lambda v: mapping.get(v, v))

	# Filter to allowed locations if provided
	if allowed_locations is not None:
		out = out[out[location_col].isin(allowed_locations)].copy()

	return out

__all__ += ["prepare_location_forecast"]

