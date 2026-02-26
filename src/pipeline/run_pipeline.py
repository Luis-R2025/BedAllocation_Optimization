

from pathlib import Path
import os
import sys
import datetime


# Ensure project root is importable (parent of src) and is the working directory
proj_root = Path(__file__).resolve().parents[2]
if str(proj_root) not in sys.path:
	sys.path.insert(0, str(proj_root))
os.chdir(proj_root)

from src.data import get_data
from src.data import prepare_data
from src.data.get_data import fetch_forecast
from src.model import ilp
import pandas as pd


def main():

	# 1.  Read Bedroster data for the last week (adjust as needed)
	# -------------------------------------------------------------------
	yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
	

	df = get_data.fetch_bedroster(conn=None, from_date=yesterday, to_date=yesterday)
	# Keep only the Units of interest
	UNITS = {
    '3F',
    'SINTER',
    'SCHIR',
    '4C',
    '3B',
    '4D',
    'SCOR',
    '3DPEDOPSY',
    '3FSP',
    '4E',
    '3D',
    '4F',
    '4A',
    '4B',
}


	df_filtered = df[df['Unit'].isin(UNITS)].copy()

	outdir = proj_root / 'data' / 'raw'
	outdir.mkdir(parents=True, exist_ok=True)
	out_path = outdir / 'df_bedroster.csv'
	df_filtered.to_csv(out_path, index=False)
	print(f'Wrote {len(df_filtered)} rows -> {out_path}')

	# 2. Prepare data: normalize column names and fill missing admissions/discharges
	# -------------------------------------------------------------------
	try:
		df_prepared = prepare_data.prepare_df(df_filtered)
		proc_dir = proj_root / 'data' / 'processed'
		proc_dir.mkdir(parents=True, exist_ok=True)
		proc_path = proc_dir / 'bedroster.csv'
		df_prepared.to_csv(proc_path, index=False)
		print(f'Wrote prepared bedroster -> {proc_path} (rows: {len(df_prepared)})')
	except Exception as e:
		print(f'Error preparing data: {e}')

	# 3. Load compatibility matrix from project data (prefer local files; fall back to legacy absolute path)
	# -------------------------------------------------------------------
	compat_root_xlsx = proj_root / 'data' / 'df_compatibility_matrix.xlsx'
	compat_xlsx = proj_root / 'data' / 'raw' / 'df_compatibility_matrix.xlsx'
	compat_csv = proj_root / 'data' / 'raw' / 'df_compatibility_matrix.csv'
	legacy_path = Path(r"Q:\VitaliteNB\Aide à la décision\IntelligencePredictive_IA\3_Optimisation Occupancy\data\df_compatibility_matrix.xlsx")
	try:
		if compat_root_xlsx.exists():
			df_compat = get_data.read_compatibility_matrix(path=str(compat_root_xlsx))
		elif compat_xlsx.exists():
			df_compat = get_data.read_compatibility_matrix(path=str(compat_xlsx))
		elif compat_csv.exists():
			df_compat = pd.read_csv(compat_csv)
		elif legacy_path.exists():
			df_compat = get_data.read_compatibility_matrix(path=str(legacy_path))
		else:
			raise FileNotFoundError('Compatibility matrix not found in project data/ , data/raw/ or legacy path')

		outdir = proj_root / 'data' / 'raw'
		outdir.mkdir(parents=True, exist_ok=True)
		out_path = outdir / 'df_compatibility_matrix.csv'
		df_compat.to_csv(out_path, index=False)
		print(f'Wrote {len(df_compat)} rows -> {out_path}')
	except Exception as e:
		print(f'Compatibility matrix not found: {e}')

	# 4. Prepare compatibility matrix (lowercase headers, coerce/fill values)
	# -------------------------------------------------------------------
	try:
		df_compat_prepared = prepare_data.prepare_df(df_compat)
		proc_dir = proj_root / 'data' / 'processed'
		proc_dir.mkdir(parents=True, exist_ok=True)
		proc_path = proc_dir / 'compatibility_matrix.csv'
		df_compat_prepared.to_csv(proc_path, index=False)
		print(f'Wrote prepared compatibility matrix -> {proc_path} (rows: {len(df_compat_prepared)})')

	except FileNotFoundError as e:
		print(f'Error reading compatibility matrix: {e}')

	# 5. Fetch forecasted averages for last "x" months
	# -------------------------------------------------------------------
	try:
		df_fore = fetch_forecast(conn=None, months=12)   ## last 12 months
		proc_dir = proj_root / 'data' / 'raw'
		proc_dir.mkdir(parents=True, exist_ok=True)
		proc_path = proc_dir / 'df_location_forecast.csv'
		df_fore.to_csv(proc_path, index=False)
		print(f'Wrote location averages  -> {proc_path} (rows: {len(df_fore)})')
	except Exception as e:
		print(f'Error fetching location averages: {e}')

	
	# 6. Fetch location description table and save
	# -------------------------------------------------------------------
	try:
		df_location_description = get_data.fetch_location_description(conn=None)
		outdir = proj_root / 'data' / 'raw'
		outdir.mkdir(parents=True, exist_ok=True)
		out_path = outdir / 'df_location_description.csv'
		df_location_description.to_csv(out_path, index=False)
		print(f'Wrote location description -> {out_path} (rows: {len(df_location_description)})')
	except Exception as e:
		print(f'Error fetching  location description table: {e}')

	# 7. Map forecast locations to descriptions and filter to UNITS
	# -------------------------------------------------------------------
	try:
		# ensure both dfs exist
		if 'df_fore' in locals() and 'df_location_description' in locals():
			df_fore_processed = prepare_data.prepare_location_forecast(
				df_forecast=df_fore,
				df_location_description=df_location_description,
				location_col='LOCATION',
				mnemonic_col='Mnemonic',
				name_col='Name',
				allowed_locations=UNITS,
				map_to_name=False,
			)
			proc_dir = proj_root / 'data' / 'processed'
			proc_dir.mkdir(parents=True, exist_ok=True)
			proc_path = proc_dir / 'location_forecast.csv'
			df_fore_processed.to_csv(proc_path, index=False)
			print(f'Wrote processed location forecast -> {proc_path} (rows: {len(df_fore_processed)})')
			# Keep raw forecast untouched (mnemonics). Processed output saved above.
		else:
			print('Skipping forecast mapping: required dataframes not available')
	except Exception as e:
		print(f'Error preparing location forecast: {e}')

	# 8. Apply generic prepare_df to the processed location forecast
	# -------------------------------------------------------------------
	try:
		proc_dir = proj_root / 'data' / 'processed'
		in_path = proc_dir / 'location_forecast.csv'
		if in_path.exists():
			df_loc = pd.read_csv(in_path)
			df_loc_prepared = prepare_data.prepare_df(df_loc)
			out_path = proc_dir / 'location_forecast.csv'
			df_loc_prepared.to_csv(out_path, index=False)
			print(f'Wrote prepared location forecast -> {out_path} (rows: {len(df_loc_prepared)})')
		else:
			print(f'Skipping step 8: {in_path} not found')
	except Exception as e:
		print(f'Error applying prepare_df to location forecast: {e}')

	# 9. Run IPL model using processed CSVs
	# -------------------------------------------------------------------
	try:
		occ_p = proj_root / 'data' / 'processed' / 'bedroster.csv'
		fore_p = proj_root / 'data' / 'processed' / 'location_forecast.csv'
		compat_p = proj_root / 'data' / 'processed' / 'compatibility_matrix.csv'
		out_p = proj_root / 'outputs' / 'inference' / 'ilp_solution.csv'

		prob, sol_df, transfers_df, meta = ilp.build_and_solve(occ_p, fore_p, compat_p, out_p)
		print(f'Wrote ILP solution -> {out_p} (rows: {len(sol_df)})')
	except Exception as e:
		print(f'Error running ILP model: {e}')

	# 10. Generate output report
	# -------------------------------------------------------------------
	try:
		from src.report.optimizedplan import generate_report
		report_path = generate_report(project_root=proj_root)
		print(f'Wrote report -> {report_path}')
	except Exception as e:
		print(f'Error generating report: {e}')

	# 11. Generate/update metrics history CSV
	# -------------------------------------------------------------------
	try:
		from src.report.metrics_history import generate_metrics
		metrics_path = generate_metrics(project_root=proj_root)
		print(f'Wrote metrics history -> {metrics_path}')
	except Exception as e:
		print(f'Error generating metrics history: {e}')


if __name__ == '__main__':
	main()

