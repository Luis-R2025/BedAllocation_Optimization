# Developer environment for this repository

Use conda to create and activate the environment, then install pip requirements.

Create and activate (Windows PowerShell):

```powershell
conda env create -f environment.yml
conda activate optimisation-transfers-env
```

Or using `requirements.txt` with an existing environment:

```powershell
python -m pip install -r requirements.txt
```

Run the data pipeline:

```powershell
python -m src.pipeline.run_pipeline
```

Notes:
- `pyodbc` requires the Microsoft ODBC Driver for SQL Server installed on the machine.
- `pulp` is installed via `pip` in this environment file.
