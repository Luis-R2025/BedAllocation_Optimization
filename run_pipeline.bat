@echo off
chcp 65001 > nul
set PYTHON=C:\Users\XJQ490\AppData\Local\anaconda3\envs\optimisation-transfers-env\python.exe
set SCRIPT=\\FNBFS01.RHA-RRS.CA\documents\VitaliteNB\Aide à la décision\IntelligencePredictive_IA\3_Optimisation Transfers\src\pipeline\run_pipeline.py
set LOG=\\FNBFS01.RHA-RRS.CA\documents\VitaliteNB\Aide à la décision\IntelligencePredictive_IA\3_Optimisation Transfers\run_pipeline.log

echo [%DATE% %TIME%] Starting pipeline >> "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] Done (exit code %ERRORLEVEL%) >> "%LOG%"
