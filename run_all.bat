@echo off
rem Runs the daily collection and forecast from this folder.
rem Uses whichever python is on PATH. To use a specific one, set PYTHON first,
rem e.g. set PYTHON=C:\path\to\python.exe

cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
if not exist logs mkdir logs

echo ===== %date% %time% ===== >> logs\daily.log
"%PYTHON%" collect_daily.py >> logs\daily.log 2>&1
"%PYTHON%" run_forecast.py >> logs\daily.log 2>&1
