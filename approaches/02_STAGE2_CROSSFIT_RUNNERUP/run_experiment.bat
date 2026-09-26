@echo off
REM 01_STAGE2_EDIT_FEATURES: validation (fold B score) then final test predictions.
REM Reuses the shared baseline cache; writes only inside this folder.
cd /d "%~dp0"
if "%ER_CACHE_DIR%"=="" set ER_CACHE_DIR=%~dp0..\..\cache
set ER_OUTPUT_DIR=%~dp0output
set PY=C:\Users\shrey\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" -u src\stage2.py %1 %2
