@echo off
rem Compares native-resolution crops with 1366x768 crops on the same captured frames.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=reports\vision\native-resolution-run.log
echo started %DATE% %TIME% > "%LOG%"
".venv\Scripts\python.exe" -m pytest tests -q -p no:cacheprovider >> "%LOG%" 2>&1
echo pytest exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\compare_native_resolution.py >> "%LOG%" 2>&1
echo compare exit %ERRORLEVEL% >> "%LOG%"
echo ALL DONE >> "%LOG%"
