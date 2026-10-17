@echo off
rem Why are tiles unknown at 1080p? Offline analysis of saved native frames.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=reports\vision\unknown-tiles-1080p.log
echo started %DATE% %TIME% > "%LOG%"
".venv\Scripts\python.exe" scripts\analyze_unknown_tiles.py --stride 3 >> "%LOG%" 2>&1
echo analysis exit %ERRORLEVEL% >> "%LOG%"
echo ALL DONE >> "%LOG%"
