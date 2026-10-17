@echo off
rem Reaction source, call banners, dense-face detection, arrow-covered tiles: tests and sweeps over saved frames.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=reports\vision\reaction-run.log
echo started %DATE% %TIME% > "%LOG%"
".venv\Scripts\python.exe" -m pytest tests -q -p no:cacheprovider >> "%LOG%" 2>&1
echo pytest exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\check_reaction_evidence.py >> "%LOG%" 2>&1
echo sweep exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\analyze_unknown_tiles.py --stride 3 --output reports\vision\unknown-tiles-1080p-draft15.json --atlas reports\vision\unknown-tiles-1080p-draft15.png --overlays reports\vision\unknown-tiles-1080p-draft15-frames >> "%LOG%" 2>&1
echo unknown exit %ERRORLEVEL% >> "%LOG%"
echo ALL DONE >> "%LOG%"
