@echo off
rem Recapture / occlusion resilience + draft-14 river rects: full tests and 1080p unknown analysis.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=reports\vision\recapture-run.log
echo started %DATE% %TIME% > "%LOG%"
".venv\Scripts\python.exe" -m pytest tests -q -p no:cacheprovider >> "%LOG%" 2>&1
echo pytest exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\analyze_unknown_tiles.py --stride 3 --output reports\vision\unknown-tiles-1080p-draft14.json --atlas reports\vision\unknown-tiles-1080p-draft14.png --overlays reports\vision\unknown-tiles-1080p-draft14-frames >> "%LOG%" 2>&1
echo unknown exit %ERRORLEVEL% >> "%LOG%"
echo ALL DONE >> "%LOG%"
