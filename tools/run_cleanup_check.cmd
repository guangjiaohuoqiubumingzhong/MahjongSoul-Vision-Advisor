@echo off
rem Red-five colour, table-art rejection, merged-tile split: tests and model checks.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=reports\vision\cleanup-run.log
echo started %DATE% %TIME% > "%LOG%"
".venv\Scripts\python.exe" -m pytest tests -q -p no:cacheprovider >> "%LOG%" 2>&1
echo pytest exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\benchmark_face_rectification.py --output reports\vision\face-rectification-benchmark-cleanup.json >> "%LOG%" 2>&1
echo benchmark exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\evidence_tile_posed_corpus.py --output reports\vision\top-face-corpus-evidence-cleanup.json >> "%LOG%" 2>&1
echo corpus exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\analyze_unknown_tiles.py --stride 3 --output reports\vision\unknown-tiles-1080p-cleanup.json --atlas reports\vision\unknown-tiles-1080p-cleanup.png --overlays reports\vision\unknown-tiles-1080p-cleanup-frames >> "%LOG%" 2>&1
echo unknown exit %ERRORLEVEL% >> "%LOG%"
echo ALL DONE >> "%LOG%"
