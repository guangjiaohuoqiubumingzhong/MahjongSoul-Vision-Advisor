@echo off
rem Runs the top-face rectification checks and writes everything to one log.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=reports\vision\face-rectification-run.log
echo started %DATE% %TIME% > "%LOG%"
echo ==== pytest ==== >> "%LOG%"
".venv\Scripts\python.exe" -m pytest tests/test_rectify.py tests/test_pjura.py tests/test_vision.py -q -p no:cacheprovider >> "%LOG%" 2>&1
echo pytest exit %ERRORLEVEL% >> "%LOG%"
echo ==== benchmark_face_rectification ==== >> "%LOG%"
".venv\Scripts\python.exe" scripts\benchmark_face_rectification.py --atlas reports\vision\face-rectification-atlas.png >> "%LOG%" 2>&1
echo benchmark exit %ERRORLEVEL% >> "%LOG%"
echo ==== evidence_tile_posed_corpus ==== >> "%LOG%"
".venv\Scripts\python.exe" scripts\evidence_tile_posed_corpus.py >> "%LOG%" 2>&1
echo evidence exit %ERRORLEVEL% >> "%LOG%"
echo finished %DATE% %TIME% >> "%LOG%"
echo ALL DONE >> "%LOG%"
