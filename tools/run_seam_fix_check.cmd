@echo off
rem White-dragon seam fix: tests, annotated rectification benchmark, 1080p comparison.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=reports\vision\seam-fix-run.log
echo started %DATE% %TIME% > "%LOG%"
".venv\Scripts\python.exe" -m pytest tests -q -p no:cacheprovider >> "%LOG%" 2>&1
echo pytest exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\benchmark_face_rectification.py --output reports\vision\face-rectification-benchmark-seam.json >> "%LOG%" 2>&1
echo benchmark exit %ERRORLEVEL% >> "%LOG%"
".venv\Scripts\python.exe" scripts\compare_native_resolution.py --stride 3 --output reports\vision\native-resolution-comparison-seam.json --atlas reports\vision\native-resolution-differences-seam.png >> "%LOG%" 2>&1
echo compare exit %ERRORLEVEL% >> "%LOG%"
echo ALL DONE >> "%LOG%"
