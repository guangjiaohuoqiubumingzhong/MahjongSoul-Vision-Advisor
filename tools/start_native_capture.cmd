@echo off
rem Background lossless capture of the game window for resolution evaluation (read-only).
cd /d "%~dp0.."
start "native capture" /min ".venv\Scripts\python.exe" scripts\capture_native_series.py --seconds 900 --every 3
