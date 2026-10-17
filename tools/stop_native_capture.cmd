@echo off
cd /d "%~dp0.."
for /d %%D in (data\screenshots\native-series\*) do echo stop> "%%D\stop.request"
