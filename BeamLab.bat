@echo off
cd /d "%~dp0"
python -m beamlab
if errorlevel 1 pause
