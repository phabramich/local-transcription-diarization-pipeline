@echo off
rem Production stack: inference engine + orchestrator service
start "audiocpp" "%~dp0bin\audio.cpp-vulkan\audiocpp_server.exe" --config "%~dp0server.json" --no-ui
timeout /t 3 /nobreak >nul
"%~dp0.venv\Scripts\python.exe" "%~dp0service.py"
