@echo off
rem Double-click to install clips on Windows (options: install.bat -Vision -Ollama -Yes; see install.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" echo. & echo Install failed, see the messages above.
pause
exit /b %CODE%
