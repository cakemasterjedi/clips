@echo off
rem Double-click to update clips on Windows.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %*
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" echo. & echo Update failed, see the messages above.
pause
exit /b %CODE%
