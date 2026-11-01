@echo off
setlocal EnableExtensions
title Infertility Prediction System

REM Run from the folder this file lives in (the project root).
cd /d "%~dp0"

set "URL=http://127.0.0.1:8000"
set "APP=backend\app.py"

if not exist "%APP%" (
  echo ERROR: %APP% was not found.
  echo Keep this file in the project root, next to the backend folder.
  pause
  exit /b 1
)

REM ---- If the server is already running, just open the browser -----
curl -fsS "%URL%/api/health" >nul 2>&1
if not errorlevel 1 (
  echo Server is already running. Opening the web app...
  start "" "%URL%"
  exit /b 0
)

REM ---- Find Python --------------------------------------------------
set "PYEXE="
where py >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3"
if not defined PYEXE (
  where python >nul 2>&1
  if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
  echo ERROR: Python 3 was not found.
  echo Install it from https://www.python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)

REM ---- Install packages only if something is missing ----------------
%PYEXE% -c "import flask, flask_cors, numpy, pandas, joblib, sklearn, shap" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages. This can take a few minutes...
  %PYEXE% -m pip install -r "backend\requirements.txt"
  if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed. Check your internet connection
    echo and the message above. Python 3.11 or newer is required.
    pause
    exit /b 1
  )
)

REM ---- Start the server in its own window ---------------------------
start "Infertility Prediction API" cmd /k %PYEXE% "%APP%"

echo Starting the prediction server. Waiting up to 60 seconds...
for /L %%i in (1,1,30) do (
  timeout /t 2 /nobreak >nul
  curl -fsS "%URL%/api/health" >nul 2>&1
  if not errorlevel 1 goto ready
)

echo.
echo ERROR: The server did not respond at %URL%.
echo Check the "Infertility Prediction API" window for the Python error.
pause
exit /b 1

:ready
echo Server is ready. Opening the web app...
start "" "%URL%"
echo.
echo Keep the "Infertility Prediction API" window open while using the app.
exit /b 0
