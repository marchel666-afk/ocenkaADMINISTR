@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Оценка звонков

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

if not exist ".venv\Scripts\python.exe" (
  echo Первый запуск: создаю окружение Python...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo.
    echo Не найден Python. Установите Python 3.12 с https://www.python.org/downloads/
    echo и при установке отметьте галочку "Add python.exe to PATH".
    pause
    exit /b 1
  )
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Создан файл настроек .env — впишите в него ключ OPENROUTER_API_KEY.
)

echo Проверяю зависимости...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo Не удалось установить зависимости. Проверьте подключение к интернету.
  pause
  exit /b 1
)

set "OPEN_BROWSER=1"
".venv\Scripts\python.exe" -m app
pause
