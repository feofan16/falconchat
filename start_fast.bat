@echo off
chcp 65001 >nul
setlocal

REM Переходим в папку батника (и app.py)
cd /d "%~dp0"

REM Активируем виртуальное окружение
call ".\venv\Scripts\activate.bat" || (
  echo [ERROR] Не удалось активировать venv
  exit /b 1
)

REM Запускаем приложение
python ".\app.py" %*

endlocal
