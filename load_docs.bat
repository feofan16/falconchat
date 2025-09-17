:: ===== load_docs.bat - Загрузка документов =====
@echo off
chcp 65001 >nul
echo ===============================================
echo   Загрузка документов в базу знаний
echo ===============================================
echo.

call venv\Scripts\activate.bat

if not exist "documents" (
    echo [ОШИБКА] Папка documents не найдена!
    echo Создайте папку и поместите туда ваши документы
    pause
    exit /b 1
)

echo Запуск индексации документов...
python ingest_windows.py

pause