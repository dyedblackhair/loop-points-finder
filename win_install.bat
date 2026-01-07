@echo off
echo ============================================
echo  Auto Loop Finder - установка для Windows
echo ============================================
echo.

echo [1/4] Проверяем Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден!
    echo.
    echo Пожалуйста, установите Python 3.8+ с:
    echo https://www.python.org/downloads/
    echo.
    echo Не забудьте отметить галочки:
    echo   [✓] Add Python to PATH
    echo   [✓] Install pip
    echo.
    pause
    exit /b 1
)

echo ✅ Python установлен
python --version
echo.

echo [2/4] Проверяем FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ⚠ FFmpeg не найден в PATH
    echo.
    echo Скачайте FFmpeg с официального сайта:
    echo https://www.gyan.dev/ffmpeg/builds/
    echo.
    echo 1. Скачайте ffmpeg-release-essentials.zip
    echo 2. Распакуйте в C:\ffmpeg
    echo 3. Добавьте C:\ffmpeg\bin в системный PATH
    echo.
    echo Быстрый способ (запустите от администратора):
    echo setx /M PATH "%%PATH%%;C:\ffmpeg\bin"
    echo.
    echo После установки FFmpeg перезапустите терминал
    echo.
    pause
    exit /b 1
)

echo ✅ FFmpeg установлен
ffmpeg -version | findstr "version"
echo.

echo [3/4] Устанавливаем Python зависимости...
pip install opencv-python numpy
if errorlevel 1 (
    echo ❌ Ошибка установки зависимостей
    echo Попробуйте: pip install --user opencv-python numpy
    pause
    exit /b 1
)

echo ✅ Зависимости установлены
echo.

echo [4/4] Проверяем установку...
python test_install.py
if errorlevel 1 (
    echo ❌ Проверка не пройдена
    pause
    exit /b 1
)

echo.
echo ============================================
echo ✅ Установка успешно завершена!
echo ============================================
echo.
echo Использование:
echo   python auto_loop_finder.py video.mp4
echo   python auto_loop_finder.py video.mp4 --preview
echo.
echo Для быстрого запуска используйте quick_start.bat
echo.
pause