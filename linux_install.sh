#!/bin/bash

echo "============================================"
echo " Auto Loop Finder - установка для Linux"
echo "============================================"
echo ""

# Проверяем, что мы на Ubuntu/Debian
if [ -f /etc/debian_version ]; then
    echo "[1/4] Обновляем систему..."
    sudo apt update
    sudo apt upgrade -y
    
    echo "[2/4] Устанавливаем Python и FFmpeg..."
    sudo apt install python3 python3-pip python3-venv ffmpeg -y
    
elif [ -f /etc/redhat-release ] || [ -f /etc/fedora-release ]; then
    echo "[1/4] Обновляем систему..."
    sudo dnf update -y
    
    echo "[2/4] Устанавливаем Python и FFmpeg..."
    sudo dnf install python3 python3-pip ffmpeg -y
    
elif [ -f /etc/arch-release ]; then
    echo "[1/4] Обновляем систему..."
    sudo pacman -Syu --noconfirm
    
    echo "[2/4] Устанавливаем Python и FFmpeg..."
    sudo pacman -S python python-pip ffmpeg --noconfirm
else
    echo "⚠ Неизвестный дистрибутив Linux"
    echo "Убедитесь, что установлены:"
    echo "  - Python 3.6+"
    echo "  - pip3"
    echo "  - FFmpeg"
    echo ""
    read -p "Продолжить? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Системные пакеты установлены"
echo ""

echo "[3/4] Проверяем Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден"
    exit 1
fi

echo "✅ Python установлен"
python3 --version
echo ""

echo "[4/4] Устанавливаем Python зависимости..."
pip3 install opencv-python numpy

if [ $? -ne 0 ]; then
    echo "❌ Ошибка установки зависимостей"
    echo "Попробуйте: pip3 install --user opencv-python numpy"
    exit 1
fi

echo "✅ Зависимости установлены"
echo ""

echo "Проверяем установку..."
python3 test_install.py

echo ""
echo "============================================"
echo "✅ Установка успешно завершена!"
echo "============================================"
echo ""
echo "Использование:"
echo "  python3 auto_loop_finder.py video.mp4"
echo "  python3 auto_loop_finder.py video.mp4 --preview"
echo ""
echo "Для быстрого запуска: chmod +x quick_start.sh"
echo ""