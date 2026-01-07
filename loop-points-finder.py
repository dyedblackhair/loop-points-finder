#!/usr/bin/env python3
"""
Auto Loop Finder - утилита для автоматического поиска точек зацикливания видео
Требует: Python 3.6+, FFmpeg, OpenCV (opencv-python)
Установка: pip install opencv-python numpy
"""

import subprocess
import json
import numpy as np
import argparse
import sys
import tempfile
import os
from pathlib import Path
import cv2

class AutoLoopFinder:
    def __init__(self, video_path, ffmpeg_path="ffmpeg"):
        """
        Инициализация анализатора
        
        Args:
            video_path: путь к видеофайлу
            ffmpeg_path: путь к FFmpeg (по умолчанию "ffmpeg")
        """
        self.video_path = video_path
        self.ffmpeg_path = ffmpeg_path
        self.video_info = None
        self.frames = None
        
    def get_video_info(self):
        """Получить информацию о видеофайле"""
        print("Получение информации о видео...")
        
        try:
            # Пробуем получить информацию через ffprobe
            probe_cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                self.video_path
            ]
            
            result = subprocess.run(
                probe_cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            info = json.loads(result.stdout)
            
            # Находим видео поток
            video_stream = next(
                (s for s in info['streams'] if s['codec_type'] == 'video'), 
                None
            )
            
            if not video_stream:
                raise ValueError("Видео поток не найден")
            
            self.video_info = {
                'duration': float(info['format']['duration']),
                'fps': eval(video_stream['r_frame_rate']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'frames': int(video_stream.get('nb_frames', 0)) or 
                         int(float(info['format']['duration']) * eval(video_stream['r_frame_rate'])),
                'codec': video_stream['codec_name']
            }
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ffprobe не найден, используем ffmpeg для получения информации...")
            # Если ffprobe не найден, используем ffmpeg
            cmd = [
                self.ffmpeg_path, '-i', self.video_path
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False  # ffmpeg всегда возвращает ошибку при использовании -i без выходного файла
                )
                
                # Парсим вывод ffmpeg
                output = result.stderr
                
                # Извлекаем информацию
                duration = 0
                fps = 30.0  # значение по умолчанию
                width = 1920
                height = 1080
                
                # Ищем длительность
                for line in output.split('\n'):
                    if 'Duration:' in line:
                        try:
                            # Формат: Duration: 00:00:10.00
                            time_str = line.split('Duration:')[1].split(',')[0].strip()
                            h, m, s = time_str.split(':')
                            duration = int(h) * 3600 + int(m) * 60 + float(s)
                        except:
                            pass
                    
                    # Ищем FPS
                    if 'fps' in line.lower() and 'Stream' in line:
                        try:
                            parts = line.split(',')
                            for part in parts:
                                if 'fps' in part:
                                    fps_str = part.strip().split(' ')[0]
                                    fps = float(fps_str)
                                    break
                        except:
                            pass
                    
                    # Ищем разрешение
                    if 'Stream' in line and 'Video:' in line:
                        try:
                            # Ищем паттерн 1920x1080
                            if 'x' in line:
                                for part in line.split(','):
                                    if 'x' in part and part.strip().split('x')[0].isdigit():
                                        res = part.strip().split(' ')[0]
                                        width, height = map(int, res.split('x'))
                                        break
                        except:
                            pass
                
                frames = int(duration * fps)
                
                self.video_info = {
                    'duration': duration,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'frames': frames,
                    'codec': 'unknown'
                }
                
            except Exception as e:
                print(f"Ошибка при получении информации о видео: {e}")
                print("Использую значения по умолчанию...")
                # Значения по умолчанию
                self.video_info = {
                    'duration': 10.0,
                    'fps': 30.0,
                    'width': 1920,
                    'height': 1080,
                    'frames': 300,
                    'codec': 'unknown'
                }
        
        # Выводим информацию
        print(f"Информация о видео:")
        print(f"  Длительность: {self.video_info['duration']:.2f} сек")
        print(f"  FPS: {self.video_info['fps']:.2f}")
        print(f"  Разрешение: {self.video_info['width']}x{self.video_info['height']}")
        print(f"  Кадров: {self.video_info['frames']}")
        print(f"  Кодек: {self.video_info['codec']}")
        
        return self.video_info
    
    def extract_frame_signatures(self, sample_rate=0.1, method="histogram"):
        """
        Извлечь сигнатуры кадров для анализа
        
        Args:
            sample_rate: доля кадров для анализа (0.0-1.0)
            method: метод сравнения ("histogram", "edges", "grayscale")
        """
        print(f"\nИзвлечение сигнатур кадров (метод: {method})...")
        
        total_frames = self.video_info['frames']
        sample_count = int(total_frames * sample_rate)
        
        if sample_count < 2:
            sample_count = min(100, total_frames)
        
        # Равномерно распределяем кадры для анализа
        frame_indices = np.linspace(0, total_frames-1, sample_count, dtype=int)
        
        signatures = []
        timestamps = []
        
        print(f"Анализируем {sample_count} кадров из {total_frames}...")
        
        for idx, frame_idx in enumerate(frame_indices):
            timestamp = frame_idx / self.video_info['fps']
            
            # Извлекаем конкретный кадр через ffmpeg
            temp_image = f"temp_frame_{idx}.png"
            cmd = [
                self.ffmpeg_path,
                '-i', self.video_path,
                '-vf', f'select=eq(n\\,{frame_idx})',
                '-vframes', '1',
                '-y',
                temp_image
            ]
            
            try:
                # Запускаем ffmpeg для извлечения кадра
                subprocess.run(cmd, capture_output=True, check=True)
                
                # Читаем изображение через OpenCV
                if os.path.exists(temp_image):
                    frame = cv2.imread(temp_image)
                    
                    if frame is not None:
                        # Вычисляем сигнатуру в зависимости от метода
                        if method == "histogram":
                            # Упрощенная гистограмма в grayscale
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                            signature = hist.flatten()
                            
                        elif method == "grayscale":
                            # Простое преобразование в grayscale и уменьшение
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            small = cv2.resize(gray, (32, 32))
                            signature = small.flatten().astype(float) / 255.0
                            
                        elif method == "edges":
                            # Детектор краев Canny
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 100, 200)
                            small = cv2.resize(edges, (32, 32))
                            signature = small.flatten().astype(float) / 255.0
                        
                        signatures.append(signature)
                        timestamps.append(timestamp)
                    
                    # Удаляем временный файл
                    os.remove(temp_image)
                
                if (idx + 1) % 10 == 0 or (idx + 1) == len(frame_indices):
                    print(f"  Обработано {idx+1}/{len(frame_indices)} кадров")
                
            except Exception as e:
                print(f"  Ошибка при извлечении кадра {frame_idx}: {e}")
                # Удаляем временный файл если он существует
                if os.path.exists(temp_image):
                    os.remove(temp_image)
                continue
        
        if len(signatures) == 0:
            print("Не удалось извлечь ни одного кадра")
            return None, None, None
        
        print(f"Извлечено {len(signatures)} сигнатур")
        return np.array(signatures), np.array(timestamps), frame_indices[:len(signatures)]
    
    def find_best_loop_points(self, signatures, timestamps, frame_indices, 
                             min_loop_duration=0.5, max_loop_duration=10.0):
        """
        Найти лучшие точки для зацикливания
        
        Args:
            signatures: массив сигнатур кадров
            timestamps: временные метки
            frame_indices: индексы кадров
            min_loop_duration: минимальная длительность цикла (сек)
            max_loop_duration: максимальная длительность цикла (сек)
        """
        print("\nПоиск лучших точек для зацикливания...")
        
        if signatures is None or len(signatures) < 2:
            print("Недостаточно данных для анализа")
            return None
        
        n = len(signatures)
        
        # Ищем пары с минимальной разницей
        best_pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Проверяем длительность цикла
                time_diff = timestamps[j] - timestamps[i]
                if time_diff < min_loop_duration or time_diff > max_loop_duration:
                    continue
                
                # Вычисляем разницу между сигнатурами
                diff = np.linalg.norm(signatures[i] - signatures[j])
                
                # Нормализуем по времени (предпочтение более коротким циклам)
                normalized_diff = diff / (time_diff + 0.1)
                
                best_pairs.append({
                    'start_idx': frame_indices[i],
                    'end_idx': frame_indices[j],
                    'start_time': timestamps[i],
                    'end_time': timestamps[j],
                    'duration': time_diff,
                    'raw_diff': diff,
                    'norm_diff': normalized_diff
                })
        
        if not best_pairs:
            print("Не найдено подходящих точек для цикла")
            return None
        
        # Сортируем по нормализованной разнице
        best_pairs.sort(key=lambda x: x['norm_diff'])
        
        # Берем топ-10 результатов
        top_results = best_pairs[:min(10, len(best_pairs))]
        
        # Также ищем по сырой разнице для сравнения
        best_pairs.sort(key=lambda x: x['raw_diff'])
        top_raw = best_pairs[:min(5, len(best_pairs))]
        
        return {
            'top_by_normalized': top_results,
            'top_by_raw': top_raw
        }
    
    def generate_preview(self, start_frame, end_frame, output_path="loop_preview.mp4"):
        """Создать превью найденного цикла"""
        print(f"\nСоздание превью цикла...")
        
        start_time = start_frame / self.video_info['fps']
        end_time = end_frame / self.video_info['fps']
        duration = end_time - start_time
        
        try:
            # Простой способ: вырезаем и повторяем 3 раза
            cmd = [
                self.ffmpeg_path,
                '-i', self.video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-filter_complex', '[0:v]loop=3:250[v];[0:a]aloop=3:1[a]',
                '-map', '[v]',
                '-map', '[a]',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                '-y',
                output_path
            ]
            
            print(f"Создаем превью: {start_time:.2f} - {end_time:.2f} (повтор 3 раза)")
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"Превью сохранено в: {output_path}")
            return True
            
        except subprocess.CalledProcessError:
            # Упрощенный вариант без звука
            try:
                cmd_simple = [
                    self.ffmpeg_path,
                    '-i', self.video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-vf', 'loop=3:250',
                    '-c:v', 'libx264',
                    '-an',
                    '-y',
                    output_path
                ]
                subprocess.run(cmd_simple, capture_output=True, check=True)
                print(f"Превью сохранено (без звука) в: {output_path}")
                return True
            except:
                print("Не удалось создать превью")
                return False
    
    def analyze(self, sample_rate=0.1, method="histogram", 
                min_duration=0.5, max_duration=5.0,
                create_preview=False):
        """
        Основной метод анализа
        
        Args:
            sample_rate: доля анализируемых кадров (0.0-1.0)
            method: метод сравнения ("histogram", "edges", "grayscale")
            min_duration: минимальная длительность цикла
            max_duration: максимальная длительность цикла
            create_preview: создать превью лучшего цикла
        """
        print("=" * 60)
        print("Auto Loop Finder - поиск точек для идеального цикла")
        print("=" * 60)
        
        # Шаг 1: Получить информацию о видео
        self.get_video_info()
        
        # Шаг 2: Извлечь сигнатуры кадров
        signatures, timestamps, frame_indices = self.extract_frame_signatures(
            sample_rate=sample_rate, 
            method=method
        )
        
        if signatures is None:
            print("\nНе удалось извлечь кадры для анализа")
            print("Попробуйте уменьшить sample_rate")
            return
        
        # Шаг 3: Найти лучшие точки
        results = self.find_best_loop_points(
            signatures, timestamps, frame_indices,
            min_loop_duration=min_duration,
            max_loop_duration=max_duration
        )
        
        if not results:
            print("\nНе удалось найти подходящие точки для цикла.")
            print("Попробуйте:")
            print("  1. Увеличить sample_rate (например, до 0.3)")
            print("  2. Попробовать другой метод (edges или grayscale)")
            print("  3. Изменить min_duration/max_duration")
            return
        
        # Шаг 4: Вывести результаты
        print("\n" + "=" * 60)
        print("ТОП-5 ЛУЧШИХ ТОЧЕК ДЛЯ ЗАЦИКЛИВАНИЯ:")
        print("=" * 60)
        
        print("\nЛучшие по качеству (нормализованная разница):")
        for i, result in enumerate(results['top_by_normalized'][:5], 1):
            print(f"\n{i}. Начало: {result['start_time']:.3f} сек (кадр {result['start_idx']})")
            print(f"   Конец:  {result['end_time']:.3f} сек (кадр {result['end_idx']})")
            print(f"   Длительность: {result['duration']:.3f} сек ({int(result['duration'] * self.video_info['fps'])} кадров)")
            print(f"   Качество: {result['norm_diff']:.4f} (меньше = лучше)")
        
        print("\n\nЛучшие по визуальному сходству (сырая разница):")
        for i, result in enumerate(results['top_by_raw'][:5], 1):
            print(f"\n{i}. Начало: {result['start_time']:.3f} сек (кадр {result['start_idx']})")
            print(f"   Конец:  {result['end_time']:.3f} сек (кадр {result['end_idx']})")
            print(f"   Длительность: {result['duration']:.3f} сек")
            print(f"   Качество: {result['raw_diff']:.2f} (меньше = лучше)")
        
        print("\n" + "=" * 60)
        print("КОМАНДЫ ДЛЯ DAVINCI RESOLVE:")
        print("=" * 60)
        
        if results['top_by_normalized']:
            best = results['top_by_normalized'][0]
            print(f"\n1. Установите IN точку на: {best['start_time']:.3f} сек")
            print(f"2. Установите OUT точку на: {best['end_time']:.3f} сек")
            print(f"3. Длительность цикла: {best['duration']:.3f} сек")
            print(f"\nИли используйте номера кадров:")
            print(f"  IN: кадр {best['start_idx']}")
            print(f"  OUT: кадр {best['end_idx'] - 1} (включительно)")
            
            # Шаг 5: Создать превью (если нужно)
            if create_preview:
                preview_file = f"loop_preview_{best['start_idx']}_{best['end_idx']}.mp4"
                self.generate_preview(best['start_idx'], best['end_idx'], preview_file)
                
                print(f"\nПревью создано: {preview_file}")
                print("Совет: Добавьте кроссфейд на 1-3 кадра в DaVinci Resolve для более плавного перехода.")
        
        # Сохранить результаты в файл
        with open("loop_points.txt", "w", encoding='utf-8') as f:
            f.write(f"DaVinci Resolve Loop Points\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"FPS: {self.video_info['fps']:.2f}\n\n")
            
            f.write("TOP 5 POINTS:\n")
            for i, result in enumerate(results['top_by_normalized'][:5], 1):
                f.write(f"\n{i}. Start: {result['start_time']:.3f}s (frame {result['start_idx']})\n")
                f.write(f"   End:   {result['end_time']:.3f}s (frame {result['end_idx']})\n")
                f.write(f"   Duration: {result['duration']:.3f}s\n")
                f.write(f"   Quality: {result['norm_diff']:.4f}\n")
        
        print(f"\nРезультаты также сохранены в: loop_points.txt")
        print("\nГотово! Используйте эти точки в DaVinci Resolve для создания цикла.")


def main():
    parser = argparse.ArgumentParser(
        description='Auto Loop Finder - поиск точек для бесшовного зацикливания видео',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s video.mp4
  %(prog)s video.mp4 --method edges --sample-rate 0.2
  %(prog)s video.mp4 --min-duration 1.0 --max-duration 3.0 --preview
  %(prog)s video.mp4 --ffmpeg "C:\\ffmpeg\\bin\\ffmpeg.exe"
        """
    )
    
    parser.add_argument('video', help='Путь к видеофайлу')
    parser.add_argument('--sample-rate', type=float, default=0.1,
                       help='Доля анализируемых кадров (0.01-0.5, по умолчанию 0.1)')
    parser.add_argument('--method', choices=['histogram', 'edges', 'grayscale'], 
                       default='histogram', help='Метод сравнения кадров')
    parser.add_argument('--min-duration', type=float, default=0.5,
                       help='Минимальная длительность цикла в секундах')
    parser.add_argument('--max-duration', type=float, default=5.0,
                       help='Максимальная длительность цикла в секундах')
    parser.add_argument('--ffmpeg', default='ffmpeg',
                       help='Путь к ffmpeg (по умолчанию "ffmpeg")')
    parser.add_argument('--preview', action='store_true',
                       help='Создать превью лучшего цикла')
    
    args = parser.parse_args()
    
    # Проверяем существование файла
    if not Path(args.video).exists():
        print(f"Ошибка: Файл '{args.video}' не найден")
        sys.exit(1)
    
    # Проверяем FFmpeg
    try:
        result = subprocess.run([args.ffmpeg, '-version'], 
                               capture_output=True, text=True, check=True)
        first_line = result.stdout.split('\n')[0]
        print(f"FFmpeg найден: {first_line}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Ошибка: FFmpeg не найден по пути '{args.ffmpeg}'")
        print("\nПожалуйста, установите FFmpeg:")
        print("1. Скачайте с https://www.gyan.dev/ffmpeg/builds/")
        print("2. Распакуйте в C:\\ffmpeg")
        print("3. Укажите полный путь: --ffmpeg \"C:\\ffmpeg\\bin\\ffmpeg.exe\"")
        print("\nИли добавьте FFmpeg в PATH и перезапустите терминал")
        sys.exit(1)
    
    # Запускаем анализ
    finder = AutoLoopFinder(args.video, args.ffmpeg)
    finder.analyze(
        sample_rate=args.sample_rate,
        method=args.method,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        create_preview=args.preview
    )


if __name__ == "__main__":
    main()