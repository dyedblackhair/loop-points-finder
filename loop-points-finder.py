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
        cmd = [
            self.ffmpeg_path, '-i', self.video_path,
            '-hide_banner',
            '-f', 'null', '-'
        ]
        
        # Получаем информацию через FFprobe
        probe_cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            self.video_path
        ]
        
        try:
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
                'fps': eval(video_stream['r_frame_rate']),  # e.g., "30000/1001"
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'frames': int(video_stream.get('nb_frames', 0)) or 
                         int(float(info['format']['duration']) * eval(video_stream['r_frame_rate'])),
                'codec': video_stream['codec_name']
            }
            
            print(f"Информация о видео:")
            print(f"  Длительность: {self.video_info['duration']:.2f} сек")
            print(f"  FPS: {self.video_info['fps']:.2f}")
            print(f"  Разрешение: {self.video_info['width']}x{self.video_info['height']}")
            print(f"  Кадров: {self.video_info['frames']}")
            print(f"  Кодек: {self.video_info['codec']}")
            
            return self.video_info
            
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при получении информации о видео: {e}")
            sys.exit(1)
    
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
        
        # Создаем временный файл для списка кадров
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for idx in frame_indices:
                timestamp = idx / self.video_info['fps']
                f.write(f"select=eq(n\\,{idx})\\n")
            f.flush()
            
            # Извлекаем кадры через FFmpeg
            cmd = [
                self.ffmpeg_path,
                '-i', self.video_path,
                '-vf', f"select='{open(f.name).read().replace(chr(10),'\\\\n')}'",
                '-vsync', '0',
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo',
                '-'
            ]
            
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                width = self.video_info['width']
                height = self.video_info['height']
                frame_size = width * height * 3
                
                for idx, frame_idx in enumerate(frame_indices):
                    # Читаем raw video frame
                    raw_frame = process.stdout.read(frame_size)
                    if len(raw_frame) != frame_size:
                        break
                    
                    # Конвертируем в numpy array
                    frame = np.frombuffer(raw_frame, dtype=np.uint8)
                    frame = frame.reshape((height, width, 3))
                    
                    # Вычисляем сигнатуру в зависимости от метода
                    if method == "histogram":
                        # Гистограмма по цветам (упрощенная)
                        hist_r = cv2.calcHist([frame], [0], None, [64], [0, 256])
                        hist_g = cv2.calcHist([frame], [1], None, [64], [0, 256])
                        hist_b = cv2.calcHist([frame], [2], None, [64], [0, 256])
                        signature = np.concatenate([hist_r, hist_g, hist_b]).flatten()
                        
                    elif method == "grayscale":
                        # Простое преобразование в grayscale и уменьшение
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        small = cv2.resize(gray, (32, 32))
                        signature = small.flatten().astype(float) / 255.0
                        
                    elif method == "edges":
                        # Детектор краев Canny
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 100, 200)
                        small = cv2.resize(edges, (32, 32))
                        signature = small.flatten().astype(float) / 255.0
                    
                    signatures.append(signature)
                    timestamps.append(frame_idx / self.video_info['fps'])
                    
                    if idx % 10 == 0:
                        print(f"  Обработано {idx+1}/{len(frame_indices)} кадров", end='\r')
                
                process.wait()
                os.unlink(f.name)
                
            except Exception as e:
                print(f"\nОшибка при извлечении кадров: {e}")
                if os.path.exists(f.name):
                    os.unlink(f.name)
                sys.exit(1)
        
        print(f"\nИзвлечено {len(signatures)} сигнатур")
        return np.array(signatures), np.array(timestamps), frame_indices
    
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
        
        n = len(signatures)
        min_frames = int(min_loop_duration * self.video_info['fps'])
        max_frames = int(max_loop_duration * self.video_info['fps'])
        
        best_pairs = []
        
        for i in range(n):
            max_j = min(i + int(max_loop_duration * self.video_info['fps'] / 
                               (self.video_info['frames'] / n)), n)
            
            for j in range(i + min_frames, max_j):
                if j >= n:
                    continue
                diff = np.linalg.norm(signatures[i] - signatures[j])
                time_diff = timestamps[j] - timestamps[i]
                normalized_diff = diff / (time_diff + 0.1)  # +0.1 чтобы избежать деления на 0
                
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
        
        best_pairs.sort(key=lambda x: x['norm_diff'])
        top_results = best_pairs[:10]
        best_pairs.sort(key=lambda x: x['raw_diff'])
        top_raw = best_pairs[:5]
        
        return {
            'top_by_normalized': top_results,
            'top_by_raw': top_raw
        }
    
    def generate_preview(self, start_frame, end_frame, output_path="loop_preview.mp4"):
        """Создать превью найденного цикла"""
        print(f"\nСоздание превью цикла...")
        
        duration = (end_frame - start_frame) / self.video_info['fps']
        cmd = [
            self.ffmpeg_path,
            '-i', self.video_path,
            '-vf', f"trim=start_frame={start_frame}:end_frame={end_frame},loop=3:250,setpts=N/FRAME_RATE/TB",
            '-af', f"atrim=start={start_frame/self.video_info['fps']}:end={end_frame/self.video_info['fps']},aloop=3:1,asetpts=N/SR/TB",
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Превью сохранено в: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при создании превью: {e.stderr.decode()}")
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
        
        self.get_video_info()
        signatures, timestamps, frame_indices = self.extract_frame_signatures(
            sample_rate=sample_rate, 
            method=method
        )
        
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
        
        best = results['top_by_normalized'][0]
        print(f"\n1. Установите IN точку на: {best['start_time']:.3f} сек")
        print(f"2. Установите OUT точку на: {best['end_time']:.3f} сек")
        print(f"3. Длительность цикла: {best['duration']:.3f} сек")
        print(f"\nИли используйте номера кадров:")
        print(f"  IN: кадр {best['start_idx']}")
        print(f"  OUT: кадр {best['end_idx'] - 1} (включительно)")
        
        if create_preview:
            preview_file = f"loop_preview_{best['start_idx']}_{best['end_idx']}.mp4"
            self.generate_preview(best['start_idx'], best['end_idx'], preview_file)
            
            print(f"\nПревью создано: {preview_file}")
            print("Совет: Добавьте кроссфейд на 1-3 кадра в DaVinci Resolve для более плавного перехода.")
        with open("loop_points.txt", "w") as f:
            f.write(f"DaVinci Resolve Loop Points\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"FPS: {self.video_info['fps']}\n\n")
            
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
  %(prog)s video.mp4 --ffmpeg /usr/local/bin/ffmpeg
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
    
    if not Path(args.video).exists():
        print(f"Ошибка: Файл '{args.video}' не найден")
        sys.exit(1)
    try:
        subprocess.run([args.ffmpeg, '-version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Ошибка: FFmpeg не найден по пути '{args.ffmpeg}'")
        print("Убедитесь, что FFmpeg установлен и добавлен в PATH")
        sys.exit(1)
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