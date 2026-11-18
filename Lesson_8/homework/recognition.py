import os
import time
import subprocess
import numpy as np
import torch
from transformers import pipeline

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))

audio_files = [
    "audio.wav",
    "sample-speech-short.wav",
    "sample-speech-long.wav",
    "sample-speech-noisy.wav"
]

# Путь к рабочему FFmpeg
FFMPEG = os.path.join(
    os.environ["LOCALAPPDATA"],
    "Microsoft", "WinGet", "Packages",
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "ffmpeg-9.0.2-full_build", "bin", "ffmpeg.exe"
)

print("Загрузка Whisper...")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device="cpu"
)

print("Модель загружена!\n")

for filename in audio_files:
    path = os.path.join(AUDIO_DIR, filename)

    print("=" * 60)
    print("Аудиофайл:", filename)

    if not os.path.exists(path):
        print("Файл не найден!")
        continue

    try:
        # Конвертируем аудио в WAV, 16 кГц, моно
        audio_bytes = subprocess.check_output([
            FFMPEG,
            "-y",
            "-i", path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "f32le",
            "-"
        ], stderr=subprocess.DEVNULL)

        audio = np.frombuffer(
            audio_bytes,
            dtype=np.float32
        ).copy()

        print("Длительность:", round(len(audio) / 16000, 2), "сек.")

        start_time = time.time()

        result = pipe(
            {"raw": audio, "sampling_rate": 16000},
            return_timestamps=True
        )

        elapsed = time.time() - start_time

        print("Время распознавания:", round(elapsed, 2), "сек.")
        print("\nТранскрипция:")
        print(result["text"])

        print("\nСегменты:")

        for chunk in result.get("chunks", []):
            print(chunk["timestamp"], chunk["text"])

    except Exception as e:
        print("Ошибка:", e)

print("\nРаспознавание завершено.")
