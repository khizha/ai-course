import torch
from transformers import pipeline
import time

device = 'cpu'

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device
)

audio_file = "Sting - Shape Of My Heart.mp3"
start = time.time()
result = pipe(
    audio_file,
    return_timestamps=True
)
end = time.time()

print(f'Total time: {end-start:.2f}')
# print(result)
print("Полная транскрипция:")
print(result["text"], end='\n\n')

print("\nСегменты:")
for chunk in result.get("chunks", []):
    start, end = chunk["timestamp"]
    text = chunk["text"]
    print(f"[{start:.2f} - {end:.2f}] {text}", end='\n\n')