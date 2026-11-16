import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

print("1. Загружаем API key")

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

print("2. Подключаем модель")

model = genai.GenerativeModel("gemini-3.8-flash")

print("3. Отправляем запрос")

start = time.time()

response = model.generate_content(
    "Какой сегодня день недели, месяц, год?"
)

end = time.time()

print("4. Получили ответ!")
print("Ответ:", response.text)
print(f"Время выполнения: {end - start:.2f} секунд")