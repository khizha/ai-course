import textwrap

from google import genai
from google.genai import types
import os
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# API-ключ - в файле .env
load_dotenv("../../../.env")

# получаем API-ключ
api_key = os.getenv("GEMINI_API_KEY")

# создаём клиента API с таймаутом 10 секунд (10000 миллисекунд)
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=10000)
)

# защита от частых запросов
last_request_time = 0

def rate_limit():
    global last_request_time

    current_time = time.time()
# сколько прошло времени с последнего запроса
    elapsed = current_time - last_request_time

# если с предыдущего запроса прошло меньше двух секунд, ждём столько, сколько осталось
    if elapsed < 2:
        time.sleep(2 - elapsed)

    last_request_time = time.time()


# автоматический повтор запроса при ошибках
@retry( # декоратор, который автоматически повторяет выполнение функции, если она выполнилась с ошибкой
    stop=stop_after_attempt(3), # повторяем максимум 3 раза
    wait=wait_exponential(multiplier=2, min=2, max=8) # время между попытками возрастает экпоненциально- 2, 4, 8 секунд
)
def generate_answer():
    rate_limit()

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=["Isn't current measured in amperes?"]
    )

    return response


# отправляем запрос и обрабатываем ошибки
try:
    response = generate_answer()
    print("Ответ Gemini:")
    print(textwrap.fill(response.text, width=80))

except Exception as error:
    print("Не удалось получить ответ от API:")
    print(error)
