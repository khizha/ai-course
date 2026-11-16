from google import genai
import os
import time
from dotenv import load_dotenv

# загружаем API-ключ из файла .env
load_dotenv("../../../.env")

# получаем API-ключ
api_key = os.getenv("GEMINI_API_KEY")

# создаём клиента API
client = genai.Client(api_key=api_key)

# отправляем запрос к модели
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=["Isn't current measured in amperes?"]
)

print(response.text)
