
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

# Загружаем API-ключ
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Не найден GEMINI_API_KEY в файле .env")

# Подключаем модель Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key
)

# System Message из PromptHub — Lunatic
system_message = """
I want you to act as a lunatic. The lunatic's sentences are meaningless.
The words used by lunatic are completely arbitrary.
The lunatic does not make logical sentences in any way.
My first suggestion request is "I need help creating lunatic sentences
for my new series called Hot Skull, so write 10 sentences for me".
"""

# Создаём шаблон промпта
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{request}")
])

# Создаём цепочку LangChain
chain = prompt | llm | StrOutputParser()

# Запрашиваем текст у пользователя
user_request = input("Введите запрос: ")

# Запускаем цепочку
result = chain.invoke({
    "request": user_request
})

# Выводим результат
print("\nРЕЗУЛЬТАТ РАБОТЫ LUNATIC:\n")
print(result)
