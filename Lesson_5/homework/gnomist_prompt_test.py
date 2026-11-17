
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

# System Message из PromptHub — Gnomist
system_message = """
I want you to act as a gnomist. You will provide me with fun,
unique ideas for activities and hobbies that can be done anywhere.
For example, I might ask you for interesting yard design suggestions
or creative ways of spending time indoors when the weather is not
favourable. Additionally, if necessary, you could suggest other
related activities or items that go along with what I requested.
My first request is "I am looking for new outdoor activities in my area".
"""

# Создаём шаблон промпта
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{request}")
])

# Создаём цепочку LangChain
chain = prompt | llm | StrOutputParser()

# Запрашиваем пользовательский ввод
request = input("Введите запрос: ")

# Получаем ответ модели
response = chain.invoke({"request": request})

print("\nРЕЗУЛЬТАТ РАБОТЫ GNOMIST:\n")
print(response)
