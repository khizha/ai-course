
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# Загружаем API-ключ
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Не найден GEMINI_API_KEY в файле .env")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = api_key

# Загружаем текстовый документ
print("Start reading document...")

loader = TextLoader("document.txt", encoding="utf-8")
documents = loader.load()

# Разбиваем документ на фрагменты
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

splits = text_splitter.split_documents(documents)

# Создаём эмбеддинги и векторное хранилище
print("Start embedding...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = InMemoryVectorStore.from_documents(
    splits,
    embeddings
)

# Создаём Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# Подключаем Gemini
llm = ChatGoogleGenerativeAI(
    #model="gemini-3.8-flash",
    model="gemini-3.1-flash-lite",
    google_api_key=api_key
)

# Создаём промпт для ответов по документу
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the user's question using only the provided context.
    If the answer is not contained in the context,
    say that the information is not available in the document.

    Context:
    {context}
    """),
    ("human", "{question}")
])

# Создаём цепочку LangChain
chain = prompt | llm | StrOutputParser()

# Задаём вопросы
while True:
    question = input("\nВведите вопрос (или exit для выхода): ")

    if question.lower() == "exit":
        print("Программа завершена.")
        break

    # Ищем подходящие фрагменты документа
    docs = retriever.invoke(question)

    # Формируем контекст
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Получаем ответ Gemini
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nОТВЕТ GEMINI:\n")
    print(response)
