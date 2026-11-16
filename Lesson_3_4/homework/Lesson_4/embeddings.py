import os
import numpy as np
import faiss

from dotenv import load_dotenv
from google import genai

# API-ключ - в файле .env
load_dotenv("../../../.env")

# получаем API-ключ
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Не найден GEMINI_API_KEY в файле .env")

# создаём клиента Gemini
client = genai.Client(api_key=api_key)

# модель для получения эмбеддингов
EMBEDDING_MODEL = "gemini-embedding-001"

print("Gemini API успешно подключён!")

print("\n" + "=" * 60)
print("1. ПОЛУЧЕНИЕ И СРАВНЕНИЕ ЭМБЕДДИНГОВ")
print("=" * 60)

# два текста для преобразования в эмбеддинги и сравнения
text1 = "Погода нынче дрянь"
text2 = "На улице холодно и туманно"

# получаем эмбеддинг первого текста
response1 = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=text1
)

# получаем эмбеддинг второго текста
response2 = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=text2
)

# преобразуем эмбеддинги в массивы NumPy
embedding1 = np.array(
    response1.embeddings[0].values,
    dtype=np.float32
)

embedding2 = np.array(
    response2.embeddings[0].values,
    dtype=np.float32
)

print("\nПервый текст:", text1)
print("Размерность:", embedding1.shape)

print("\nВторой текст:", text2)
print("Размерность:", embedding2.shape)

# функция сравнения двух векторов. использует косинусное сходство
def get_similarity(vector1, vector2):
    """
        Результат показывает, насколько близки два вектора по направлению.
            1.0 — векторы направлены одинаково.
            0.0 — векторы перпендикулярны.
            -1.0 — векторы направлены противоположно.
    """
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


# сравниваем два эмбеддинга
similarity = get_similarity(embedding1, embedding2)

print("\nСходство текстов:")
print(round(float(similarity), 4))

print("\n" + "=" * 60)
print("2. ПОИСК ПОХОЖИХ ТЕКСТОВ")
print("=" * 60)

texts_to_index = [
    "А погода сегодня действительно дрянь",
    "На улице шёл дождь и рота красногвардейцев",
    "Я люблю музыку",
    "Нынче ветренно и волны с перехлёстом",
    "Сегодня дождь и скверно",
    "Кот уронил цветочный горшок",
    "Завтра будет новый день",
    "Мне нравится думать"
]

def build_index(texts):
    """
    Создаёт векторный индекс FAISS на основе списка текстов.

    Каждый текст преобразуется в эмбеддинг с помощью Gemini API.
    Полученные векторы объединяются в массив, нормализуются
    и добавляются в индекс FAISS.

    """
    embeddings = []

    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        vector = np.array(
            response.embeddings[0].values,
            dtype=np.float32
        )

        embeddings.append(vector)

    embeddings_array = np.ascontiguousarray(
        np.stack(embeddings),
        dtype=np.float32
    )

    faiss.normalize_L2(embeddings_array)

    index = faiss.IndexFlatIP(embeddings_array.shape[1])

    index.add(embeddings_array)

    return index


def semantic_search(query, index, texts, k=3):
    """
    Выполняет семантический поиск наиболее похожих текстов.

    Преобразует поисковый запрос в эмбеддинг с помощью Gemini API,
    нормализует полученный вектор и ищет наиболее близкие
    векторы в индексе FAISS.

        query (str): Поисковый запрос пользователя.
        index (faiss.IndexFlatIP): Индекс с эмбеддингами текстов.
        texts (list): Исходный список текстов.
        k (int, optional): Количество результатов поиска.
            По умолчанию 3.

        list: Список найденных текстов и значений сходства.
    """

    # получаем эмбеддинг поискового запроса
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query
    )

    query_vector = np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )

    # пдготавливаем вектор для поиска
    query_array = np.ascontiguousarray(
        query_vector.reshape(1, -1),
        dtype=np.float32
    )

    faiss.normalize_L2(query_array)

    # ищем наиболее похожие тексты
    scores, indices = index.search(query_array, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text": texts[idx],
            "similarity": float(score)
        })

    return results

# создаём индекс
index = build_index(texts_to_index)

print("\nРазмерность индекса:", index.d)
print("Количество векторов:", index.ntotal)

# поисковый запрос
search_query = "Какая сегодня погода?"

# выполняем поиск
results = semantic_search(
    search_query,
    index,
    texts_to_index,
    k=3
)

print("\n--- Результаты поиска ---")
print("Запрос:", search_query)

for result in results:
    print(
        f"{result['similarity']:.4f} | "
        f"{result['text']}"
    )