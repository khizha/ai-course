import pandas as pd

"""
СОЗДАНИЕ ДАТАСЕТА
Создаём собственный датасет (25 записей) для fine-tuning.
Тема: вопросы и ответы по основам языка Python.
Каждая запись содержит вопрос и правильный ответ.
Данные сохраняем в DataFrame с помощью библиотеки pandas.
"""

# Датасет: вопросы и ответы по Python
data = [
    {"question": "Что такое Python?",
     "answer": "Python — это высокоуровневый язык программирования."},

    {"question": "Что такое переменная?",
     "answer": "Переменная — это именованная область памяти для хранения данных."},

    {"question": "Как создать список в Python?",
     "answer": "Список создаётся с помощью квадратных скобок, например: my_list = [1, 2, 3]."},

    {"question": "Что такое кортеж?",
     "answer": "Кортеж — это упорядоченная неизменяемая коллекция элементов."},

    {"question": "Для чего используется словарь?",
     "answer": "Словарь хранит данные в формате ключ-значение."},

    {"question": "Что такое цикл for?",
     "answer": "Цикл for используется для перебора элементов последовательности."},

    {"question": "Что такое цикл while?",
     "answer": "Цикл while повторяет выполнение кода, пока условие истинно."},

    {"question": "Для чего нужна функция?",
     "answer": "Функция — это блок кода, который можно вызывать многократно."},

    {"question": "Как объявить функцию?",
     "answer": "Для объявления функции используется ключевое слово def."},

    {"question": "Что такое условный оператор?",
     "answer": "Условный оператор if позволяет выполнять код в зависимости от условия."},

    {"question": "Что такое строка?",
     "answer": "Строка — это последовательность символов."},

    {"question": "Как узнать длину списка?",
     "answer": "Для определения длины списка используется функция len()."},

    {"question": "Что такое индекс списка?",
     "answer": "Индекс — это позиция элемента в списке. Нумерация начинается с нуля."},

    {"question": "Что делает оператор break?",
     "answer": "Оператор break немедленно завершает выполнение цикла."},

    {"question": "Что делает оператор continue?",
     "answer": "Оператор continue пропускает текущую итерацию цикла."},

    {"question": "Что такое библиотека в Python?",
     "answer": "Библиотека — это набор готовых инструментов и функций для решения задач."},

    {"question": "Для чего используется библиотека pandas?",
     "answer": "Pandas используется для обработки и анализа табличных данных."},

    {"question": "Что такое исключение?",
     "answer": "Исключение — это ошибка, возникающая во время выполнения программы."},

    {"question": "Для чего нужен try except?",
     "answer": "Конструкция try except позволяет обрабатывать исключения."},

    {"question": "Что такое цикл?",
     "answer": "Цикл — это конструкция, позволяющая многократно выполнять блок кода."},

    {"question": "Что такое целое число?",
     "answer": "Целое число — это тип данных int, например 10 или -5."},

    {"question": "Что такое float?",
     "answer": "Float — это тип данных для хранения чисел с плавающей точкой."},

    {"question": "Что такое Boolean?",
     "answer": "Boolean — это логический тип данных со значениями True и False."},

    {"question": "Что такое None?",
     "answer": "None обозначает отсутствие значения."},

    {"question": "Как вывести текст на экран?",
     "answer": "Для вывода текста используется функция print()."}
]

df = pd.DataFrame(data)

print("Количество записей:", len(df))
print(df.head())

"""
ОЧИСТКА И РАЗДЕЛЕНИЕ ДАННЫХ
Подготавливаем датасет к обучению модели:
- удаляем повторяющиеся вопросы;
- очищаем текст от лишних пробелов;
- удаляем записи с отсутствующими значениями;
- разделяем данные на обучающую (80%) и тестовую (20%) выборки.

Обучающая выборка будет использоваться для fine-tuning,
тестовая — для последующей проверки качества модели.
"""

from sklearn.model_selection import train_test_split

# 1. Удаляем дубликаты
df = df.drop_duplicates(subset=["question"])

# 2. Удаляем лишние пробелы
df["question"] = df["question"].str.strip()
df["answer"] = df["answer"].str.strip()

# 3. Удаляем пустые записи
df = df.dropna()

# 4. Разделяем данные на обучающую и тестовую выборки
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("После очистки:", len(df))
print("Обучающая выборка:", len(train_df))
print("Тестовая выборка:", len(test_df))

"""
ПОДГОТОВКА ДАННЫХ ДЛЯ HUGGING FACE
Преобразуем обучающую и тестовую выборки pandas
в формат Dataset библиотеки Hugging Face.
Объединяем вопрос и ответ в один текстовый пример,
который будет использоваться для обучения модели.
"""
from datasets import Dataset

def format_data(df):
    texts = []

    for _, row in df.iterrows():
        text = (
            f"### Вопрос: {row['question']}\n"
            f"### Ответ: {row['answer']}"
        )
        texts.append({"text": text})

    return Dataset.from_list(texts)


train_dataset = format_data(train_df)
test_dataset = format_data(test_df)

print("Обучающих примеров:", len(train_dataset))
print("Тестовых примеров:", len(test_dataset))

print("\nПример обучающих данных:")
print(train_dataset[0]["text"])

"""
ЗАГРУЗКА ПРЕДОБУЧЕННОЙ МОДЕЛИ
Загружаем предварительно обученную русскоязычную модель
и токенизатор из Hugging Face.
Модель будет использоваться для последующего fine-tuning.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2" # небольшая русскоязычная предварительно обученная модель

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Загружаем модель
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Устанавливаем токен заполнения
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print("Модель успешно загружена!")
print("Количество параметров:", sum(p.numel() for p in model.parameters()))

"""
ТОКЕНИЗАЦИЯ ДАННЫХ
Преобразуем текстовые вопросы и ответы в токены,
понятные языковой модели.
Ограничиваем длину последовательности 128 токенами.
"""

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )


tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True
)

print("Токенизация завершена!")
print("Обучающих примеров:", len(tokenized_train))
print("Тестовых примеров:", len(tokenized_test))

"""
НАСТРОЙКА LoRA
Настраиваем эффективное дообучение модели с помощью LoRA.
Вместо изменения всех параметров модели обучаются
дополнительные небольшие матрицы.
Это позволяет сократить вычислительные затраты.
"""

from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["c_attn"],
    fan_in_fan_out=True
)

# Подключаем LoRA к модели
model = get_peft_model(model, peft_config)

# Выводим количество обучаемых параметров
model.print_trainable_parameters()

"""
НАСТРОЙКА И ЗАПУСК ОБУЧЕНИЯ
Настраиваем параметры обучения и запускаем fine-tuning.
Используем небольшой batch size и ограниченное
количество шагов, поскольку обучение выполняется на CPU.
"""

from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

# Подготовка данных для языковой модели
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=20,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator,
    processing_class=tokenizer
)

# Запуск обучения
print("Начинаем fine-tuning...")
trainer.train()

print("Fine-tuning завершён!")

"""
ТЕСТИРОВАНИЕ ДОOБУЧЕННОЙ МОДЕЛИ
Проверяем качество модели на тестовых данных,
которые не использовались при обучении.
Сравниваем ответы модели с правильными ответами
из исходного датасета.
"""

model.eval()

def generate_answer(question):
    prompt = f"### Вопрос: {question}\n### Ответ:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.split("### Ответ:")[-1].strip()


print("\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")

for _, row in test_df.iterrows():
    question = row["question"]
    expected_answer = row["answer"]

    predicted_answer = generate_answer(question)

    print("\nВопрос:", question)
    print("Правильный ответ:", expected_answer)
    print("Ответ модели:", predicted_answer)
    print("-" * 60)

"""    
ОЦЕНКА КАЧЕСТВА МОДЕЛИ
Вычисляем значение ошибки модели на тестовой выборке.
Чем ниже значение loss, тем лучше модель предсказывает
следующие токены на ранее не использованных данных.
"""

test_results = trainer.evaluate(
    eval_dataset=tokenized_test
)

print("Результаты оценки:")
print("Test loss:", test_results["eval_loss"])



"""
СОХРАНЕНИЕ ДОOБУЧЕННОЙ МОДЕЛИ
Сохраняем обученные LoRA-параметры и токенизатор,
чтобы использовать результат обучения в дальнейшем.
"""

model.save_pretrained("./fine_tuned_python_model")
tokenizer.save_pretrained("./fine_tuned_python_model")

print("Дообученная модель успешно сохранена!")