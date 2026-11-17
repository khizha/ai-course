
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import WebBaseLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

model.eval()

def generate_text(prompt_value):

    messages = prompt_value.to_messages()

    hf_messages = []

    for message in messages:
        if message.type == "system":
            role = "system"
        elif message.type == "human":
            role = "user"
        else:
            role = "assistant"

        hf_messages.append({
            "role": role,
            "content": message.content
        })

    inputs = tokenizer.apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=6000
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=250,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs.shape[-1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


llm = RunnableLambda(generate_text)


def get_docs(url):
    loader = WebBaseLoader(url)
    return loader.load()


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты помощник, специализирующийся на суммаризации текстов. "
        "Отвечай исключительно на русском языке. "
        "Не рассказывай о себе. Не используй шаблонные вступления. "
        "Составь краткое и информативное изложение текста."
    ),
    (
        "human",
        "Сделай краткое изложение следующего текста:\n\n{context}"
    )
])


chain = create_stuff_documents_chain(llm, prompt)


user_url = input("Введите адрес веб-страницы: ")

docs = get_docs(user_url)

print("Количество документов:", len(docs))
print("Размер текста:", sum(len(doc.page_content) for doc in docs))

result = chain.invoke({
    "context": docs
})

print("\nРЕЗУЛЬТАТ СУММАРИЗАЦИИ:\n")
print(result)
