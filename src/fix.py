import os
from openai import OpenAI

# Надо бы поумнее с ключом поступить
client = OpenAI(api_key="Your key here")


def fix(raw_latex: str) -> str:
    """
    Отправляет сырой LaTeX-код в ChatGPT и просит исправить ошибки компиляции.
    Возвращает откорректированный вариант.
    """
    developer_message = "You are helpful assistant whose task is to fix Latex code so that it compiles successfully. Write only fixed Latex code - no additional text is needed. Also do not add extra constructions - you only need to fix the given formula and users themselves will decide how to put it into their files"
    user_message = f"Fix this Latex code - {raw_latex}"
    try:
        response = client.chat.completions.create(
            model="o3-mini-2025-01-31",
            messages=[
                {
                    "role": "developer",
                    "content": developer_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            response_format = {"type" : "text"}
        )
        return response.choices[0].message.content
    except:
        return raw_latex
        
