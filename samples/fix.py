import json

# Загружаем данные из файла
with open("vocab_swapped.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Извлекаем словарь token_to_id
token_to_id = data["token_to_id"]

# Преобразуем все значения в int
token_to_id = {key: int(value) for key, value in token_to_id.items()}

# Если нужно, обновляем исходный словарь и сохраняем в новый JSON
data["token_to_id"] = token_to_id

with open("vocab.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
