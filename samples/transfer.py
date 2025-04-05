import os
import shutil

# Пути к исходной и целевой папкам
source_dir = "part5"
target_dir = "part1"

# Создать целевую папку, если её нет
os.makedirs(target_dir, exist_ok=True)

# Переместить все файлы из part2 в part1
for filename in os.listdir(source_dir):
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    
    # Проверка, что это файл (а не подпапка)
    if os.path.isfile(source_path):
        shutil.move(source_path, target_path)
        print(f"Файл {filename} перемещён.")