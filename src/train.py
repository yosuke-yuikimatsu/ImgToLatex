import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pathlib import Path


from model import ImageToLatexModel
from data.dataloader import DataGen, collate_fn, indices_to_latex, visualize_batch_with_labels

# ------------------------- ПАРАМЕТРЫ --------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к данным:
SAMPLES_DIR = Path.cwd() / ".." / "samples"
DATA_BASE_DIR = SAMPLES_DIR / "images" / "formula_images_processed"
TRAIN_DATA_PATH = SAMPLES_DIR / "im2latex_train_filter.lst"
TRAIN_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
VAL_DATA_PATH   = SAMPLES_DIR / "im2latex_validate_filter.lst"    # 
VAL_LABEL_PATH  = SAMPLES_DIR / "im2latex_formulas.tok.lst"

# Гиперпараметры
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3

# Размер словаря и специальные токены
VOCAB_SIZE = 131          # примерный размер словаря
PAD_IDX = 0               # токен паддинга
SOS_IDX = 129             # специальный токен начала последовательности
EOS_IDX = 130             # специальный токен конца последовательности
MAX_LENGTH = 300          # максимально допустимая длина при генерации

# ---------------------- ФУНКЦИЯ ОБУЧЕНИЯ ОДНОГО ЭПОХА ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, epoch, teacher_forcing_ratio=0.5):
    """
    Одна эпоха обучения:
      - Идём по батчам
      - Считаем лосс
      - Обновляем параметры
    """
    model.train()
    total_loss = 0.0
    for step, (images, targets, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Модель возвращает (logits, alphas) при tgt_tokens != None
        logits, alphas = model(images, tgt_tokens=targets, teacher_forcing_ratio=teacher_forcing_ratio)
        # logits: (B, T-1, vocab_size)
        # targets: (B, T)

        B, T = targets.shape
        vocab_size = logits.size(-1)
        # Сдвигаем targets на 1, чтобы сравнить предсказания i-го шага с токеном на позиции i+1
        loss = criterion(
            logits.view(-1, vocab_size),           # (B*(T-1), vocab_size)
            targets[:, 1:].contiguous().view(-1)   # (B*(T-1),)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:  # Периодический вывод
            print(f"Epoch [{epoch}] Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------ ФУНКЦИЯ ДЛЯ ИНФЕРЕНСА (PREDICT) ------------------- #
def predict(model, dataloader, num_batches=1, visualize_attention=True):
    """
    Делает инференс на нескольких батчах (по умолчанию 1).
      - Выводит предсказанные токены и их преобразование в строку.
      - (Опционально) Выводит визуализацию внимания для нескольких примеров.
    """
    model.eval()
    batches_processed = 0

    with torch.no_grad():
        for images, targets, img_paths in dataloader:
            images = images.to(DEVICE)
            # Генерация (tgt_tokens=None)
            generated_tokens, alphas_all = model(images, tgt_tokens=None, teacher_forcing_ratio=0.0)

            # Переводим на CPU
            generated_tokens = generated_tokens.cpu()
            targets = targets.cpu()

            # Для каждого образца в батче выводим реальную метку и предсказанную
            for i in range(len(images)):
                # Реальная метка
                real_latex = indices_to_latex(targets[i].tolist())
                # Предсказанная метка
                pred_latex = indices_to_latex(generated_tokens[i].tolist())

                print(f"=== Sample {i+1} ===")
                print(f"  Image path : {img_paths[i]}")
                print(f"  Real  text : {real_latex}")
                print(f"  Pred  text : {pred_latex}")

            # (Опционально) Визуализируем картинки + карты внимания
            if visualize_attention:
                visualize_attention_maps(images, alphas_all, generated_tokens)

            batches_processed += 1
            if batches_processed >= num_batches:
                break

# ------------- ВИЗУАЛИЗАЦИЯ КАРТ ВНИМАНИЯ (ATTENTION MAPS) ------------ #
def visualize_attention_maps(images, alphas_all, generated_tokens, max_samples=2):
    """
    Визуализация attention maps с учётом выхода энкодера (B, H/8, W/8, 512).
    
    - images: тензор (B, 3, H, W)
    - alphas_all: список или тензор весов внимания на каждом шаге (длина ~ длина сгенерированной последовательности)
    - generated_tokens: (B, seq_len) - сгенерированные токены
    - max_samples: сколько примеров из батча хотим визуализировать
    """
    import matplotlib.pyplot as plt
    import cv2  # Для более качественного масштабирования

    # Перекладываем изображения на CPU и в numpy
    images_np = images.cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
    B, H, W, _ = images_np.shape

    # Количество примеров для визуализации
    num_show = min(B, max_samples)

    # Вычисляем H_ и W_ на основе размеров изображений
    H_ = H // 8
    W_ = W // 8

    for i in range(num_show):
        # Определяем количество шагов для визуализации
        steps_to_show = min(4, generated_tokens.size(1))  # Или другое число шагов

        # Создаём подграфики: оригинальное изображение + attention maps
        fig, axes = plt.subplots(1, steps_to_show + 1, figsize=(5 * (steps_to_show + 1), 5))
        fig.suptitle(f"Attention Visualization (Sample {i + 1})", fontsize=16)

        # Показываем оригинальное изображение слева
        axes[0].imshow(images_np[i])
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        for t in range(steps_to_show):
            if isinstance(alphas_all, list):
                alpha_t = alphas_all[t]  # alphas формируется как список
                # alpha_t: (B, H'*W') -> (B, H/8 * W/8)
                alpha_t_i = alpha_t[i]    # Для конкретного примера из батча
            else:
                # Если alphas_all - тензор, например (T, B, H'*W')
                alpha_t = alphas_all[t]    # (B, H'*W')
                alpha_t_i = alpha_t[i]     # (H'*W',)

            # Восстанавливаем пространственные размеры
            alpha_t_i = alpha_t_i.view(H_, W_).cpu().numpy()

            # Нормализуем attention map
            alpha_t_i = alpha_t_i / (alpha_t_i.max() + 1e-8)

            # Масштабируем attention map до размеров оригинального изображения
            # Используем cv2.resize для качественного масштабирования
            alpha_t_i_resized = cv2.resize(alpha_t_i, (W, H))
            
            # Визуализируем поверх изображения
            axes[t + 1].imshow(images_np[i], alpha=0.8)
            axes[t + 1].imshow(alpha_t_i_resized, cmap='jet', alpha=0.6)
            token_id = generated_tokens[i, t].item()
            axes[t + 1].set_title(f"Step {t + 1}, Token={token_id}")
            axes[t + 1].axis('off')

        plt.tight_layout()
        plt.show()



def main():
    # ----------------- СОЗДАЁМ ДАТАСЕТЫ И DATALOADER'Ы ------------------
    print("Loading datasets...")
    train_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=TRAIN_DATA_PATH,
        label_path=TRAIN_LABEL_PATH
    )
    print("Checking current device:", DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

    val_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=VAL_DATA_PATH,
        label_path=VAL_LABEL_PATH
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)

    # ----------------- СОЗДАЁМ МОДЕЛЬ, ОПТИМАЙЗЕР -----------------------
    print("Creating model...")
    model = ImageToLatexModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=256,
        enc_hidden_dim=256,
        dec_hidden_dim=512,
        pad_idx=PAD_IDX,
        sos_index=SOS_IDX,
        eos_index=EOS_IDX,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ----------------- ЦИКЛ ОБУЧЕНИЯ ------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== EPOCH {epoch} ===")
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            teacher_forcing_ratio=0.2  # можно менять в процессе обучения
        )
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        # (пример) после каждой эпохи делаем небольшой predict
        print("\n--- Пример инференса на паре батчей ---")
        predict(model, val_loader, num_batches=1, visualize_attention=True)

    print("Training done!")

if __name__ == "__main__":
    main()
