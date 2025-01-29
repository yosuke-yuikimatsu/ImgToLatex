import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pathlib import Path

from model import ImageToLatexModel
from data.dataloader import DataGen, collate_fn, indices_to_latex, visualize_batch_with_labels

# --- Добавляем из torch.cuda.amp ---
from torch.amp import autocast, GradScaler

# ------------------------- ПАРАМЕТРЫ --------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Пути к данным:
SAMPLES_DIR = Path.cwd() / ".." / "samples"
DATA_BASE_DIR = SAMPLES_DIR / "images" / "formula_images_processed"
TRAIN_DATA_PATH = SAMPLES_DIR / "im2latex_train_filter.lst"
TRAIN_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
VAL_DATA_PATH   = SAMPLES_DIR / "im2latex_validate_filter.lst"
VAL_LABEL_PATH  = SAMPLES_DIR / "im2latex_formulas.tok.lst"

# Гиперпараметры
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3

# Размер словаря и специальные токены
VOCAB_SIZE = 131    # примерный размер словаря
PAD_IDX = 0         # токен паддинга
SOS_IDX = 129       # специальный токен начала последовательности
EOS_IDX = 130       # специальный токен конца последовательности
MAX_LENGTH = 300    # максимально допустимая длина при генерации

# ---------------------- ФУНКЦИЯ ОБУЧЕНИЯ ОДНОГО ЭПОХА ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, teacher_forcing_ratio=0.5):
    """
    Одна эпоха обучения:
      - Идём по батчам
      - Считаем лосс (с помощью autocast)
      - Обновляем параметры (GradScaler)
      - После каждого батча чистим память
    """
    model.train()
    total_loss = 0.0

    for step, (images, targets, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Применяем Mixed Precision
        with autocast(device_type=str(DEVICE)):
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

        # backward через GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch [{epoch}] Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Явная очистка GPU-памяти от временных тензоров
        del images, targets, logits, alphas, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------ ФУНКЦИЯ ДЛЯ ИНФЕРЕНСА (PREDICT) ------------------- #
def predict(model, dataloader, num_batches=1, visualize_attention=False):
    """
    Делает инференс на нескольких батчах (по умолчанию 1).
      - Выводит предсказанные токены
      - (Опционально) Визуализируем внимание
    """
    model.eval()
    batches_processed = 0

    with torch.no_grad():
        for images, targets, img_paths in dataloader:
            images = images.to(DEVICE)

            # Генерация (tgt_tokens=None) в автокасте смысла нет, т.к. там нет backward,
            # но можно для консистентности
            with autocast(device_type=str(DEVICE)):
                generated_tokens, alphas_all = model(images, tgt_tokens=None, teacher_forcing_ratio=0.0)

            generated_tokens = generated_tokens.cpu()
            targets = targets.cpu()

            for i in range(len(images)):
                real_latex = indices_to_latex(targets[i].tolist())
                pred_latex = indices_to_latex(generated_tokens[i].tolist())

                print(f"=== Sample {i+1} ===")
                print(f"  Image path : {img_paths[i]}")
                print(f"  Real  text : {real_latex}")
                print(f"  Pred  text : {pred_latex}")

            # (Опционально) Визуализируем attention
            if visualize_attention:
                visualize_attention_maps(images, alphas_all, generated_tokens)

            del images, targets, generated_tokens, alphas_all
            torch.cuda.empty_cache()

            batches_processed += 1
            if batches_processed >= num_batches:
                break

# -------------------- ВИЗУАЛИЗАЦИЯ КАРТ ВНИМАНИЯ ---------------------- #
def visualize_attention_maps(images, alphas_all, generated_tokens, max_samples=2):
    """
    Визуализация attention maps. Предполагается, что
    alphas_all[t] имеет форму (B, H'*W') или аналогичную.
    """
    import matplotlib.pyplot as plt
    import cv2

    images_np = images.cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
    B, H, W, _ = images_np.shape
    num_show = min(B, max_samples)

    H_ = H // 8
    W_ = W // 8

    for i in range(num_show):
        steps_to_show = min(4, generated_tokens.size(1))
        fig, axes = plt.subplots(1, steps_to_show + 1, figsize=(5*(steps_to_show+1), 5))
        fig.suptitle(f"Attention Visualization (Sample {i + 1})", fontsize=16)

        axes[0].imshow(images_np[i])
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        for t in range(steps_to_show):
            if isinstance(alphas_all, list):
                alpha_t = alphas_all[t]    # (B, H'*W')
                alpha_t_i = alpha_t[i]
            else:
                alpha_t = alphas_all[t]    # (B, H'*W')
                alpha_t_i = alpha_t[i]

            alpha_t_i = alpha_t_i.view(H_, W_).cpu().numpy()
            alpha_t_i = alpha_t_i / (alpha_t_i.max() + 1e-8)

            alpha_t_i_resized = cv2.resize(alpha_t_i, (W, H))
            axes[t+1].imshow(images_np[i], alpha=0.8)
            axes[t+1].imshow(alpha_t_i_resized, cmap='jet', alpha=0.6)
            token_id = generated_tokens[i, t].item()
            axes[t+1].set_title(f"Step {t+1}, Token={token_id}")
            axes[t+1].axis('off')

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

    val_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=VAL_DATA_PATH,
        label_path=VAL_LABEL_PATH
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)

    print("Checking current device:", DEVICE)

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

    # Создаём GradScaler для amp
    scaler = GradScaler(device=str(DEVICE))

    # ----------------- ЦИКЛ ОБУЧЕНИЯ ------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== EPOCH {epoch} ===")

        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,  # Перадаем scaler
            epoch=epoch,
            teacher_forcing_ratio=0.2
        )
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        # После эпохи можно сделать небольшой predict (1 батч)
        print("\n--- Пример инференса ---")
        predict(model, val_loader, num_batches=1, visualize_attention=False)

    print("Training done!")


if __name__ == "__main__":
    main()
