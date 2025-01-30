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

# Пути к данным
SAMPLES_DIR = Path.cwd() / ".." / "samples"
DATA_BASE_DIR = SAMPLES_DIR / "images" / "formula_images_processed"
TRAIN_DATA_PATH = SAMPLES_DIR / "im2latex_train_filter.lst"
TRAIN_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
VAL_DATA_PATH   = SAMPLES_DIR / "im2latex_validate_filter.lst"
VAL_LABEL_PATH  = SAMPLES_DIR / "im2latex_formulas.tok.lst"

# Путь для сохранения модели
MODEL_SAVE_PATH = Path.cwd() / "models" / "image_to_latex_model.pth"
os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

# Гиперпараметры
BATCH_SIZE = 6
NUM_EPOCHS = 10         
LEARNING_RATE = 1e-4     
START_TEACHER_FORCING = 0.7  
END_TEACHER_FORCING   = 0.2 

# Размер словаря и специальные токены
VOCAB_SIZE = 131
PAD_IDX = 0
SOS_IDX = 129
EOS_IDX = 130
MAX_LENGTH = 300

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0

    for step, (images, targets, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Применяем Mixed Precision
        with autocast(device_type=str(DEVICE)):
            logits, alphas = model(
                images, 
                tgt_tokens=targets, 
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            B, T = targets.shape
            vocab_size = logits.size(-1)
            loss = criterion(
                logits.view(-1, vocab_size),
                targets[:, 1:].contiguous().view(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Печать каждые 50 итераций
        if (step + 1) % 50 == 0:
            print(f"[Epoch {epoch}] Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Явно очищаем память
        del images, targets, logits, alphas, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------ ИНФЕРЕНС (PREDICT) ------------------- #
def predict(model, dataloader, num_batches=1):
    model.eval()
    batches_processed = 0

    with torch.no_grad():
        for images, targets, img_paths in dataloader:
            images = images.to(DEVICE)

            with autocast(dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32,device_type=str(DEVICE)):
                generated_tokens, alphas_all = model(images, tgt_tokens=None, teacher_forcing_ratio=0.0)

            generated_tokens = generated_tokens.cpu()
            targets = targets.cpu()

            for i in range(len(images)):
                real_str = indices_to_latex(targets[i].tolist())
                pred_str = indices_to_latex(generated_tokens[i].tolist())
                print(f"=== Sample {i+1} ===")
                print(f"  Path : {img_paths[i]}")
                print(f"  Real : {real_str}")
                print(f"  Pred : {pred_str}")

            # Здесь можно визуализировать внимание, если нужно
            # visualize_attention_maps(images, alphas_all, generated_tokens)

            del images, targets, generated_tokens, alphas_all
            torch.cuda.empty_cache()

            batches_processed += 1
            if batches_processed >= num_batches:
                break

# -------------------- MAIN --------------------- #
def main():
    # Создаём датасеты
    train_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=TRAIN_DATA_PATH,
        label_path=TRAIN_LABEL_PATH
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        drop_last=True,
        num_workers=4
    )

    val_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=VAL_DATA_PATH,
        label_path=VAL_LABEL_PATH
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn, 
        drop_last=False,
        num_workers=4
    )

    print("Device:", DEVICE)
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

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler(device=str(DEVICE))

    # Schedule для teacher forcing: линейно уменьшаем от START до END за NUM_EPOCHS
    teacher_forcing_schedule = torch.linspace(START_TEACHER_FORCING, END_TEACHER_FORCING, steps=NUM_EPOCHS).tolist()

    for epoch in range(1, NUM_EPOCHS+1):
        tf_ratio = teacher_forcing_schedule[epoch-1]
        print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS}, teacher_forcing_ratio={tf_ratio:.2f} ===")

        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            epoch,
            teacher_forcing_ratio=tf_ratio
        )

        print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")

        # Небольшой predict
        print("--- Пример инференса (1 батч) ---")
        predict(model, val_loader, num_batches=1)

    print("Training done!")

    # Сохраняем модель
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
