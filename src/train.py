import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pathlib import Path

from model import ImageToLatexModel
from data.dataloader import DataGen, dynamic_collate_fn
from metrics.bleu_score import compute_bleu

from torch.amp import autocast, GradScaler

def indices_to_latex(sequence):
    annotation = ''.join([chr(idx) if idx > 2 else '' for idx in sequence])
    return annotation

# ------------------------- ПАРАМЕТРЫ --------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к данным
SAMPLES_DIR = Path.cwd() / ".." / "samples"
DATA_BASE_DIR = SAMPLES_DIR / "images" / "formula_images_processed"
TRAIN_DATA_PATH = SAMPLES_DIR / "im2latex_train_filter.lst"
TRAIN_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
VAL_DATA_PATH = SAMPLES_DIR / "im2latex_validate_filter.lst"
VAL_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"

# ---------------- ПУТЬ ДЛЯ СОХРАНЕНИЯ МОДЕЛИ ------------------------- #
PARAMS_DIR = Path("/content/drive/MyDrive/params_new_model")
os.makedirs(PARAMS_DIR, exist_ok=True)

MODEL_SAVE_PATH = Path.cwd() / "models" / "image_to_latex_model.pth"
os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

# Гиперпараметры
BATCH_SIZE = 10
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5

# Размер словаря и специальные токены (обновлены для соответствия вашей модели)
VOCAB_SIZE = 131
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MAX_LENGTH = 40

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, targets, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=str(DEVICE)):
            # В режиме обучения передаём targets, получаем только логиты
            logits = model(images, tgt_tokens=targets)
            # Учитываем сдвиг: предсказываем токены начиная с позиции 1
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),  # (B * T - 1, vocab_size)
                targets[:, 1:].contiguous().view(-1)  # (B * T - 1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"[Epoch {epoch}] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        del images, targets, logits, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------ ИНФЕРЕНС (PREDICT) ------------------- #
def predict(model, dataloader, num_batches=1, compute_bleu_metric=True):
    model.eval()
    all_bleu = []
    batches_processed = 0

    with torch.no_grad():
        for images, targets, img_paths in dataloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            with autocast(dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32,
                          device_type=str(DEVICE)):
                # В режиме инференса получаем логиты и токены
                logits, generated_tokens = model(images, tgt_tokens=None)

            # Переносим на CPU для вычислений
            targets = targets.cpu()
            # generated_tokens — это список переменной длины, каждый элемент — тензор

            for i in range(len(images)):
                ref_tokens = targets[i, 1:].tolist()  # Удаляем SOS из целевой последовательности
                cand_tokens = generated_tokens[i][1:].tolist()  # Предсказанные токены уже обрезаны по EOS

                if compute_bleu_metric:
                    bleu_score = compute_bleu(cand_tokens, [ref_tokens])
                    all_bleu.append(bleu_score)
                else:
                    print("BLEU вычисление отключено.")
                    bleu_score = None

                real_str = indices_to_latex(ref_tokens)
                pred_str = indices_to_latex(cand_tokens)
                print(f"=== Sample {i + 1} ===")
                print(f"  Path : {img_paths[i]}")
                print(f"  Real : {real_str}")
                print(f"  Pred : {pred_str}")
                print(f"BLEU : {bleu_score:.2f}" if bleu_score is not None else "BLEU: N/A")

            del images, targets, logits, generated_tokens
            torch.cuda.empty_cache()

            batches_processed += 1
            if batches_processed >= num_batches:
                break

    if compute_bleu_metric and all_bleu:
        avg_bleu = sum(all_bleu) / len(all_bleu)
        print(f"Average BLEU: {avg_bleu:.2f}")
    elif compute_bleu_metric:
        print("No BLEU scores computed.")

# -------------------- MAIN --------------------- #
def main():
    # Создаём датасеты
    train_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=TRAIN_DATA_PATH,
        label_path=TRAIN_LABEL_PATH,
        max_decoder_l=MAX_LENGTH
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dynamic_collate_fn,
        drop_last=True,
        num_workers=2
    )

    val_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=VAL_DATA_PATH,
        label_path=VAL_LABEL_PATH,
        max_decoder_l=MAX_LENGTH
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dynamic_collate_fn,
        drop_last=False,
        num_workers=2
    )

    print("Device:", DEVICE)
    print("Creating model...")
    model = ImageToLatexModel(
        vocab_size=VOCAB_SIZE,
        enc_hidden_dim=1920,  # Должно быть кратно количеству голов в энкодере
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Учительское принуждение не используется в трансформерной модели
    # START_TEACHER_FORCING и END_TEACHER_FORCING убраны

    # Восстановление последней контрольной точки
    checkpoint_files = list(PARAMS_DIR.glob("model_epoch_*.pth"))
    if checkpoint_files:
        def extract_epoch(checkpoint_path: Path):
            return int(checkpoint_path.stem.split("_")[-1])

        checkpoint_files.sort(key=extract_epoch)
        latest_checkpoint = checkpoint_files[-1]
        latest_epoch = extract_epoch(latest_checkpoint)
        print(f"Найден чекпоинт {latest_checkpoint}, возобновляем обучение с эпохи {latest_epoch + 1}")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE,weights_only=True))
        start_epoch = latest_epoch + 1
    else:
        print("Чекпоинты не найдены, начинаем обучение с нуля.")
        start_epoch = 1

    # Обучение
    predict(model, val_loader, num_batches=1, compute_bleu_metric=True)
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} ===")

        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            epoch
        )

        print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")
        print("--- Пример инференса (1 батч) ---")
        predict(model, val_loader, num_batches=1, compute_bleu_metric=True)

        checkpoint_path = PARAMS_DIR / f"model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Чекпоинт сохранён: {checkpoint_path}")

    print("Training done!")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()