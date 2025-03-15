import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from pathlib import Path

from model import ImageToLatexModel
from data.dataloader import DataGen, dynamic_collate_fn, indices_to_latex
from metrics.bleu_score import compute_bleu

from torch.amp import autocast, GradScaler

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
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
START_TEACHER_FORCING = 0.9
END_TEACHER_FORCING = 0.0
RL_START_EPOCH = 50  # Эпоха, с которой начинается RL-обучение

# Размер словаря и специальные токены
VOCAB_SIZE = 131
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MAX_LENGTH = 30

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ (Supervised) ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0

    for step, (images, targets, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

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

        if (step + 1) % 50 == 0:
            print(f"[Epoch {epoch}] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.8f}")

        del images, targets, logits, alphas, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ (REINFORCE) ----------------- #
def train_one_epoch_rl(model, dataloader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, _, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=str(DEVICE)):
            corrected_predicted_tokens, rewards, loss = model(
                images,
                train=True
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            avg_reward = torch.mean(rewards).item() if rewards is not None else 0.0
            print(f"[Epoch {epoch} RL] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.8f}, Avg Reward: {avg_reward:.4f}")

        del images, corrected_predicted_tokens, rewards, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------ ИНФЕРЕНС (PREDICT) ------------------- #
def predict(model, dataloader, num_batches=1, compute_bleu_metric=True):
    model.eval()
    all_bleu = []
    batches_processed = 0
    bleu_score = None

    with torch.no_grad():
        for images, targets, img_paths in dataloader:
            images = images.to(DEVICE)

            with autocast(dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32,
                          device_type=str(DEVICE)):
                # Для инференса используем обычный forward без teacher forcing
                generated_tokens, alphas_all = model(images, tgt_tokens=None, teacher_forcing_ratio=0.0)

            generated_tokens = generated_tokens.cpu()
            targets = targets.cpu()

            for i in range(len(images)):
                ref_tokens = targets[i].tolist()[1:]
                cand_tokens = generated_tokens[i].tolist()

                if compute_bleu_metric:
                    bleu_score = compute_bleu(cand_tokens, [ref_tokens])
                    all_bleu.append(bleu_score)
                else:
                    print("BLEU вычисление отключено.")

                real_str = indices_to_latex(targets[i].tolist()[1:])
                pred_str = indices_to_latex(generated_tokens[i].tolist())
                print(f"=== Sample {i + 1} ===")
                print(f"  Path : {img_paths[i]}")
                print(f"  Real : {real_str}")
                print(f"  Pred : {pred_str}")
                print(f"BLEU : {bleu_score:.2f}" if bleu_score else "BLEU: N/A")

            del images, targets, generated_tokens, alphas_all
            torch.cuda.empty_cache()

            batches_processed += 1
            if batches_processed >= num_batches:
                break

    if compute_bleu_metric and all_bleu:
        avg_bleu = sum(all_bleu) / len(all_bleu)
        var_bleu = np.var(all_bleu)
        print(f"Average BLEU: {avg_bleu:.2f}")
        print(f"Variance of BLEU: {var_bleu}")
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
        drop_last=False,
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
        embed_dim=1024,
        enc_hidden_dim=2048,
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

    # Schedule для teacher forcing
    teacher_forcing_schedule = torch.linspace(START_TEACHER_FORCING, END_TEACHER_FORCING, steps=NUM_EPOCHS).tolist()

    # Восстановление последней контрольной точки
    checkpoint_files = list(PARAMS_DIR.glob("model_epoch_*.pth"))
    if checkpoint_files:
        def extract_epoch(checkpoint_path: Path):
            return int(checkpoint_path.stem.split("_")[-1])

        checkpoint_files.sort(key=extract_epoch)
        latest_checkpoint = checkpoint_files[-1]
        latest_epoch = extract_epoch(latest_checkpoint)
        print(f"Найден чекпоинт {latest_checkpoint}, возобновляем обучение с эпохи {latest_epoch + 1}")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE, weights_only=True))
        start_epoch = latest_epoch + 1
    else:
        print("Чекпоинты не найдены, начинаем обучение с нуля.")
        start_epoch = 1

    # ----------------- ОБУЧЕНИЕ ----------------- #
    predict(model, val_loader, num_batches=1, compute_bleu_metric=True)
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if epoch < RL_START_EPOCH:
            # Supervised обучение с teacher forcing
            tf_ratio = teacher_forcing_schedule[epoch - 1]
            print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS}, teacher_forcing_ratio={tf_ratio:.2f} (Supervised) ===")
            avg_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                epoch,
                teacher_forcing_ratio=tf_ratio
            )
        else:
            # RL обучение с REINFORCE
            print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} (REINFORCE) ===")
            avg_loss = train_one_epoch_rl(
                model,
                train_loader,
                optimizer,
                scaler,
                epoch
            )

        print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")

        # Пример инференса
        print("--- Пример инференса (5 батчей) ---")
        predict(model, val_loader, num_batches=5, compute_bleu_metric=True)

        # Сохранение чекпоинта
        checkpoint_path = PARAMS_DIR / f"model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Чекпоинт сохранён: {checkpoint_path}")

    print("Training done!")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()