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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
print(f"Device: {DEVICE}, Number of GPUs: {NUM_GPUS}")

def indices_to_latex(sequence):
    annotation = [chr(idx) if idx > 2 else '' for idx in sequence]
    return annotation

# ------------------------- ПАРАМЕТРЫ --------------------------------- #
SAMPLES_DIR = Path.cwd() / "samples"
DATA_BASE_DIR = SAMPLES_DIR / "images" / "formula_images_processed"
TRAIN_DATA_PATH = SAMPLES_DIR / "im2latex_train_filter.lst"
TRAIN_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
VAL_DATA_PATH = SAMPLES_DIR / "im2latex_validate_filter.lst"
VAL_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
TEST_DATA_PATH = SAMPLES_DIR / "im2latex_test_filter.lst"
TEST_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"

CHECKPOINT_DIR = Path("/kaggle/input/model-params")
OUTPUT_DIR = Path("/kaggle/working/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = OUTPUT_DIR / "image_to_latex_model.pth"

BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 3e-5
BEAM_WIDTH = 5
RL_START_EPOCH = 80

VOCAB_SIZE = 131
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MAX_LENGTH = 70

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ (Supervised) ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, targets, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
            logits = model(images, tgt_tokens=targets)
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                targets[:, 1:].contiguous().view(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % 500 == 0:
            pred_tokens = torch.argmax(logits, dim=-1)
            gen_sequence = indices_to_latex(pred_tokens[0, :].tolist())
            print("Generated sequence:", gen_sequence)
            print(f"[Epoch {epoch}] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.8f}")

        # Очистка памяти
        del images, targets, logits, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ (REINFORCE) ----------------- #
def train_one_epoch_rl(model, dataloader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    num_batches = 0

    for step, (images, _, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
            predicted_tokens, rewards, loss = model(images, tgt_tokens=None, train=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_reward += torch.mean(rewards).item() if rewards is not None else 0.0
        num_batches += 1

        if (step + 1) % 500 == 0:
            avg_reward = torch.mean(rewards).item() if rewards is not None else 0.0
            gen_sequence = indices_to_latex(predicted_tokens[0].tolist())
            print("Generated sequence:", gen_sequence)
            print(f"[Epoch {epoch} RL] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.8f}, Avg Reward: {avg_reward:.4f}")

        del images, predicted_tokens, rewards, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_reward = total_reward / num_batches
    return avg_loss, avg_reward

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
                          device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
                logits, generated_tokens = model(images, tgt_tokens=None)

            targets = targets.cpu()
            for i in range(len(images)):
                ref_tokens = indices_to_latex(targets[i, 1:].tolist())
                cand_tokens = indices_to_latex(generated_tokens[i][1:].tolist())

                if compute_bleu_metric:
                    bleu_score = compute_bleu(cand_tokens, [ref_tokens])
                    all_bleu.append(bleu_score)
                else:
                    print("BLEU вычисление отключено.")
                    bleu_score = None

                print(f"=== Sample {i + 1} ===")
                print(f"  Path : {img_paths[i]}")
                print(f"  Real : {''.join(ref_tokens)}")
                print(f"  Pred : {''.join(cand_tokens)}")
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
        num_workers=2,
        pin_memory=True  # Ускоряет передачу данных на GPU
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
        num_workers=2,
        pin_memory=True
    )

    test_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=TEST_DATA_PATH,
        label_path=TEST_LABEL_PATH,
        max_decoder_l=MAX_LENGTH
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )

    print("Creating model...")
    model = ImageToLatexModel(
        vocab_size=VOCAB_SIZE,
        enc_hidden_dim=1536,
        pad_idx=PAD_IDX,
        sos_index=SOS_IDX,
        eos_index=EOS_IDX,
        max_length=MAX_LENGTH,
        beam_width=BEAM_WIDTH
    )

    # Перемещаем модель на GPU и используем DataParallel для двух GPU
    model = model.to(DEVICE)
    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs with DataParallel!")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler(device="cuda" if DEVICE.type == "cuda" else "cpu")

    # Восстановление последней контрольной точки
    checkpoint_files = list(CHECKPOINT_DIR.glob("model_epoch_*.pth"))
    if checkpoint_files:
        def extract_epoch(checkpoint_path: Path):
            return int(checkpoint_path.stem.split("_")[-1])

        checkpoint_files.sort(key=extract_epoch)
        latest_checkpoint = checkpoint_files[-1]
        latest_epoch = extract_epoch(latest_checkpoint)
        print(f"Найден чекпоинт {latest_checkpoint}, возобновляем обучение с эпохи {latest_epoch + 1}")
        state_dict = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=True)
        if NUM_GPUS > 1:
            # Убираем префикс 'module.' из DataParallel при загрузке
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        start_epoch = latest_epoch + 1
    else:
        print("Чекпоинты не найдены, начинаем обучение с нуля.")
        start_epoch = 1

    # Обучение
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if epoch < RL_START_EPOCH:
            print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} (Supervised) ===")
            avg_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                epoch
            )
            print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.8f}")
        else:
            print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} (REINFORCE) ===")
            avg_loss, avg_reward = train_one_epoch_rl(
                model,
                train_loader,
                optimizer,
                scaler,
                epoch
            )
            print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.8f}, Avg Reward: {avg_reward:.4f}")

        print("--- Пример инференса (5 батчей) ---")
        predict(model, val_loader, num_batches=5, compute_bleu_metric=False)

        # Сохранение чекпоинта
        checkpoint_path = OUTPUT_DIR / f"model_epoch_{epoch}.pth"
        if NUM_GPUS > 1:
            # Сохраняем state_dict без префикса 'module.' для совместимости
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        print(f"Чекпоинт сохранён: {checkpoint_path}")

    print("Training done!")
    if NUM_GPUS > 1:
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
    else:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()