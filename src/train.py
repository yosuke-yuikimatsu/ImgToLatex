import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from torch.amp import autocast, GradScaler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from model import ImageToLatexModel
from data.dataloader import DataGen, dynamic_collate_fn
from metrics.bleu_score import compute_bleu

# Функция для преобразования индексов в LaTeX
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
TEST_LABEL_PATH = SAMPLES_DIR / "im2latex_aformulas.tok.lst"

CHECKPOINT_DIR = Path("/kaggle/input/model-params")
OUTPUT_DIR = Path("/kaggle/working/checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = OUTPUT_DIR / "image_to_latex_model.pth"

BATCH_SIZE = 8  # Увеличиваем для TPU благодаря 128 ГБ памяти
NUM_EPOCHS = 100
LEARNING_RATE = 3e-5
BEAM_WIDTH = 5
RL_START_EPOCH = 80
VOCAB_SIZE = 131
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MAX_LENGTH = 70
GRADIENT_ACCUMULATION_STEPS = 4  # Накопление градиентов

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ (Supervised) ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, rank):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    train_loader = pl.MpDeviceLoader(dataloader, xm.xla_device())  # Оборачиваем для TPU
    for step, (images, targets, _) in enumerate(train_loader):
        with autocast():  # autocast работает с TPU
            logits = model(images, tgt_tokens=targets)
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                targets[:, 1:].contiguous().view(-1)
            ) / GRADIENT_ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        total_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            xm.optimizer_step(optimizer)  # Синхронизация градиентов на TPU
            optimizer.zero_grad()

        if (step + 1) % 500 == 0 and rank == 0:
            pred_tokens = torch.argmax(logits, dim=-1)
            gen_sequence = indices_to_latex(pred_tokens[0, :].tolist())
            xm.master_print(f"[Epoch {epoch}] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.8f}")
            xm.master_print("Generated sequence:", gen_sequence)
        del images, targets, logits, loss
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ (REINFORCE) ----------------- #
def train_one_epoch_rl(model, dataloader, optimizer, scaler, epoch, rank):
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    num_batches = 0
    optimizer.zero_grad()
    train_loader = pl.MpDeviceLoader(dataloader, xm.xla_device())  # Оборачиваем для TPU
    for step, (images, _, _) in enumerate(train_loader):
        with autocast():
            predicted_tokens, rewards, loss = model(images, tgt_tokens=None, train=True)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        total_loss += loss.item()
        total_reward += torch.mean(rewards).item() if rewards is not None else 0.0
        num_batches += 1

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            xm.optimizer_step(optimizer)  # Синхронизация на TPU
            optimizer.zero_grad()

        if (step + 1) % 500 == 0 and rank == 0:
            avg_reward = torch.mean(rewards).item() if rewards is not None else 0.0
            gen_sequence = indices_to_latex(predicted_tokens[0].tolist())
            xm.master_print(
                f"[Epoch {epoch} RL] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.8f}, Avg Reward: {avg_reward:.4f}")
            xm.master_print("Generated sequence:", gen_sequence)
        del images, predicted_tokens, rewards, loss
    avg_loss = total_loss / num_batches
    avg_reward = total_reward / num_batches
    return avg_loss, avg_reward

# ------------------ ИНФЕРЕНС (PREDICT) ------------------- #
def predict(model, dataloader, num_batches=1, compute_bleu_metric=True, rank=0):
    model.eval()
    all_bleu = []
    batches_processed = 0
    val_loader = pl.MpDeviceLoader(dataloader, xm.xla_device())  # Оборачиваем для TPU
    with torch.no_grad():
        for images, targets, img_paths in val_loader:
            with autocast():
                logits, generated_tokens = model(images, tgt_tokens=None)
            targets = targets.cpu()
            for i in range(len(images)):
                ref_tokens = indices_to_latex(targets[i, 1:].tolist())
                cand_tokens = indices_to_latex(generated_tokens[i][1:].tolist())
                if compute_bleu_metric:
                    bleu_score = compute_bleu(cand_tokens, [ref_tokens])
                    all_bleu.append(bleu_score)
                else:
                    xm.master_print("BLEU вычисление отключено.")
                    bleu_score = None
                if rank == 0:
                    xm.master_print(f"=== Sample {i + 1} ===")
                    xm.master_print(f"  Path : {img_paths[i]}")
                    xm.master_print(f"  Real : {''.join(ref_tokens)}")
                    xm.master_print(f"  Pred : {''.join(cand_tokens)}")
                    xm.master_print(f"BLEU : {bleu_score:.2f}" if bleu_score is not None else "BLEU: N/A")
            del images, targets, logits, generated_tokens
            batches_processed += 1
            if batches_processed >= num_batches:
                break
    if compute_bleu_metric and all_bleu and rank == 0:
        avg_bleu = sum(all_bleu) / len(all_bleu)
        xm.master_print(f"Average BLEU: {avg_bleu:.2f}")
    elif compute_bleu_metric and rank == 0:
        xm.master_print("No BLEU scores computed.")

# -------------------- MAIN --------------------- #
def main(rank, world_size):
    # Устройство TPU для текущего процесса
    DEVICE = xm.xla_device()
    xm.master_print(f"Running on rank {rank}, Device: {DEVICE}")

    # Создаём датасеты
    train_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=TRAIN_DATA_PATH,
        label_path=TRAIN_LABEL_PATH,
        max_decoder_l=MAX_LENGTH
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=dynamic_collate_fn,
        drop_last=True,
        num_workers=2,
        pin_memory=False  # Не нужен для TPU
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
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        drop_last=False,
        num_workers=2,
        pin_memory=False
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
        pin_memory=False
    )

    # Создаём модель
    xm.master_print(f"Creating model on rank {rank}...")
    model = ImageToLatexModel(
        vocab_size=VOCAB_SIZE,
        enc_hidden_dim=1536,
        pad_idx=PAD_IDX,
        sos_index=SOS_IDX,
        eos_index=EOS_IDX,
        max_length=MAX_LENGTH,
        beam_width=BEAM_WIDTH
    ).to(DEVICE)

    # Загрузка чекпоинта
    checkpoint_files = list(CHECKPOINT_DIR.glob("model_epoch_*.pth"))
    if checkpoint_files:
        def extract_epoch(checkpoint_path: Path):
            return int(checkpoint_path.stem.split("_")[-1])
        checkpoint_files.sort(key=extract_epoch)
        latest_checkpoint = checkpoint_files[-1]
        latest_epoch = extract_epoch(latest_checkpoint)
        xm.master_print(f"Найден чекпоинт {latest_checkpoint}, возобновляем с эпохи {latest_epoch + 1}")
        state_dict = torch.load(latest_checkpoint, map_location='cpu')  # Загружаем на CPU
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        start_epoch = latest_epoch + 1
    else:
        xm.master_print("Чекпоинты не найдены, начинаем с нуля.")
        model = model.to(DEVICE)
        start_epoch = 1

    # Проверка устройства
    xm.master_print(f"Устройство модели на rank {rank}: {next(model.parameters()).device}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler()

    # Обучение
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)  # Обновляем sampler для каждой эпохи
        if epoch < RL_START_EPOCH:
            if rank == 0:
                xm.master_print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} (Supervised) ===")
            avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, rank)
            if rank == 0:
                xm.master_print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.8f}")
        else:
            if rank == 0:
                xm.master_print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} (REINFORCE) ===")
            avg_loss, avg_reward = train_one_epoch_rl(model, train_loader, optimizer, scaler, epoch, rank)
            if rank == 0:
                xm.master_print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.8f}, Avg Reward: {avg_reward:.4f}")

        if rank == 0:
            xm.master_print("--- Пример инференса (5 батчей) ---")
            predict(model, val_loader, num_batches=5, compute_bleu_metric=False, rank=rank)

        # Сохранение чекпоинта
        if rank == 0:
            checkpoint_path = OUTPUT_DIR / f"model_epoch_{epoch}.pth"
            xm.save(model.state_dict(), checkpoint_path)
            xm.master_print(f"Чекпоинт сохранён: {checkpoint_path}")

    if rank == 0:
        xm.master_print("Training done!")
        xm.save(model.state_dict(), MODEL_SAVE_PATH)
        xm.master_print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    world_size = 8  # TPU v3-8 имеет 8 ядер
    xmp.spawn(main, args=(world_size,), nprocs=world_size, start_method='fork')