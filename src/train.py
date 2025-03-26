import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from pathlib import Path
from torch.amp import autocast, GradScaler

from model import ImageToLatexModel
from data.dataloader import DataGen, dynamic_collate_fn
from metrics.bleu_score import compute_bleu

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# Функция для преобразования индексов в LaTeX
def indices_to_latex(sequence):
    annotation = [chr(idx) if idx > 2 else '' for idx in sequence]
    return annotation


# ------------------------- ПАРАМЕТРЫ --------------------------------- #
SAMPLES_DIR = Path("/kaggle/working/samples/ImgToLatex-Kaggle-Learning/samples")
DATA_BASE_DIR = SAMPLES_DIR / "images" / "formula_images_processed"
TRAIN_DATA_PATH = SAMPLES_DIR / "im2latex_train_filter.lst"
TRAIN_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
VAL_DATA_PATH = SAMPLES_DIR / "im2latex_validate_filter.lst"
VAL_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"
TEST_DATA_PATH = SAMPLES_DIR / "im2latex_test_filter.lst"
TEST_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"

CHECKPOINT_DIR = Path("/kaggle/input/model-params")
OUTPUT_DIR = Path("/kaggle/working/checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = OUTPUT_DIR / "image_to_latex_model.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 3e-5
BEAM_WIDTH = 5
RL_START_EPOCH = 100
VOCAB_SIZE = 131
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MAX_LENGTH = 70


# ---------------------- ОБУЧЕНИЕ ОДНОЙ ЭПОХИ (Supervised) ----------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, rank):
    model.train()
    total_loss = 0.0
    for step, (images, targets, _) in enumerate(dataloader):
        images, targets = images.to(rank), targets.to(rank)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(images, tgt_tokens=targets)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                targets[:, 1:].contiguous().view(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if (step + 1) % 500 == 0 and rank == 0:
            pred_tokens = torch.argmax(logits, dim=-1)
            gen_sequence = indices_to_latex(pred_tokens[0, :].tolist())
            print(f"[New Epoch {epoch}] Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.8f}")
            print("Generated sequence:", gen_sequence)
        del images, targets, logits, loss
        torch.cuda.empty_cache()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ------------------ ИНФЕРЕНС (PREDICT) ------------------- #
def predict(model, dataloader, num_batches=1, compute_bleu_metric=True, rank=0):
    model.eval()
    all_bleu = []
    batches_processed = 0
    with torch.no_grad():
        for images, targets, img_paths in dataloader:
            images, targets = images.to(rank), targets.to(rank)
            with autocast(device_type="cuda"):
                outputs = model(images, tgt_tokens=None)
                if isinstance(outputs, tuple):
                    _, generated_tokens = outputs
                else:
                    generated_tokens = outputs
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
                if rank == 0:
                    print(f"=== Sample {i + 1} ===")
                    print(f"  Path : {img_paths[i]}")
                    print(f"  Real : {''.join(ref_tokens)}")
                    print(f"  Pred : {''.join(cand_tokens)}")
                    print(f"BLEU : {bleu_score:.2f}" if bleu_score is not None else "BLEU: N/A")
            del images, targets, generated_tokens
            torch.cuda.empty_cache()
            batches_processed += 1
            if batches_processed >= num_batches:
                break
    if compute_bleu_metric and all_bleu and rank == 0:
        avg_bleu = sum(all_bleu) / len(all_bleu)
        print(f"Average BLEU: {avg_bleu:.2f}")
    elif compute_bleu_metric and rank == 0:
        print("No BLEU scores computed.")


# -------------------- MAIN --------------------- #
def main(rank, world_size):
    ddp_setup(rank, world_size)
    print(f"Running on rank {rank}")

    # Создаём датасеты
    train_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=TRAIN_DATA_PATH,
        label_path=TRAIN_LABEL_PATH,
        max_decoder_l=MAX_LENGTH
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=dynamic_collate_fn,
        drop_last=True,
        num_workers=2,
        pin_memory=True
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

    print(f"Creating model on rank {rank} with enc_hidden_dim=768...")
    model = ImageToLatexModel(
        vocab_size=VOCAB_SIZE,
        enc_hidden_dim=768,  # Уменьшено с 1536 до 768
        pad_idx=PAD_IDX,
        sos_index=SOS_IDX,
        eos_index=EOS_IDX,
        max_length=MAX_LENGTH,
        beam_width=BEAM_WIDTH
    ).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Загрузка и обрезка чекпоинта
    checkpoint_files = list(CHECKPOINT_DIR.glob("model_epoch_*.pth"))
    if checkpoint_files:
        def extract_epoch(checkpoint_path: Path):
            return int(checkpoint_path.stem.split("_")[-1])

        checkpoint_files.sort(key=extract_epoch)
        latest_checkpoint = checkpoint_files[-1]
        latest_epoch = extract_epoch(latest_checkpoint)
        if rank == 0:
            print(f"Найден чекпоинт {latest_checkpoint}, возобновляем с эпохи {latest_epoch + 1}")

        # Загружаем старый state_dict
        old_state_dict = torch.load(latest_checkpoint, map_location="cpu")

        # Адаптируем веса для новой модели
        new_state_dict = model.module.state_dict()
        adapted_state_dict = {}
        for key in old_state_dict.keys():
            old_param = old_state_dict[key]
            new_param_shape = new_state_dict[key].shape
            if old_param.shape == new_param_shape:
                adapted_state_dict[key] = old_param
            else:
                if rank == 0:
                    print(f"Обрезаем {key}: {old_param.shape} -> {new_param_shape}")
                if len(old_param.shape) == 1:  # Bias
                    adapted_state_dict[key] = old_param[:new_param_shape[0]]
                elif len(old_param.shape) == 2:  # Linear/Embedding weights
                    adapted_state_dict[key] = old_param[:new_param_shape[0], :new_param_shape[1]]
                elif len(old_param.shape) == 4:  # CNN weights
                    adapted_state_dict[key] = old_param[:new_param_shape[0], :new_param_shape[1], :new_param_shape[2],
                                              :new_param_shape[3]]

        # Загружаем обрезанные веса в модель
        model.module.load_state_dict(adapted_state_dict)
        if rank == 0:
            print(f"Веса из {latest_checkpoint} обрезаны и загружены в модель с enc_hidden_dim=768")
        start_epoch = latest_epoch + 1

        if rank == 0:
            trimmed_checkpoint = OUTPUT_DIR / f"model_epoch_{latest_epoch}_trimmed.pth"
            torch.save(adapted_state_dict, trimmed_checkpoint)
            print(f"Обрезанный чекпоинт сохранён: {trimmed_checkpoint}")
    else:
        if rank == 0:
            print("Чекпоинты не найдены, начинаем с нуля.")
        start_epoch = 1

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler()

    # Обучение
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        if epoch < RL_START_EPOCH:
            if rank == 0:
                print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} (Supervised) ===")
            avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, rank)
            if rank == 0:
                print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.8f}")
        if rank == 0:
            print("--- Пример инференса (5 батчей) ---")
            predict(model, val_loader, num_batches=5, compute_bleu_metric=False, rank=rank)

        # Сохранение чекпоинта
        if rank == 0:
            checkpoint_path = OUTPUT_DIR / f"model_epoch_{epoch}.pth"
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Чекпоинт сохранён: {checkpoint_path}")

    if rank == 0:
        print("Training done!")
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs found. This code requires CUDA support.")
    print(f"World size: {world_size}")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)