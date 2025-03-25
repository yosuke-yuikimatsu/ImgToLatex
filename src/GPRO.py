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
from fix import fix
from train import predict

from torch.amp import autocast, GradScaler

import Levenshtein

def indices_to_latex(sequence):
    annotation = [chr(idx) if idx > 2 else '' for idx in sequence]
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
TEST_DATA_PATH = SAMPLES_DIR / "im2latex_test_filter.lst"
TEST_LABEL_PATH = SAMPLES_DIR / "im2latex_formulas.tok.lst"

# ---------------- ПУТЬ ДЛЯ СОХРАНЕНИЯ МОДЕЛИ ------------------------- #
PARAMS_DIR = Path("/content/drive/MyDrive/params_new_model")
os.makedirs(PARAMS_DIR, exist_ok=True)

MODEL_SAVE_PATH = Path.cwd() / "models" / "image_to_latex_model.pth"
os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

# Гиперпараметры
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 3e-5
BEAM_WIDTH = 5

VOCAB_SIZE = 131
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MAX_LENGTH = 70


def compute_reward(cand_sequence, ref_sequence):
    edit_distance = Levenshtein.distance(cand_sequence, ref_sequence)
    reward = 1 / (1 + edit_distance)
    return reward

def train_one_epoch_policy_gradient(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    for images, targets, _ in dataloader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        features = model.cnn(images)
        memory = model.encoder(features)

        generated_tokens, total_log_probs = model.decoder.sample_sequence(memory)

        generated_sequences = [indices_to_latex(seq.cpu().numpy()) for seq in generated_tokens]
        ref_sequences = [indices_to_latex(target[1:].cpu().numpy()) for target in targets]

        rewards = []
        for cand_seq, ref_seq in zip(generated_sequences, ref_sequences):
            reward = compute_reward(cand_seq, ref_seq)
            rewards.append(reward)

        rewards = torch.Tensor(rewards).to(DEVICE)
        loss = - (rewards * total_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss




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

    test_dataset = DataGen(
        data_base_dir=DATA_BASE_DIR,
        data_path=TEST_DATA_PATH,
        label_path=TEST_LABEL_PATH,
        max_decoder_l=MAX_LENGTH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle = False,
        collate_fn=dynamic_collate_fn,
        drop_last=False,
        num_workers=2
    )

    print("Device:", DEVICE)
    print("Creating model...")
    model = ImageToLatexModel(
        vocab_size=VOCAB_SIZE,
        enc_hidden_dim=1536,  # Должно быть кратно количеству голов в энкодере
        pad_idx=PAD_IDX,
        sos_index=SOS_IDX,
        eos_index=EOS_IDX,
        max_length=MAX_LENGTH,
        beam_width = BEAM_WIDTH
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
    

    for epoch in range(start_epoch, 101):
        avg_loss = train_one_epoch_policy_gradient(model, train_loader, optimizer)
        print(f"Policy Gradient Epoch {epoch} done. Avg Loss: {avg_loss:.8f}")
        predict(model, val_loader, num_batches=1, compute_bleu_metric=True)


if (__name__ == "__main__") : 
    main()