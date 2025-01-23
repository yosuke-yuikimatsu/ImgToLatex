import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from data.dataloader import DataGen, collate_fn, indices_to_latex

from model import OCRModel

def train_main():
    ## Инициализируем модель и dataloader
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_base_dir = os.path.join(BASE_DIR, "..", "samples", "images", "formula_images_processed")
    data_path = os.path.join(BASE_DIR, "..", "samples", "im2latex_train_filter.lst")
    label_path = os.path.join(BASE_DIR, "..", "samples", "im2latex_formulas.tok.lst")

    # Параметры DataLoader
    batch_size = 4
    num_workers = 0 
    shuffle = True

    train_dataset = DataGen(
        data_base_dir=data_base_dir,
        data_path=data_path,
        label_path=label_path,
        # Можно при желании указать max_aspect_ratio, max_encoder_l_h, max_decoder_l и т.д.
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # Задаём конфиг модели
    config = {
        "encoder_num_hidden": 256,
        "encoder_num_layers": 1,
        "decoder_num_hidden": 256,
        "decoder_num_layers": 1,
        # Предположим, что общее количество символов, которые мы можем встретить, — 256 
        # (так как мы используем ord(c) и паддинг = 0). Можно больше, если нужно.
        "target_vocab_size": 256,
        # Размер embedding для символов
        "target_embedding_size": 128,
        "dropout": 0.0,
        "ignore_index": 0,
    }

    model = OCRModel(config)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Инициализируем оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Обучение 
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        step_count = 0

        for batch_idx, (images, targets, img_paths) in enumerate(train_loader):
            # images.shape = (batch_size, 3, H, W)
            # targets.shape = (batch_size, seq_len) (ord(c), паддинг=0)
            images = images.to(device)
            targets = targets.to(device)

            # 1) Обнуляем градиенты
            optimizer.zero_grad()

            # 2) Forward: model(images, targets) должен вернуть loss
            loss = model(images, targets)

            # 3) Backward
            loss.backward()

            # 4) Обновляем параметры
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

            if (batch_idx+1) % 50 == 0:
                avg_loss = total_loss / step_count
                print(f"[Epoch {epoch+1}/{num_epochs}] Step {batch_idx+1}/{len(train_loader)} "
                      f"Loss = {avg_loss:.4f}")
                # Обнулить, чтоб каждые 50 шагов заново считать среднюю
                total_loss = 0.0
                step_count = 0

        print(f"=== Epoch {epoch+1} done! ===")

    # ===== 7) (Опционально) Пример инференса на одном батче =====
    ##

if __name__ == "__main__":
    train_main()
