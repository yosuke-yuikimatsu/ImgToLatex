import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import json


def load_labels(label_path):
    with open(label_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


def load_token_dict(token_dict_path):
    with open(token_dict_path, 'r', encoding='utf-8') as file:
        return json.load(file)["token_to_id"]


class DataGen(Dataset):
    def __init__(self, data_base_dir, data_path, label_path, token_dict_path, max_decoder_l=20):
        self.data_base_dir = Path(data_base_dir)
        self.max_decoder_l = max_decoder_l
        self.labels = load_labels(label_path)
        self.token_to_id = load_token_dict(token_dict_path)

        with open(data_path, 'r', encoding='utf-8') as file:
            self.samples = [line.strip().split() for line in file.readlines()]

        self.PAD = self.token_to_id.get("<PAD>", 0)
        self.GO = self.token_to_id.get("<SOS>", 1)
        self.EOS = self.token_to_id.get("<EOS>", 2)
        self.UNK = self.token_to_id.get("<UNK>", 691)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_index = self.samples[idx]
        img = Image.open(self.data_base_dir / img_path)
        label = self.labels[int(label_index)].split()
        label_tensor = torch.tensor(self._label_to_numeric(label), dtype=torch.long)
        return img, label_tensor, img_path

    def _label_to_numeric(self, label_tokens):
        label_indices = [self.GO] + [self.token_to_id.get(token, self.UNK) for token in label_tokens] + [self.EOS]
        return label_indices[:self.max_decoder_l]

    def indices_to_latex(self, indices):
        id_to_token = {v: k for k, v in self.token_to_id.items()}
        tokens = [id_to_token.get(idx, "<UNK>") for idx in indices if idx not in {self.GO, self.EOS, self.PAD}]
        return ' '.join(tokens)


def dynamic_collate_fn(batch):
    images, targets, img_paths = zip(*batch)
    augmented_images = []
    for i, img in enumerate(images):
        if i < len(images) // 10:
                img = apply_random_transform(img)
        augmented_images.append(img)

    images = [transforms.ToTensor()(img) for img in augmented_images]
    images = torch.stack(images)

    images = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(images)

    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return images, targets, img_paths



def find_empty_columns(img_array, threshold=40):
    column_sums = np.sum(img_array, axis=(0, 2)) if len(img_array.shape) == 3 else np.sum(img_array, axis=0)
    empty_columns = np.where(column_sums < threshold)[0]
    return len(empty_columns)


def apply_random_transform(img):
    transform = random.choice([
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.GaussianBlur(kernel_size=3),
        lambda x: x  # 20% chance to keep original
    ])
    return transform(img)


# if __name__ == "__main__":
#     dataset = DataGen(
#         data_base_dir="/Users/semencinman/Documents/GitHub/ImgToLatex/samples/images",
#         data_path="/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_new_train_filter.lst",
#         label_path="/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_new_formulas.tok.lst",
#         token_dict_path="/Users/semencinman/Documents/GitHub/ImgToLatex/samples/vocab.json"
#     )
#
#     dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,
#         collate_fn=dynamic_collate_fn,
#         num_workers=4
#     )
#
#     for batch_idx, (images, targets, paths) in enumerate(dataloader):
#         print(f"\nBatch {batch_idx + 1} size: {images.shape}")
#         plt.figure(figsize=(15, 5))
#         for i in range(3):
#             img = images[i].permute(1, 2, 0).numpy()
#             img = (img * 0.5 + 0.5).clip(0, 1)
#             target_indices = targets[i].tolist()
#             annotation = dataset.indices_to_latex(target_indices)
#             plt.subplot(1, 3, i + 1)
#             plt.imshow(img)
#             plt.title(f"Path: {paths[i]}\nAnnotation: {annotation}", fontsize=8)
#             plt.axis('off')
#         plt.tight_layout()
#         plt.show()
#         if batch_idx == 0:
#             break