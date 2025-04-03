import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import math
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
    def __init__(self, data_base_dir, data_path, label_path, token_dict_path,
                 max_aspect_ratio=4.0, max_decoder_l=20, img_width_range=(12, 320)):
        self.data_base_dir = Path(data_base_dir)
        self.max_aspect_ratio = max_aspect_ratio
        self.max_decoder_l = max_decoder_l
        self.img_width_range = img_width_range
        self.labels = load_labels(label_path)
        self.token_to_id = load_token_dict(token_dict_path)

        with open(data_path, 'r', encoding='utf-8') as file:
            self.samples = [line.strip().split() for line in file.readlines()]

        self.bucket_specs = [
            (int(64 / 4), 9 + 2),
            (int(108 / 4), 15 + 2),
            (int(140 / 4), 17 + 2),
            (int(256 / 4), 20 + 2),
            (int(math.ceil(img_width_range[1] / 4)), max_decoder_l + 2)
        ]
        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.bucket_data = {i: [] for i in range(self.bucket_max_width + 1)}

        self.PAD = self.token_to_id.get("<PAD>",0)
        self.GO = self.token_to_id.get("<SOS>", 1)
        self.EOS = self.token_to_id.get("<EOS>", 2)
        self.UNK = self.token_to_id.get("<UNK>", 691)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_index = self.samples[idx]
        img = Image.open(self.data_base_dir / img_path)
        # Разбиваем метку по пробелам
        label = self.labels[int(label_index)].split()
        label_tensor = torch.tensor(self._label_to_numeric(label), dtype=torch.long)

        width = img.size[0]
        bucket_idx = min(width, self.bucket_max_width)
        self.bucket_data[bucket_idx].append((img, label_tensor, img_path))

        return img, label_tensor, img_path

    def _label_to_numeric(self, label_tokens):
        label_indices = [self.GO] + [self.token_to_id.get(token, self.UNK) for token in label_tokens] + [self.EOS]
        return label_indices[:self.max_decoder_l]

    def indices_to_latex(self, indices):
        id_to_token = {v: k for k, v in self.token_to_id.items()}
        tokens = [id_to_token.get(idx, "<UNK>") for idx in indices if idx not in {self.GO, self.EOS,self.PAD}]
        return ' '.join(tokens)

    def get_buckets(self, batch_size):
        for bucket_idx, bucket in self.bucket_data.items():
            random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                batch_data = bucket[i:i + batch_size]
                images, labels, img_paths = zip(*batch_data)

                max_height = max(img.size[1] for img in images)
                max_width = max(img.size[0] for img in images)

                images = [pad_and_transform(img, max_width, max_height) for img in images]
                images = torch.stack(images, 0)
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

                yield images, labels, img_paths


def find_empty_columns(img_array, threshold=40):
    column_sums = np.sum(img_array, axis=(0, 2)) if len(img_array.shape) == 3 else np.sum(img_array, axis=0)
    empty_columns = np.where(column_sums < threshold)[
        0]  # выбираем столбцы, векторная 1-норма которых достаточно близка к нулю
    return len(empty_columns)


def cyclic_shift_image(img):
    img_array = np.array(img)
    empty_columns = find_empty_columns(img_array)

    if empty_columns > 0:
        shift = np.random.randint(empty_columns // 2, empty_columns + 1)  # сдвигаем по околонулевым столбцам
        shifted_img_array = np.roll(img_array, shift, axis=1)
        return Image.fromarray(shifted_img_array)
    else:
        return img


def apply_random_transform(img):
    transform_list = [
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
        transforms.RandomAutocontrast(p=1.0),
        transforms.RandomEqualize(p=1.0),
        transforms.RandomPosterize(bits=4, p=1.0),
        transforms.RandomSolarize(threshold=128, p=1.0),
        transforms.GaussianBlur(kernel_size=3)
    ]
    transform = random.choice(transform_list)
    return transform(img)


def adjust_color_balance(img):
    img_array = np.array(img)
    for i in range(3):  # Для каждого канала (R, G, B)
        img_array[:, :, i] = np.clip(img_array[:, :, i] * random.uniform(0.9, 1.1), 0, 255)
    return Image.fromarray(img_array)


def dynamic_collate_fn(batch):
    batch.sort(key=lambda x: x[0].size[0] / x[0].size[1], reverse=True)

    images, targets, img_paths = zip(*batch)

    num_images = len(images)
    num_to_augment = num_images // 2
    indices_to_augment = random.sample(range(num_images), num_to_augment)

    augmented_images = []
    for i, img in enumerate(images):
        if i in indices_to_augment:
            if random.random() < 0.5:
                augmented_images.append(cyclic_shift_image(img))
            else:
                augmented_images.append(apply_random_transform(img))
        else:
            augmented_images.append(img)

    max_height = max(img.size[1] for img in augmented_images)
    max_width = max(img.size[0] for img in augmented_images)

    if max_width / max_height > 4.0:
        max_width = int(max_height * 4.0)

    images = [pad_and_transform(img, max_width, max_height) for img in augmented_images]
    images = torch.stack(images, 0)

    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return images, targets, img_paths


def pad_and_transform(image, target_width, target_height):
    width, height = image.size
    padding = [0, 0, target_width - width, target_height - height]
    padded_image = transforms.functional.pad(image, padding, fill=255)
    return transforms.ToTensor()(padded_image)


# def test_augmentation():
#     img_path = "/Users/semencinman/Downloads/IM2LATEX-100K/formula_images_processed/formula_images_processed/1a0a5ac59d.png"  # Укажите путь к тестовому изображению
#     img = Image.open(img_path)
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#     axes[0, 0].imshow(img, cmap='gray')
#     axes[0, 0].set_title("Original Image")
#     axes[0, 0].axis('off')
#     shifted_img = cyclic_shift_image(img)
#     axes[0, 1].imshow(shifted_img, cmap='gray')
#     axes[0, 1].set_title("Cyclic Shift")
#     axes[0, 1].axis('off')
#     filtered_img = apply_random_transform(img)
#     axes[1, 0].imshow(filtered_img, cmap='gray')
#     axes[1, 0].set_title("Random Filter")
#     axes[1, 0].axis('off')
#     plt.show()


# def test_loader():
#
#     dataset = DataGen(
#         data_base_dir='/Users/semencinman/Documents/GitHub/ImgToLatex/samples/images/formula_images_processed',
#         data_path='/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_train_filter.lst',
#         label_path='/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_formulas.tok.lst'
#     )
#
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dynamic_collate_fn)
#
#     for images, targets, img_paths in dataloader:
#         print("Batch loaded with image paths and annotations:")
#         for img_path, target in zip(img_paths, targets):
#             annotation = ''.join([chr(idx) if idx > 2 else '' for idx in target.tolist()])\n            print(f\"Image path: {img_path}, Annotation: {annotation}\")\n         break


""" if __name__ == "__main__":
    token_dict_path = "/Users/semencinman/Downloads/latex_vocab.json"
    data_base_dir = "/Users/semencinman/Documents/GitHub/ImgToLatex/samples/images/formula_images_processed"
    data_path = "/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_train_filter.lst"
    label_path = "/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_formulas.tok.lst"

    # токены разделены пробелом
    test_label = "\\int _ { 0 } ^ { \\infty } e ^ { - x ^ 2 } d x".split()

    datagen = DataGen(data_base_dir, data_path, label_path, token_dict_path)
    numeric_representation = datagen._label_to_numeric(test_label)
    print("Numeric Representation:", numeric_representation)

    latex_representation = datagen.indices_to_latex(numeric_representation)
    print("Reconstructed LaTeX:", latex_representation) """
