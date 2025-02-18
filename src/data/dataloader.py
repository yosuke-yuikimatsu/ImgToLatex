import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import math
import random


def load_labels(label_path):
    with open(label_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


class DataGen(Dataset):
    GO = 1
    EOS = 2

    def __init__(self, data_base_dir, data_path, label_path, max_aspect_ratio=4.0, max_decoder_l=20,
                 img_width_range=(12, 320)):
        self.data_base_dir = Path(data_base_dir)
        self.max_aspect_ratio = max_aspect_ratio
        self.max_decoder_l = max_decoder_l
        self.img_width_range = img_width_range
        self.labels = load_labels(label_path)
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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_index = self.samples[idx]
        img = Image.open(self.data_base_dir / img_path)
        label = self.labels[int(label_index)]
        label_tensor = torch.tensor(self._label_to_numeric(label), dtype=torch.long)

        width = img.size[0]

        bucket_idx = min(width, self.bucket_max_width)
        self.bucket_data[bucket_idx].append((img, label_tensor, img_path))

        return img, label_tensor, img_path

    def _label_to_numeric(self, label):
        label_indices = [self.GO] + [ord(c) for c in label] + [self.EOS]
        return label_indices[:self.max_decoder_l]  # Усечение длинных меток

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

def indices_to_latex(indices):
    cleaned_indices = [idx for idx in indices if idx != 0]  # Убираем нули
    return ''.join([chr(idx) for idx in cleaned_indices])


def dynamic_collate_fn(batch):
    batch.sort(key=lambda x: x[0].size[0] / x[0].size[1], reverse=True)

    images, targets, img_paths = zip(*batch)

    max_height = max(img.size[1] for img in images)
    max_width = max(img.size[0] for img in images)

    if max_width / max_height > 4.0:
        max_width = int(max_height * 4.0)

    images = [pad_and_transform(img, max_width, max_height) for img in images]
    images = torch.stack(images, 0)

    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return images, targets, img_paths


def pad_and_transform(image, target_width, target_height):
    width, height = image.size
    padding = [0, 0, target_width - width, target_height - height]
    padded_image = transforms.functional.pad(image, padding, fill=255)
    return transforms.ToTensor()(padded_image)


def test_loader():

    dataset = DataGen(
        data_base_dir='/Users/semencinman/Documents/GitHub/ImgToLatex/samples/images/formula_images_processed',
        data_path='/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_train_filter.lst',
        label_path='/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_formulas.tok.lst'
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dynamic_collate_fn)

    for images, targets, img_paths in dataloader:
        print("Batch loaded with image paths and annotations:")
        for img_path, target in zip(img_paths, targets):
            annotation = ''.join([chr(idx) if idx > 2 else '' for idx in target.tolist()])
            print(f"Image path: {img_path}, Annotation: {annotation}")
        break

if __name__ == "__main__":
    test_loader()


