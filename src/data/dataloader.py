import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
class DataGen(Dataset):
    def __init__(self, data_base_dir, data_path, label_path, max_aspect_ratio=None, max_encoder_l_h=None,
                 max_encoder_l_w=None, max_decoder_l=None):
        self.data_base_dir = data_base_dir
        self.data_path = data_path
        self.label_path = label_path
        self.max_aspect_ratio = max_aspect_ratio or float('inf')
        self.max_encoder_l_h = max_encoder_l_h or float('inf')
        self.max_encoder_l_w = max_encoder_l_w or float('inf')
        self.max_decoder_l = max_decoder_l or float('inf')
        # читаем данные
        with open(self.data_path, 'r') as file:
            self.lines = [line.strip().split() for line in file.readlines()]
        with open(self.label_path, 'r') as file:
            self.latex_lines = file.readlines()
        self.cursor = 0
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        img_path, label_index = self.lines[idx]
        img = Image.open(os.path.join(self.data_base_dir, img_path)).convert("RGB")
        label_str = self.latex_lines[int(label_index)].strip()
        label_list = self._label_to_numeric(label_str) # метки в числа
        return img, label_list, img_path
    def _label_to_numeric(self, label):
        return [ord(c) for c in label]
# Для корректной обработки батчей с изображениями разного размера
def collate_fn(batch):
    images, targets, img_paths = zip(*batch)
    #максимальная высота и ширину среди всех изображений в батче
    max_height = max([img.size[1] for img in images])
    max_width = max([img.size[0] for img in images])
    # Преобразуем изображения в одинаковый размер (через паддинг, может, это плохая затея, вдруг нолики в тензоре плохо скажутся на модели)
    images = [pad_image(img, max_width, max_height) for img in images]
    images = torch.stack(images, 0)  # Ставим изображения в батч
    # Преобразуем метки в тензоры, с паддингом для одинаковой длины
    targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(target) for target in targets], batch_first=True,
                                              padding_value=0)
    return images, targets, img_paths

def pad_image(image, target_width, target_height):
    width, height = image.size
    padding_left = (target_width - width) // 2
    padding_top = (target_height - height) // 2
    padding_right = target_width - width - padding_left
    padding_bottom = target_height - height - padding_top
    # Добавляем паддинг (слева, сверху, справа, снизу)
    padded_image = transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom),
                                             fill=255)
    return transforms.ToTensor()(padded_image)

def indices_to_latex(indices):
    cleaned_indices = [idx for idx in indices if idx != 0]  # Убираем нули
    return ''.join([chr(idx) for idx in cleaned_indices])



# пытаемся эстетично визуализировать
def visualize_batch_with_labels(images, targets, img_paths):
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Переводим в numpy
    batch_size = images.shape[0]
    num_images = min(batch_size, 4)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.set_title(f"Image {i + 1}")
        ax.axis('off')
        label_str = indices_to_latex(targets[i])
        ax.text(0.5, -0.2, label_str, ha='center', va='top', transform=ax.transAxes, fontsize=5, color='black')
    plt.tight_layout()
    plt.show()

def test_loader():
    data_base_dir = '/Users/semencinman/Documents/GitHub/ImgToLatex/samples/images/formula_images_processed'
    data_path = '/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_train_filter.lst'
    label_path = '/Users/semencinman/Documents/GitHub/ImgToLatex/samples/im2latex_formulas.tok.lst'
    dataset = DataGen(data_base_dir, data_path, label_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    for images, targets, img_paths in dataloader:
        visualize_batch_with_labels(images, targets, img_paths)  # Визуализируем батч
        break
if __name__ == "__main__":
    test_loader()