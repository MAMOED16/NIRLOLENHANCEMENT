import os
import cv2
import torch
from torch.utils.data import Dataset

class LowLightDataset(Dataset):
    def __init__(self, low_path, normal_path):
        self.low_images = self.load_images(low_path)
        self.normal_images = self.load_images(normal_path)

    def load_images(self, path):
        images = []
        for file_name in os.listdir(path):
            full_path = os.path.join(path, file_name)
            img = cv2.imread(full_path)
            if img is not None:
                img = img.astype('float32') / 255.0  # Нормализация
                images.append(img)
        return images

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low = torch.tensor(self.low_images[idx]).permute(2, 0, 1).float()
        normal = torch.tensor(self.normal_images[idx]).permute(2, 0, 1).float()
        return low, normal
