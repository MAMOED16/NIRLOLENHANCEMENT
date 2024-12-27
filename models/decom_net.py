import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.reflectance = nn.Conv2d(256, 3, kernel_size=3, padding=1)  # Для Reflectance
        self.illumination = nn.Conv2d(256, 1, kernel_size=3, padding=1)  # Для Illumination

    def forward(self, x):
        features = self.encoder(x)
        reflectance = torch.sigmoid(self.reflectance(features))  # Reflectance в диапазоне [0, 1]
        illumination = torch.sigmoid(self.illumination(features))  # Illumination в диапазоне [0, 1]
        return reflectance, illumination
