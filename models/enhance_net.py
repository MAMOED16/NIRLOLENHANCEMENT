import torch.nn as nn
import torch

# class EnhanceNet(nn.Module):
#     def __init__(self):
#         super(EnhanceNet, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
#             nn.Sigmoid(),  # Нормализация в диапазон [0, 1]
#         )

#     def forward(self, x):
#         skip = x  # Сохраняем вход
#         x = self.encoder(x)
#         x = self.decoder(x)
        
#         return torch.clamp(0.7 * x + 0.3 * skip, 0, 1)  # Контролируем вклад skip connections



# import torch.nn as nn

# class EnhanceNet(nn.Module):
#     def __init__(self):
#         super(EnhanceNet, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid(),  # Нормализация в диапазон [0, 1]
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x




class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.clamp(x, 0, 1)
