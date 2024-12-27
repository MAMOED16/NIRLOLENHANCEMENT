import torch
from torch.utils.data import DataLoader
from models.decom_net import DecomNet
from models.enhance_net import EnhanceNet
from utils.data_loader import LowLightDataset
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()  # Скалирование градиентов


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")
decom_net = DecomNet().to(device)
enhance_net = EnhanceNet().to(device)


def train_decom_net(decom_net, train_loader, optimizer, epochs):
    decom_net.train()
    for epoch in range(epochs):
        total_loss = 0
        for S_low, S_normal in train_loader:
            S_low, S_normal = S_low.to(device), S_normal.to(device)
            optimizer.zero_grad()
            with autocast():  # Включаем Mixed Precision
                R_low, I_low = decom_net(S_low)
                R_normal, I_normal = decom_net(S_normal)
                loss = decomposition_loss(S_low, S_normal, R_low, I_low, R_normal, I_normal)

        # Масштабирование и обратное распространение
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"decom Эпоха [{epoch + 1}/{epochs}], Потеря: {total_loss / len(train_loader)}")

# Функция потерь
def decomposition_loss(S_low, S_normal, R_low, I_low, R_normal, I_normal):
    # Реконструкция
    reconstruction_loss = torch.mean((R_low * I_low - S_low) ** 2) + torch.mean((R_normal * I_normal - S_normal) ** 2)

    # Консистентность Reflectance
    reflectance_consistency = torch.mean((R_low - R_normal) ** 2)

    # Гладкость карты освещенности
    def gradient_loss(I):
        grad_x = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:])
        grad_y = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :])
        return torch.mean(grad_x) + torch.mean(grad_y)
    illumination_smoothness = gradient_loss(I_low) + gradient_loss(I_normal)

    # Сохранение текстуры в Reflectance
    texture_preservation = torch.mean((S_low - R_low * I_low) ** 2)

    # Итоговая потеря
    return (
        reconstruction_loss
        + 0.1 * reflectance_consistency
        + 0.1 * illumination_smoothness
        + 0.1 * texture_preservation
    )



def train_enhance_net(decom_net, enhance_net, train_loader, optimizer, epochs):
    decom_net.eval()  # Замораживаем Decom-Net
    enhance_net.train()

    for epoch in range(epochs):
        total_loss = 0
        for S_low, S_normal in train_loader:
            optimizer.zero_grad()

            # Прямой проход через Decom-Net (карты освещенности)
            with torch.no_grad():
                _, I_low = decom_net(S_low)
                _, I_normal = decom_net(S_normal)

            # Прямой проход через Enhance-Net
            I_low_enhanced = enhance_net(I_low)

            # Потери для Enhance-Net
            # Основная цель — улучшить карту освещенности так, чтобы реконструированное изображение стало ближе к нормальному
            recon_loss = torch.mean((I_low_enhanced - I_normal) ** 2)
            loss = recon_loss

            # Обновление градиентов
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"enhance Эпоха [{epoch + 1}/{epochs}], Потеря: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    # Загрузка данных
    dataset = LowLightDataset("dataset/train/low", "dataset/train/normal")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Инициализация модели и оптимизатора
    decom_net = DecomNet().to(device)
    optimizer = torch.optim.Adam(decom_net.parameters(), lr=1e-4)

    # Обучение модели
    train_decom_net(decom_net, train_loader, optimizer, 3)


if __name__ == "__main__":
    # Загрузка данных
    dataset = LowLightDataset("dataset/train/low", "dataset/train/normal")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Инициализация моделей
    decom_net = DecomNet().to(device)
    enhance_net = EnhanceNet().to(device)

    # Оптимизатор для Enhance-Net
    optimizer = torch.optim.Adam(enhance_net.parameters(), lr=1e-4)

    # Обучение Enhance-Net
    train_enhance_net(decom_net, enhance_net, train_loader, optimizer, 3)

import cv2
import os

def test_pipeline(decom_net, enhance_net, test_loader):
    decom_net.eval()
    enhance_net.eval()
    output_dir = "output/test_results"
    os.makedirs(output_dir, exist_ok=True)

    for S_low, _ in test_loader:
        S_low = S_low.to(device)  # Перемещаем данные на GPU
        with torch.no_grad():
            # Декомпозиция
            R_low, I_low = decom_net(S_low)
            print(f"R_low min: {R_low.min().item()}, max: {R_low.max().item()}")

            # Коррекция освещенности
            I_low_enhanced = enhance_net(I_low)
            print(f"I_low min: {I_low.min().item()}, max: {I_low.max().item()}")
            print(f"I_low_enhanced min: {I_low_enhanced.min().item()}, max: {I_low_enhanced.max().item()}")

            # Реконструкция изображения
            S_reconstructed = R_low * I_low_enhanced

        # Сохранение результатов
        for i in range(S_reconstructed.size(0)):
            result = S_reconstructed[i].permute(1, 2, 0).cpu().numpy() * 255.0
            result = (result * 255.0).clip(0, 255).astype('uint8')
            cv2.imwrite(os.path.join(output_dir, f"reconstructed_{i}.png"), result)
            
            os.makedirs("output/intermediate", exist_ok=True)

            # Сохранение Reflectance
            reflectance = R_low[0].permute(1, 2, 0).cpu().numpy() * 255.0
            reflectance = reflectance.clip(0, 255).astype('uint8')
            cv2.imwrite(os.path.join(output_dir, f"reflectance_{i}.png"), reflectance)

            # Сохранение оригинальной и улучшенной карты освещенности
            illumination = I_low[0, 0].cpu().numpy() * 255.0
            illumination_e = I_low_enhanced[0, 0].cpu().numpy() * 255.0
            cv2.imwrite(os.path.join(output_dir, f"illumination_{i}.png"), illumination)
            cv2.imwrite(os.path.join(output_dir, f"illumination_e_{i}.png"), illumination_e)


    print(f"Результаты тестирования сохранены в {output_dir}")

# Запуск тестирования
if __name__ == "__main__":
    # Тестовые данные
    test_dataset = LowLightDataset("dataset/eval/low", "dataset/eval/normal")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Тестируем модель
    test_pipeline(decom_net, enhance_net, test_loader)
