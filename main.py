import os
import torch
import cv2
import argparse
import piq
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from models.decom_net import DecomNet
from models.enhance_net import EnhanceNet
from utils.data_loader import LowLightDataset

# --- Парсер аргументов ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train Decom-Net and Enhance-Net for low-light image enhancement.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for training: 'cuda' or 'cpu' (default: 'cuda').") # --cuda для обучения на GPU
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8).") # --batch_size размер пачек
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3).") # --epochs кол-во эпох
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (default: 1e-4).") # --learning_rate пока что только для enhance_net
    return parser.parse_args()
    
# --- Сглаживание для Enhance-Net ---
def contrast_loss(I_low_enhanced):
    grad_x = torch.abs(I_low_enhanced[:, :, :, :-1] - I_low_enhanced[:, :, :, 1:])
    grad_y = torch.abs(I_low_enhanced[:, :, :-1, :] - I_low_enhanced[:, :, 1:, :])
    avg_gradient = torch.mean(grad_x) + torch.mean(grad_y)
    return -avg_gradient  # Отрицательное значение для стимулирования увеличения контраста

def gradient_loss(I):
    grad_x = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:])
    grad_y = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :])
    return torch.mean(grad_x) + torch.mean(grad_y)


def weighted_mse_loss(output, target, alpha=2.0):
    weight = 1 + alpha * (1 - target)  # Тёмные области получают больший вес
    return torch.mean(weight * (output - target) ** 2)

def range_loss(I_low_enhanced):
    max_value = torch.max(I_low_enhanced)
    min_value = torch.min(I_low_enhanced)
    return -min_value + max_value  # Увеличиваем диапазон, но ограничиваем максимум


# --- Функция потерь для Enhance-Net ---

def enhance_loss(I_low_enhanced, I_normal, alpha=2.0):
    # Веса для тёмных областей
    weight = 1 + alpha * (1 - I_normal)  # Тёмные области получают больший вес
    # Основная MSE потеря с учётом весов
    mse_loss = torch.mean(weight * (I_low_enhanced - I_normal) ** 2)
    # mse_loss = torch.mean((I_low_enhanced - I_normal) ** 2)
    smoothness_loss = gradient_loss(I_low_enhanced)
    ssim_loss = 1 - piq.ssim(I_low_enhanced, I_normal, data_range=1.0) # SSIM для сохранения структуры
    # return mse_loss + 0.05 * ssim_loss + 0.2 * smoothness_loss
    contrast_reg = contrast_loss(I_low_enhanced)  # Регуляризация контраста
    range_reg = torch.min(I_low_enhanced)  # Регуляризация минимальных значений
    range_reg = range_loss(I_low_enhanced)
    print(f"MSE Loss = {mse_loss.item():.6f}, Smoothness Loss = {smoothness_loss.item():.6f}, SSIM Loss = {ssim_loss.item():.6f}, Contrast Reg = {contrast_reg.item():.6f}, Range Reg = {range_reg.item():.6f}")
    return mse_loss + 0.05 * smoothness_loss + 0.1 * contrast_reg + 0.1 * ssim_loss + 0.1 * range_reg



# --- Функция потерь для Decom-Net ---
def decomposition_loss(S_low, S_normal, R_low, I_low, R_normal, I_normal):
    # Реконструкция
    reconstruction_loss = torch.mean((R_low * I_low - S_low) ** 2) + torch.mean((R_normal * I_normal - S_normal) ** 2)

    # Консистентность Reflectance
    reflectance_consistency = torch.mean((R_low - R_normal) ** 2)

    # Гладкость Illumination
    illumination_smoothness = gradient_loss(I_low) + gradient_loss(I_normal)

    # Сохранение текстуры
    texture_preservation = torch.mean((S_low - R_low * I_low) ** 2)

    return (
        reconstruction_loss
        + 0.1 * reflectance_consistency
        + 0.1 * illumination_smoothness
        + 0.1 * texture_preservation
    )

# --- Цикл обучения ---
def train_models(decom_net, enhance_net, train_loader, decom_optimizer, enhance_optimizer, scaler, epochs):
    for epoch in range(epochs):
        total_decom_loss = 0
        total_enhance_loss = 0

        for batch_idx, (S_low, S_normal) in enumerate(train_loader):
            # Перенос данных на устройство
            S_low, S_normal = S_low.to(device), S_normal.to(device)

            # --- Обучение Decom-Net ---
            decom_optimizer.zero_grad()
            with autocast(device_type="cuda"):
                R_low, I_low = decom_net(S_low)
                R_normal, I_normal = decom_net(S_normal)
                decom_loss = decomposition_loss(S_low, S_normal, R_low, I_low, R_normal, I_normal)
            scaler.scale(decom_loss).backward()
            scaler.step(decom_optimizer)

            # --- Обучение Enhance-Net ---
            enhance_optimizer.zero_grad()
            # Detach illumination map to prevent backward through Decom-Net's graph
            I_low_detached = I_low.detach()
            I_normal_detached = I_normal.detach()
            with autocast(device_type="cuda"):
                I_low_enhanced = enhance_net(I_low_detached)
                enhance_loss_value = enhance_loss(I_low_enhanced, I_normal_detached)
            scaler.scale(enhance_loss_value).backward()
            scaler.step(enhance_optimizer)
            scaler.update()

            total_decom_loss += decom_loss.item()
            total_enhance_loss += enhance_loss_value.item()
            print(f"I_low_enhanced min: {I_low_enhanced.min().item()}, max: {I_low_enhanced.max().item()}")

            # Сохранение промежуточных результатов
            # if batch_idx % 1 == 0:
            save_intermediate_layers(R_low, I_low, I_low_enhanced, epoch, batch_idx)

        print(f"Эпоха [{epoch + 1}/{epochs}], DecomNet Потеря: {total_decom_loss / len(train_loader):.6f}, EnhanceNet Потеря: {total_enhance_loss / len(train_loader):.6f}")
        

        # Сохраняем модели после каждой эпохи
        torch.save(decom_net.state_dict(), f"checkpoints/decom_net_epoch_{epoch + 1}.pth")
        torch.save(enhance_net.state_dict(), f"checkpoints/enhance_net_epoch_{epoch + 1}.pth")


# --- Сохранение промежуточных слоев ---
def save_intermediate_layers(R_low, I_low, I_low_enhanced, epoch, batch_idx):
    output_dir = f"output/intermediate/epoch_{epoch + 1}_batch_{batch_idx + 1}"
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем Reflectance, Illumination и Enhanced Illumination
    for i in range(R_low.size(0)):
        reflectance = R_low[i].permute(1, 2, 0).detach().cpu().numpy() * 255
        illumination = I_low[i, 0].detach().cpu().numpy() * 255
        illumination_enhanced = I_low_enhanced[i, 0].detach().cpu().numpy() * 255

        cv2.imwrite(os.path.join(output_dir, f"reflectance_{i}.png"), reflectance.clip(0, 255).astype('uint8'))
        cv2.imwrite(os.path.join(output_dir, f"illumination_{i}.png"), illumination.clip(0, 255).astype('uint8'))
        cv2.imwrite(os.path.join(output_dir, f"illumination_enhanced_{i}.png"), illumination_enhanced.clip(0, 255).astype('uint8'))

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)  # Xavier инициализация
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --- Основной код ---
if __name__ == "__main__":
    # --- Получение параметров ---
    args = parse_args()

    # --- Настройка устройства ---
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Используется устройство: {device}")

    # --- Параметры ---
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    # --- Инициализация моделей ---
    decom_net = DecomNet().to(device)
    enhance_net = EnhanceNet().to(device)
    decom_net.apply(init_weights)
    enhance_net.apply(init_weights)

    # --- Оптимизаторы ---
    decom_optimizer = torch.optim.Adam(decom_net.parameters(), lr=1e-4) # !ЭТО ВРЕМЕННОЕ
    enhance_optimizer = torch.optim.Adam(enhance_net.parameters(), lr=learning_rate)

    # --- Скалер для Mixed Precision Training ---
    scaler = GradScaler()

    # --- Загрузка данных ---
    train_dataset = LowLightDataset("dataset/train_tiny/low", "dataset/train_tiny/normal")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Обучение моделей ---
    train_models(decom_net, enhance_net, train_loader, decom_optimizer, enhance_optimizer, scaler, epochs)
