import os
import torch
import cv2
from torch.utils.data import DataLoader
from models.decom_net import DecomNet
from models.enhance_net import EnhanceNet
from utils.data_loader import LowLightDataset

# --- Настройка устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# --- Тестирование ---
def test_pipeline(decom_net, enhance_net, test_loader, output_dir="output/test_results"):
    decom_net.eval()
    enhance_net.eval()
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, (S_low, _) in enumerate(test_loader):
        S_low = S_low.to(device)

        with torch.no_grad():
            # Декомпозиция изображения
            R_low, I_low = decom_net(S_low)

            # Улучшение карты освещенности
            I_low_enhanced = enhance_net(I_low)

            # Реконструкция улучшенного изображения
            S_reconstructed = R_low * I_low_enhanced

        # Сохранение результатов
        for i in range(S_low.size(0)):
            input_image = S_low[i].permute(1, 2, 0).cpu().numpy() * 255
            reflectance = R_low[i].permute(1, 2, 0).cpu().numpy() * 255
            illumination = I_low[i, 0].cpu().numpy() * 255
            illumination_enhanced = I_low_enhanced[i, 0].cpu().numpy() * 255
            reconstructed_image = S_reconstructed[i].permute(1, 2, 0).cpu().numpy() * 255

            # Сохраняем каждое изображение
            cv2.imwrite(os.path.join(output_dir, f"input_{batch_idx}_{i}.png"), input_image.clip(0, 255).astype('uint8'))
            cv2.imwrite(os.path.join(output_dir, f"reflectance_{batch_idx}_{i}.png"), reflectance.clip(0, 255).astype('uint8'))
            cv2.imwrite(os.path.join(output_dir, f"illumination_{batch_idx}_{i}.png"), illumination.clip(0, 255).astype('uint8'))
            cv2.imwrite(os.path.join(output_dir, f"illumination_enhanced_{batch_idx}_{i}.png"), illumination_enhanced.clip(0, 255).astype('uint8'))
            cv2.imwrite(os.path.join(output_dir, f"reconstructed_{batch_idx}_{i}.png"), reconstructed_image.clip(0, 255).astype('uint8'))
    
    print(f"Результаты тестирования сохранены в папке {output_dir}")

# --- Основной код ---
if __name__ == "__main__":
    # Параметры
    test_data_path = "dataset/eval/low"  # Путь к тестовым данным
    decom_net_weights = "checkpoints/decom_net_epoch_5.pth"  # чекпоинты Decom-Net
    enhance_net_weights = "checkpoints/enhance_net_epoch_5.pth"  # чекпоинты Enhance-Net

    # Загрузка тестового набора
    test_dataset = LowLightDataset(test_data_path, test_data_path)  # Пары не нужны
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Загрузка моделей
    decom_net = DecomNet().to(device)
    enhance_net = EnhanceNet().to(device)

    # Загрузка весов
    decom_net.load_state_dict(torch.load(decom_net_weights, map_location=device))
    enhance_net.load_state_dict(torch.load(enhance_net_weights, map_location=device))

    # Тестирование
    test_pipeline(decom_net, enhance_net, test_loader)
