import os
import cv2
import numpy as np

# Функция загрузки изображений
def load_images(path):
    images = []
    for file_name in os.listdir(path):
        full_path = os.path.join(path, file_name)
        img = cv2.imread(full_path)
        if img is not None:
            images.append(img)
    return images

# Предобработка данных
def preprocess_images(low_images, normal_images):
    # Нормализация пикселей
    low_normalized = [img / 255.0 for img in low_images]
    normal_normalized = [img / 255.0 for img in normal_images]
    return np.array(low_normalized), np.array(normal_normalized)

# Реализация Retinex
def single_scale_retinex(img, sigma):
    img = img.astype(np.float32) / 255.0  # Нормализуем изображение
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)  # Размытие
    retinex = np.log1p(img) - np.log1p(blurred)  # Вычисляем Retinex
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)  # Нормализуем результат
    return retinex.astype(np.uint8)

def multi_scale_retinex(img, sigmas):
    img = img.astype(np.float32) / 255.0  # Нормализация
    retinex = np.zeros_like(img)
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)  # Размытие
        retinex += np.log1p(img) - np.log1p(blurred)  # Логарифмическая разница
    retinex /= len(sigmas)  # Усреднение результатов
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)  # Нормализация
    return retinex.astype(np.uint8)


# Загрузка изображений
low_images = load_images("LOLdataset/eval/low")
normal_images = load_images("LOLdataset/eval/normal")

# Проверка данных
assert len(low_images) == len(normal_images), "Количество изображений в наборах не совпадает!"

# Предобработка
low_processed, normal_processed = preprocess_images(low_images, normal_images)

print(f"Обработанные данные: низкоосвещенные {low_processed.shape}, нормальные {normal_processed.shape}")

# Применение SSR к изображениям с низкой освещенностью
# enhanced_images = []
# for img in low_processed:
#     enhanced = single_scale_retinex(img, sigma=150)
#     enhanced_images.append(enhanced)


# Применение MSR к изображениям с низкой освещенностью
sigmas = [15, 40, 200]  # Разные значения σ для разных масштабов
enhanced_images = [multi_scale_retinex(img, sigmas) for img in low_processed]


# Сохранение результата
output_dir = "output/enhanced"
os.makedirs(output_dir, exist_ok=True)
for i, enhanced_img in enumerate(enhanced_images):
    cv2.imwrite(os.path.join(output_dir, f"enhanced_{i}.png"), enhanced_img)

print(f"Улучшенные изображения сохранены в папку: {output_dir}")