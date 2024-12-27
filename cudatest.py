import torch

if torch.cuda.is_available():
    print("GPU доступен:", torch.cuda.get_device_name(0))
else:
    print("GPU не найден. Проверьте установку PyTorch и драйверов CUDA.")
