import torch
import pytorch_lightning as pl
from pytorch_lightning import __version__ # type: ignore

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
print(f"Lightning版本: {__version__}")
print("环境验证通过！可以开始CIFAR-10项目。")