import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from config import Config
'''
这个文件是数据加载模块（数据加载器）
主类是CIFAR10DataModule，继承自LightningDataModule，同时实现了数据增强函数。
1. 根据数据的流动，第一步实现了prepare_data函数
2. 分别定义了train val test三种transform流程，其中train启用了数据增强
3. 定义setup函数设置数据初始化分割，将数据从data中转移到xxx_dataset中
4. 实现三个dataloader，方便数据传递到模型以供训练
'''
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        
        # 传入超参数
        self.config = config
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = True
        
        self.train_transform = self.get_train_transform()
        self.val_transform = self.get_val_transform()
        self.test_transform = self.get_test_transform()
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    # 数据准备
    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
        
    # 将数据从data中分别提取到不同的数据集中
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR10(
                self.data_dir, 
                train=True, 
                transform=self.train_transform
            )
            # 分割训练集和验证集 (45000:5000)
            train_size = 45000
            val_size = 5000
            self.train_dataset, self.val_dataset = random_split(
                cifar_full, [train_size, val_size]
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                self.data_dir, 
                train=False, 
                transform=self.test_transform
            )
    
    # 定义三种transform方式，对于train，实现数据增强
    def get_train_transform(self):
        transform_list = []
        if self.config.augmentation:
            if self.config.random_crop:
                transform_list.append(transforms.RandomCrop(self.config.image_size, padding=4))
            if self.config.random_horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform_list.append(transforms.RandomRotation(15))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        if self.config.cutout and self.config.augmentation:
            transform_list.append(Cutout(self.config.cutout_size))
            
        return transforms.Compose(transform_list)
    
    def get_val_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    def get_test_transform(self):
        return self.get_val_transform()
    
    # 实现三个dataloader，方便数据传递到模型以供训练
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("数据集未初始化")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("数据集未初始化")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("数据集未初始化")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )


# Cutout 数据增强
class Cutout:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)
        
        mask[y1:y2, x1:x2] = 0
        mask = mask.expand_as(img)
        img = img * mask
        return img