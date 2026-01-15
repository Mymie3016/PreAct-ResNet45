import torch

class Config:
    # 数据配置
    data_dir = "./data"
    batch_size = 256
    num_workers = 8
    pin_memory = True
    
    # 图像配置
    image_size = 32
    num_channels = 3
    num_classes = 10
    
    # 数据增强配置
    augmentation = True
    random_crop = True
    random_horizontal_flip = True
    cutout = True
    cutout_size = 8
    
    # 训练配置
    max_epochs = 200
    learning_rate = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    warmup_epochs = 5
    T_max = 200
    
    # 日志配置
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    checkpoint_monitor = "val_accuracy"
    checkpoint_mode = "max"
    
    # 设备配置
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    
config = Config()