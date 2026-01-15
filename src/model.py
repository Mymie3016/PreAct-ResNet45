import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR, LambdaLR, SequentialLR
from torchmetrics import Accuracy
from config import Config

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, isChange = False):
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if isChange else 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 下采样层（如果需要）
        self.downsample = nn.Identity()
        if isChange:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        ind = self.downsample(x)
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out += ind
        
        return out

class ResNet(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # 用于捕获模型的全部属性，方便检查点的保存
        self.save_hyperparameters()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1 = self._make_stage(16, 16, num_blocks=3, stride=1)
        
        self.stage2 = self._make_stage(16, 32, num_blocks=3, stride=2)
        
        self.stage3 = self._make_stage(32, 64, num_blocks=3, stride=2)
        
        self.stage4 = self._make_stage(64, 128, num_blocks=3, stride=2)
        
        self.stage5 = self._make_stage(128, 256, num_blocks=3, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, config.num_classes)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 评估指标
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        
        # 第一个块可能进行下采样
        if in_channels != out_channels:
            layers.append(BasicBlock(in_channels, out_channels, isChange=True))
        else:
            layers.append(BasicBlock(in_channels, out_channels))
        
        # 剩余的块
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):

        optimizer = SGD(
            self.parameters(), 
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Warmup调度器
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / self.config.warmup_epochs
            if epoch < self.config.warmup_epochs else 1.0
        )
        # 余弦退火调度器（在warmup后开始）
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.T_max,
        )
        # 组合调度器：先warmup，后余弦退火
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs]
        )
        return [optimizer], [scheduler]

    
def on_train_epoch_end(self: pl.LightningModule):
    # 每个epoch结束后打印学习率
    optimizer = self.optimizers()
    
    if isinstance(optimizer, list):
        optimizer = optimizer[0]  # 提取列表中的优化器
    current_lr = optimizer.param_groups[0]['lr']
    
    self.log("learning_rate", current_lr, prog_bar=True, on_step=False, on_epoch=True)