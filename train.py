import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch

from config import Config
from src.data_module import CIFAR10DataModule
from src.model import ResNet
from src.history_checkpoint import HistoryCheckpoint

'''
在这个文件中定义了启动训练的main函数，依靠着lighting框架，训练被封装在fit中。
1. 将数据加载器初始化
2. 将模型初始化
3. 从模型中将训练器初始化
4. 调用训练器的fit方法执行训练
5. 为训练添加检查点回调和早停回调，这一步的目的是在训练中保存检查点，检测是否过拟合来早停
'''

def main():
    config = Config()
    pl.seed_everything(42, workers=True)
    
    # 模型和数据加载器初始化
    data_module = CIFAR10DataModule(config)
    model = ResNet(config)
    
    # 设置三个回调对象
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="cifar10-{epoch:03d}-{val_accuracy:.3f}",
        monitor=config.checkpoint_monitor,
        mode=config.checkpoint_mode,
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    # HistoryCheckpoint: 每10分钟复制最佳检查点到history目录
    history_callback = HistoryCheckpoint(
        checkpoint_callback=checkpoint_callback,
        interval_minutes=10
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy",
        patience=10000,
        mode="max",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # 设置logger
    tb_logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name="cifar10_cnn"
    )
    
    csv_logger = CSVLogger(
        save_dir=config.log_dir,
        name="cifar10_csv"
    )
    
    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, history_callback],
        deterministic=True, # 决定性算法启动，保障可复现性
        gradient_clip_val=1.0, # 梯度裁剪，防止梯度爆炸
    )
    
    # 启动训练
    trainer.fit(model, data_module)
    
    print("训练完成！")

if __name__ == "__main__":
    main()