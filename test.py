import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os

from config import Config
from src.data_module import CIFAR10DataModule
from src.model import ResNet

'''
这个文件是定义了测试，可以使用--checkpoint参数指定使用的checkpoint路径。
1. 调用arg函数获取传入的参数
2. 初始化数据加载器，设置测试模式
3. 加载模型checkpoint，设置eval模式
4. 初始化训练器，取消logger和checkpoint
5. 调用test方法进行测试，得到输出结果，获取准确度和loss
'''

def parse_args():
    parser = argparse.ArgumentParser(description="测试模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载配置
    config = Config()
    checkpoint = args.checkpoint
    
    # 设置随机种子，保证确定性可复现
    pl.seed_everything(42, workers=True)
    
    # 初始化数据加载器
    data_module = CIFAR10DataModule(config)
    data_module.setup(stage="test") # 数据加载器切换成测试模式
    
    # 加载模型
    print(f"加载模型从: {checkpoint}")
    model = ResNet.load_from_checkpoint(checkpoint, config=config, weights_only=False)
    model.eval() # 模型切换成应用模式，不再进行反向传递和loss计算
    
    # 初始化训练器
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        logger=False, #取消logger和检查点保存
        enable_checkpointing=False,
        deterministic=True # 决定性算法启动，保障可复现性
    )
    
    # 测试模型
    print("\n" + "="*50)
    print("开始测试...")
    results = trainer.test(model, data_module)
    
    # 输出结果
    print("\n" + "="*50)
    print("测试结果:")
    print(f"测试准确率: {results[0]['test_accuracy']:.4f}")
    print(f"测试损失: {results[0]['test_loss']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()