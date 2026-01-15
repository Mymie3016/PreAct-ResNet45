# CIFAR-10 ResNet20 with PyTorch Lightning

基于PyTorch Lightning的CIFAR-10图像分类项目，使用PreAct-ResNet45预激活残差网络架构
经过训练 PreAct-ResNet45-3500-0.931.ckpt 检查点有93.1%准确率

## 项目结构

```
.
├── config.py                # 配置文件
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── requirements.txt         
├── README.md                
├── Dockerfile               
├── .dockerignore            
├── data/                    # 数据目录（自动下载CIFAR-10）
├── logs/                    # 训练日志（TensorBoard和CSV）
├── checkpoints/             # 模型检查点
│   ├── history/             # 历史检查点（每10分钟自动保存）
│   └── eg/                  # 示例检查点
│       └── PreAct-ResNet45-3500-0.931.ckpt  # PreAct-ResNet45检查点（93.1%准确率）
└── src/
    ├── __init__.py         
    ├── data_module.py          # 数据加载模块
    ├── model.py                # ResNet模型定义
    └── history_checkpoint.py   # 历史检查点回调
```

## 环境配置

### 1. Python环境
```bash
pip install -r requirements.txt
```

### 2. Docker支持
项目提供Docker支持：
```bash
# 构建Docker镜像（使用清华源加速）
docker build -t cifar10-cnn .

# 运行训练
docker run --gpus all cifar10-cnn

# 挂载数据目录持久化
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/logs:/app/logs cifar10-cnn
```

**注意**：Dockerfile默认使用CUDA 12.6运行环境，如需其他版本请修改基础镜像。

## 快速开始

### 1. 训练模型

```bash
python train.py
```

训练过程中会：
- 自动下载CIFAR-10数据集（~163MB）
- 使用多种数据增强技术
- 保存最佳模型检查点（top-3 + last.ckpt）
- 自动保存历史检查点（每10分钟到`checkpoints/history/`）
- 记录TensorBoard和CSV双日志

### 2. 测试模型

使用示例检查点进行测试：
```bash
# 使用PreAct-ResNet45 3500步训练的检查点（约93.1%准确率）
python test.py --checkpoint ./checkpoints/eg/PreAct-ResNet45-3500-0.931.ckpt

# 使用历史检查点
python test.py --checkpoint ./checkpoints/history/1.ckpt
```

### 3. 使用TensorBoard查看训练过程

```bash
tensorboard --logdir ./logs
```

可查看以下指标：
- 训练/验证损失和准确率曲线
- 学习率变化曲线
- 模型检查点时间线

## 模型架构

项目实现了基于残差网络的深度卷积神经网络PreAct-ResNet45，专为CIFAR-10图像分类任务优化。
- 使用预激活（Pre-Activation）的残差块，归一化和激活函数在卷积层之前
- 更深层的网络结构（45层），提升特征提取能力

### 残差块结构（BasicBlock）
```
输入 → [批归一化 → ReLU → 3×3卷积] → [批归一化 → ReLU → 3×3卷积] → 残差连接 → 输出
```

## 数据增强

训练时使用多种数据增强技术提升模型泛化能力：

### 训练集增强
1. **随机裁剪**：
   - 32×32随机裁剪，四周填充4像素
   - 增加位置不变性

2. **随机水平翻转**：
   - 概率0.5的水平镜像
   - 增强左右对称性识别

3. **随机旋转**：
   - ±15度范围内随机旋转
   - 增强旋转不变性

4. **Cutout遮挡**：
   - 随机遮挡8×8像素区域
   - 增强局部特征鲁棒性

### 验证/测试集处理
- 仅进行`ToTensor`和`Normalize`变换
- 无随机性，保证评估一致性

## 训练策略

### 1. 优化器配置
- **优化器**：随机梯度下降（SGD）
  - 学习率：0.1
  - 动量：0.9
  - 权重衰减：5e-4
- **梯度裁剪**：全局梯度裁剪阈值1.0，防止梯度爆炸

### 2. 学习率调度
- **策略**：Warmup + 余弦退火
  - Warmup周期：5 epochs（线性增长）
  - 余弦退火周期：195 epochs（总epochs 200减去warmup周期）
  - 最小学习率：0
- **调度器组合**：使用SequentialLR组合warmup和余弦退火调度器

### 3. 训练控制
- **最大周期数**：200 epochs (可调)
- **重启周期**：200 epochs (可调)
- **早停策略**：监控验证准确率，patience=10000（实际不触发早停，确保完整训练）
- **随机种子**：42
- **批次大小**：256
- **工作进程数**：8（数据加载并行化）

## 日志和监控

### 1. 双日志
- **TensorBoard日志**：`./logs/cifar10_cnn/`
- **CSV日志**：`./logs/cifar10_csv/`

### 2. 检查点策略
- **最佳模型检查点**：
  - 保存top-3验证准确率最高的模型
  - 文件名格式：`cifar10-{epoch:03d}-{val_accuracy:.3f}.ckpt`
  - 同时保存`last.ckpt`（最后一个epoch）
- **历史检查点**（自动）：
  - 每10分钟复制最新检查点到`checkpoints/history/`
  - 文件名格式：`1.ckpt`、`2.ckpt`...
  - 创建训练过程的时间序列备份

### 3. 监控指标
- **训练指标**：
  - `train_loss`：训练损失
  - `train_accuracy`：训练准确率
- **验证指标**：
  - `val_loss`：验证损失
  - `val_accuracy`：验证准确率（检查点监控指标）

### 4. 回调函数
- `ModelCheckpoint`：模型检查点保存
- `EarlyStopping`：早停控制（实际上不触发）
- `LearningRateMonitor`：学习率监控
- `HistoryCheckpoint`：历史检查点自动保存


## 许可证

MIT License
