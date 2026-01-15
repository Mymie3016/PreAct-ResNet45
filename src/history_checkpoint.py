import os
import time
import shutil
from pytorch_lightning.callbacks import Callback


class HistoryCheckpoint(Callback):
    """每10分钟复制最新检查点到history目录，重命名为'num.ckpt'格式"""
    
    def __init__(self, checkpoint_callback, interval_minutes=10, history_dir="./checkpoints/history"):
        """
            checkpoint_callback: ModelCheckpoint回调实例，用于获取检查点路径
            interval_minutes: 复制检查点的间隔分钟数，默认10分钟
            history_dir: history目录路径，相对或绝对路径
        """
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.interval_minutes = interval_minutes
        self.history_dir = history_dir
        
        self.start_time = None
        self.last_processed_minute = -interval_minutes  # 初始化为-interval_minutes确保第一次触发
        self.interval_count = 0  # 记录第几个10分钟间隔
        
    def on_train_start(self, trainer, pl_module):
        """记录训练开始时间，创建history目录"""
        self.start_time = time.time()
        
        # 创建history目录
        os.makedirs(self.history_dir, exist_ok=True)
        
        # 初始处理：复制当前最新检查点（如果有）
        self._process_latest_checkpoint()
        
    def on_train_epoch_end(self, trainer, pl_module):
        """每个epoch结束时检查是否到达时间间隔"""
        if self.start_time is None:
            return
            
        elapsed_seconds = time.time() - self.start_time
        elapsed_minutes = elapsed_seconds // 60
        
        # 计算当前时间点所属的分钟标记（向下取整到interval的倍数）
        current_minute_mark = (elapsed_minutes // self.interval_minutes) * self.interval_minutes
        
        # 如果超过了上次处理的分钟标记，执行复制
        if current_minute_mark > self.last_processed_minute:
            self._process_latest_checkpoint()
            self.last_processed_minute = current_minute_mark
            self.interval_count += 1  # 增加间隔计数
    
    def _process_latest_checkpoint(self):
        """复制最新检查点到history目录"""
        # 获取最新检查点路径（last.ckpt）
        last_model_path = getattr(self.checkpoint_callback, "last_model_path", None)
        
        # 如果last_model_path不存在，尝试从checkpoint目录查找最新的检查点文件[6](@ref)
        # if not last_model_path or not os.path.exists(last_model_path):
        #     last_model_path = self._find_latest_checkpoint()
            
        if not last_model_path or not os.path.exists(last_model_path):
            return
            
        # 生成新的文件名：第几个间隔.ckpt
        new_filename = f"{self.interval_count}.ckpt"
        new_filepath = os.path.join(self.history_dir, new_filename)
        
        # 复制文件
        try:
            shutil.copy2(last_model_path, new_filepath)
            # print(f"[HistoryCheckpoint] 复制检查点到 history: {new_filename} (间隔 {self.interval_count})")
        except Exception as e:
            print(f"[HistoryCheckpoint] 复制失败: {e}")
    
    def _find_latest_checkpoint(self):
        """在checkpoint目录中查找最新的检查点文件[6](@ref)"""
        # 获取checkpoint目录路径
        checkpoint_dir = getattr(self.checkpoint_callback, "dirpath", None)
        if not checkpoint_dir or not os.path.exists(checkpoint_dir):
            return None
            
        # 查找目录中所有的.ckpt文件
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if not ckpt_files:
            return None
            
        # 按修改时间排序，获取最新的文件
        latest_file = max(ckpt_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
        return os.path.join(checkpoint_dir, latest_file)