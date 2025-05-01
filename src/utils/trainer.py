import torch
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging
from pathlib import Path

# 导入配置文件
import sys
from pathlib import Path

# 获取项目根目录
root_dir = Path(__file__).parent.parent.parent
config_path = root_dir / 'config' / 'train_config.py'

# 检查配置文件是否存在
if not config_path.exists():
    raise FileNotFoundError(f"配置文件不存在: {config_path}")

# 添加项目根目录到Python路径
sys.path.append(str(root_dir))

# 导入配置
from config.train_config import *

# 训练参数设置
NUM_EPOCHS = 100  # 训练轮数
SAVE_EVERY = 5    # 每5轮保存一次模型
EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
EARLY_STOPPING_MIN_DELTA = 0.01  # 早停最小变化值
KL_WEIGHT = 0.1   # KL散度权重
MODEL_SAVE_DIR = 'results/models/vae/weights'  # 模型保存目录

from src.models.vae import ESMVAE, vae_loss
from src.utils.data_utils import load_sequences, create_data_loaders

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 7, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss: float):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss

class VAETrainer:
    """VAE训练器类"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_save_dir: str
    ):
        """初始化训练器
        
        Args:
            model: VAE模型
            optimizer: 优化器
            device: 设备
            model_save_dir: 模型保存目录
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_save_dir = model_save_dir
        
        # 创建模型保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 初始化早停
        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA
        )
        
        # 初始化最佳验证损失
        self.best_val_loss = float('inf')
        
        # 初始化logger
        self._setup_logger()
    
    def _setup_logger(self):
        """设置logger"""
        # 创建日志目录
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger('VAETrainer')
        self.logger.setLevel(LOG_LEVEL)
        
        # 创建文件处理器
        log_file = os.path.join(LOG_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVEL)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logger initialized")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch in train_loader:
            # 将数据移到GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(batch)
            reconstructed_batch, mean, logvar = outputs
            
            # 计算损失
            embeddings_batch = batch['embeddings']
            recon_loss = self.criterion(reconstructed_batch, embeddings_batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + self.kl_weight * kl_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        # 计算平均损失
        num_batches = len(train_loader)
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_recon_loss': total_recon_loss / num_batches,
            'train_kl_loss': total_kl_loss / num_batches
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证指标
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                recon_batch, mean, logvar = self.model(data)
                loss, recon_loss, kl_loss = vae_loss(
                    recon_batch, data, mean, logvar, KL_WEIGHT
                )
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ) -> None:
        """保存模型检查点
        
        Args:
            epoch: 当前epoch
            val_loss: 验证损失
            is_best: 是否是最佳模型
        """
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_save_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {val_loss:.6f}")
        else:
            torch.save(checkpoint, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"保存第 {epoch} 轮模型，验证损失: {val_loss:.6f}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
        save_every: int = SAVE_EVERY
    ) -> Dict[str, List[float]]:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_every: 每隔多少轮保存一次模型
            
        Returns:
            训练过程中的指标记录
        """
        self.logger.info("开始训练...")
        
        # 初始化指标记录
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': []
        }
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 记录指标
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_metrics["train_loss"]:.6f}, '
                f'Val Loss: {val_metrics["loss"]:.6f}'
            )
            
            # 记录指标
            metrics_history['train_loss'].append(train_metrics['train_loss'])
            metrics_history['val_loss'].append(val_metrics['loss'])
            metrics_history['train_recon_loss'].append(train_metrics['train_recon_loss'])
            metrics_history['train_kl_loss'].append(train_metrics['train_kl_loss'])
            metrics_history['val_recon_loss'].append(val_metrics['recon_loss'])
            metrics_history['val_kl_loss'].append(val_metrics['kl_loss'])
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(
                    epoch,
                    val_metrics['loss'],
                    is_best=True
                )
                self.logger.info(f"保存最佳模型，验证损失: {val_metrics['loss']:.6f}")
            
            # 定期保存模型
            if epoch % save_every == 0:
                self.save_checkpoint(
                    epoch,
                    val_metrics['loss']
                )
                self.logger.info(f"保存第 {epoch} 轮模型，验证损失: {val_metrics['loss']:.6f}")
            
            # 早停检查
            if self.early_stopping(val_metrics['loss']):
                self.logger.info("触发早停，停止训练")
                break
        
        return metrics_history 