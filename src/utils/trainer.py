import torch
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

# 导入配置文件
import sys
from pathlib import Path

# 获取项目根目录
root_dir = Path(__file__).parent.parent.parent
config_path = root_dir / 'config' / 'config.py'

# 检查配置文件是否存在
if not config_path.exists():
    raise FileNotFoundError(f"配置文件不存在: {config_path}")

# 添加项目根目录到Python路径
sys.path.append(str(root_dir))

# 导入配置
from config.config import *

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
        model_save_dir: str,
        pad_token_id: int,
        kl_weight: float = 0.1
    ):
        """初始化训练器
        
        Args:
            model: VAE模型
            optimizer: 优化器
            device: 设备
            model_save_dir: 模型保存目录
            pad_token_id: padding token的ID
            kl_weight: KL散度的权重
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_save_dir = model_save_dir
        self.pad_token_id = pad_token_id
        self.kl_weight = kl_weight
        
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
    
    def _compute_loss(
        self,
        recon_logits: torch.Tensor,
        target_ids: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算VAE损失
        
        Args:
            recon_logits: 重构的logits，形状为 [batch_size, seq_len, vocab_size]
            target_ids: 目标token IDs，形状为 [batch_size, seq_len]
            mu: 潜在空间均值，形状为 [batch_size, latent_dim]
            logvar: 潜在空间对数方差，形状为 [batch_size, latent_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]（可选）
            
        Returns:
            (总损失, 重构损失, KL散度)
        """
        batch_size = recon_logits.size(0)
        
        # 重构损失
        # 将logits和targets展平为2D张量
        logits = recon_logits.view(-1, recon_logits.size(-1))  # [batch_size * seq_len, vocab_size]
        targets = target_ids.view(-1)  # [batch_size * seq_len]
        
        # 使用交叉熵损失，忽略padding token
        recon_loss = F.cross_entropy(logits, targets, ignore_index=self.pad_token_id)
        
        # 如果提供了attention mask，可以计算非padding token的平均损失（可选）
        if attention_mask is not None:
            # 计算非padding token的数量
            num_tokens = attention_mask.sum().item()
            if num_tokens > 0:
                # 只考虑非padding token的损失
                recon_loss = recon_loss * (logits.size(0) / num_tokens)
        
        # KL散度
        # 先按潜变量维度求和，再按batch求平均
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # 总损失
        total_loss = recon_loss + self.kl_weight * kld_loss
        
        return total_loss, recon_loss, kld_loss
    
    def train_epoch(self, train_loader: DataLoader) -> None:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # 将数据移到GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            recon_logits, mu, logvar = self.model(batch['embeddings'])
            
            # 计算损失
            loss, recon_loss, kld_loss = self._compute_loss(
                recon_logits=recon_logits,
                target_ids=batch['token_ids'],
                mu=mu,
                logvar=logvar,
                attention_mask=batch['attention_mask']
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kld_loss = total_kld_loss / num_batches
        
        self.logger.info(
            f"训练损失: {avg_loss:.6f} "
            f"(重构: {avg_recon_loss:.6f}, KL: {avg_kld_loss:.6f})"
        )
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证损失
        """
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移到GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                recon_logits, mu, logvar = self.model(batch['embeddings'])
                
                # 计算损失
                loss, recon_loss, kld_loss = self._compute_loss(
                    recon_logits=recon_logits,
                    target_ids=batch['token_ids'],
                    mu=mu,
                    logvar=logvar,
                    attention_mask=batch['attention_mask']
                )
                
                # 更新统计信息
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kld_loss += kld_loss.item()
                num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kld_loss = total_kld_loss / num_batches
        
        self.logger.info(
            f"验证损失: {avg_loss:.6f} "
            f"(重构: {avg_recon_loss:.6f}, KL: {avg_kld_loss:.6f})"
        )
        
        return avg_loss
    
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
    ) -> None:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_every: 每隔多少轮保存一次模型
        """
        self.logger.info("开始训练...")
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            self.train_epoch(train_loader)
            
            # 验证并获取损失
            val_loss = self.validate(val_loader)
            
            # 记录当前epoch
            self.logger.info(f'Epoch {epoch} 完成，验证损失: {val_loss:.6f}')
            
            # 定期保存模型
            if epoch % save_every == 0:
                self.save_checkpoint(
                    epoch,
                    val_loss
                )
                self.logger.info(f"保存第 {epoch} 轮模型") 