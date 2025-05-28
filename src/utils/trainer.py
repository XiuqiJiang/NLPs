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

# 修改导入语句
from src.models.vae_token import ESMVAEToken
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
    """VAE模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        pad_token_id: int,
        beta: float = 1.0,
        max_grad_norm: float = 1.0,
        log_dir: str = "results/logs"
    ):
        """初始化训练器
        
        Args:
            model: VAE模型
            optimizer: 优化器
            device: 设备
            pad_token_id: padding token的ID
            beta: KL散度权重
            max_grad_norm: 梯度裁剪阈值
            log_dir: 日志目录
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.pad_token_id = pad_token_id
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        
        # 设置日志
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 将模型移动到设备
        self.model.to(device)
    
    def _compute_loss(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_ids: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算 VAE 损失，使用Free Bits策略
        
        Args:
            x: 输入张量
            attention_mask: 注意力掩码
            target_token_ids: 目标token ids
            epoch: 当前epoch
            
        Returns:
            (总损失, 损失字典)
        """
        # 计算当前epoch的beta值，使用config.py全局变量
        beta = get_beta(epoch, max_beta=MAX_BETA, annealing_epochs=ANNEALING_EPOCHS)
        
        # 编码
        mu, logvar = self.model.encode(x)
        
        # 重参数化
        z = self.model.reparameterize(mu, logvar)
        
        # 解码（使用teacher forcing）
        logits = self.model.decode(z, target_token_ids)
        
        # 计算重构损失
        flat_logits = logits.view(-1, self.model.vocab_size)
        flat_target = target_token_ids.view(-1)
        
        recon_loss = F.cross_entropy(
            flat_logits,
            flat_target,
            ignore_index=self.pad_token_id
        )
        
        # 计算KL散度
        # mu, logvar: [batch, latent_dim]
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [batch]
        kl_loss = kl_per_sample.mean()  # 标量
        # KL target loss软约束
        kld_target_component = KLD_TARGET_WEIGHT * (kl_loss - KLD_TARGET) ** 2
        total_loss = recon_loss + beta * kl_loss + kld_target_component
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),  # 记录原始kl_loss用于监控
            'beta': beta
        }
        
        print(f"mu mean: {mu.mean().item():.4f}, logvar mean: {logvar.mean().item():.4f}, kl_loss: {kl_loss.item():.4f}")
        
        return total_loss, loss_dict
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            包含训练指标的字典
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # 数据检查
            if (torch.isnan(batch['embeddings']).any() or torch.isinf(batch['embeddings']).any() or
                torch.isnan(batch['token_ids'].float()).any() or torch.isinf(batch['token_ids'].float()).any() or
                torch.isnan(batch['attention_mask'].float()).any() or torch.isinf(batch['attention_mask'].float()).any()):
                print(f"[Train数据异常] batch: nan/inf detected! 跳过该batch。")
                print(f"embeddings nan: {torch.isnan(batch['embeddings']).any().item()}, inf: {torch.isinf(batch['embeddings']).any().item()}")
                print(f"token_ids nan: {torch.isnan(batch['token_ids'].float()).any().item()}, inf: {torch.isinf(batch['token_ids'].float()).any().item()}")
                print(f"attention_mask nan: {torch.isnan(batch['attention_mask'].float()).any().item()}, inf: {torch.isinf(batch['attention_mask'].float()).any().item()}")
                continue
            # 极端值检查
            if (batch['embeddings'].abs().max() > 1e4 or batch['token_ids'].abs().max() > 1e6):
                print(f"[Train数据异常] batch: 极端值 detected! 跳过该batch。")
                print(f"embeddings min: {batch['embeddings'].min().item()}, max: {batch['embeddings'].max().item()}")
                print(f"token_ids min: {batch['token_ids'].min().item()}, max: {batch['token_ids'].max().item()}")
                continue
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 计算损失，传入所有参数
            loss, loss_dict = self._compute_loss(
                batch['embeddings'],
                batch['attention_mask'],
                batch['token_ids'],
                epoch
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss']
            total_kld_loss += loss_dict['kl_loss']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kld': f"{loss_dict['kl_loss']:.4f}",
                'beta': f"{loss_dict['beta']:.4f}"
            })
        
        num_batches = len(train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kld_loss': total_kld_loss / num_batches,
            'beta': loss_dict['beta']
        }
        
        self.logger.info(
            f"Epoch {epoch} - "
            f"Loss: {metrics['loss']:.4f}, "
            f"Recon: {metrics['recon_loss']:.4f}, "
            f"KLD: {metrics['kld_loss']:.4f}, "
            f"Beta: {metrics['beta']:.4f}"
        )
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None,
        save_freq: int = 5
    ) -> Dict[str, list]:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            val_loader: 验证数据加载器
            save_dir: 模型保存目录
            save_freq: 保存频率
            
        Returns:
            包含训练历史的字典
        """
        history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kld_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kld_loss': [],
            'beta': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # 更新训练历史
            history['train_loss'].append(train_metrics['loss'])
            history['train_recon_loss'].append(train_metrics['recon_loss'])
            history['train_kld_loss'].append(train_metrics['kld_loss'])
            history['beta'].append(train_metrics['beta'])
            
            # 验证
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                
                # 更新验证历史
                history['val_loss'].append(val_metrics['loss'])
                history['val_recon_loss'].append(val_metrics['recon_loss'])
                history['val_kld_loss'].append(val_metrics['kld_loss'])
                
                # 保存最佳模型
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if save_dir is not None:
                        self._save_checkpoint(
                            epoch,
                            os.path.join(save_dir, 'best_model.pt')
                        )
            
            # 定期保存检查点
            if save_dir is not None and epoch % save_freq == 0:
                self._save_checkpoint(
                    epoch,
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                )
        
        return history
    
    def _validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 数据检查
                if (torch.isnan(batch['embeddings']).any() or torch.isinf(batch['embeddings']).any() or
                    torch.isnan(batch['token_ids'].float()).any() or torch.isinf(batch['token_ids'].float()).any() or
                    torch.isnan(batch['attention_mask'].float()).any() or torch.isinf(batch['attention_mask'].float()).any()):
                    print(f"[Val数据异常] batch: nan/inf detected! 跳过该batch。")
                    print(f"embeddings nan: {torch.isnan(batch['embeddings']).any().item()}, inf: {torch.isinf(batch['embeddings']).any().item()}")
                    print(f"token_ids nan: {torch.isnan(batch['token_ids'].float()).any().item()}, inf: {torch.isinf(batch['token_ids'].float()).any().item()}")
                    print(f"attention_mask nan: {torch.isnan(batch['attention_mask'].float()).any().item()}, inf: {torch.isinf(batch['attention_mask'].float()).any().item()}")
                    continue
                # 极端值检查
                if (batch['embeddings'].abs().max() > 1e4 or batch['token_ids'].abs().max() > 1e6):
                    print(f"[Val数据异常] batch: 极端值 detected! 跳过该batch。")
                    print(f"embeddings min: {batch['embeddings'].min().item()}, max: {batch['embeddings'].max().item()}")
                    print(f"token_ids min: {batch['token_ids'].min().item()}, max: {batch['token_ids'].max().item()}")
                    continue
                
                # 将数据移动到正确的设备上
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 计算损失
                loss, loss_dict = self._compute_loss(
                    batch['embeddings'],
                    batch['attention_mask'],
                    batch['token_ids'],
                    -1
                )
                
                # 更新统计信息
                total_loss += loss.item()
                total_recon_loss += loss_dict['recon_loss']
                total_kld_loss += loss_dict['kl_loss']
        
        # 计算平均损失
        num_batches = len(val_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kld_loss': total_kld_loss / num_batches
        }
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        path: str
    ) -> None:
        """保存检查点
        
        Args:
            epoch: 当前epoch
            path: 保存路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'pad_token_id': self.pad_token_id,
            'beta': self.beta
        }
        torch.save(checkpoint, path)
        self.logger.info(f"保存检查点到 {path}")
    
    def load_checkpoint(
        self,
        path: str
    ) -> int:
        """加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            加载的epoch
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.pad_token_id = checkpoint['pad_token_id']
        self.beta = checkpoint['beta']
        self.logger.info(f"从 {path} 加载检查点")
        return checkpoint['epoch']

def get_beta(epoch: int, max_beta: float = 1.0, annealing_epochs: int = ANNEALING_EPOCHS) -> float:
    """计算当前epoch的beta值
    
    Args:
        epoch: 当前epoch
        max_beta: 最大beta值
        annealing_epochs: beta退火的周期数
        
    Returns:
        beta值，范围从0到max_beta
    """
    warmup_epochs = 200  # 预热期epoch数
    annealing_epochs_after_warmup = 800  # 预热后达到max_beta所需的epoch数
    
    if epoch < warmup_epochs:
        return 0.0
    else:
        # 计算预热期结束后的当前退火阶段的epoch
        current_annealing_epoch = epoch - warmup_epochs
        # 线性增长 beta
        beta = min(1.0, current_annealing_epoch / annealing_epochs_after_warmup) * max_beta
        return beta 