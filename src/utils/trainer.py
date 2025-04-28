import torch
import numpy as np
from tqdm import tqdm
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.model_config import *
from config.train_config import *
from config.data_config import *
from models.vae import ESMVAE, vae_loss
from utils.data_utils import load_sequences, create_data_loaders

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 7, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss: float):
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss

class VAETrainer:
    """VAE训练器类"""
    
    def __init__(
        self,
        model: ESMVAE,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_save_dir: str = MODEL_SAVE_DIR,
        log_dir: str = LOG_DIR,
        log_level: str = LOG_LEVEL
    ):
        """初始化训练器
        
        Args:
            model: VAE模型
            optimizer: 优化器
            device: 设备
            model_save_dir: 模型保存目录
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        
        # 设置日志
        self.setup_logging(log_level)
        
        # 初始化学习率调度器
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            **LEARNING_RATE_SCHEDULER
        )
        
        # 初始化早停
        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            verbose=True
        )
        
        # 初始化最佳验证损失
        self.best_val_loss = float('inf')
    
    def setup_logging(self, log_level: str) -> None:
        """设置日志记录"""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(
            self.log_dir,
            f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            recon_batch, mean, logvar = self.model(data)
            loss, recon_loss, kl_loss = vae_loss(
                recon_batch, data, mean, logvar, KL_WEIGHT
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            if batch_idx % 100 == 0:
                logging.info(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.6f}'
                )
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
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
        else:
            torch.save(checkpoint, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = VAE_EPOCHS
    ) -> None:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
        """
        logging.info("开始训练...")
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_metrics['loss'])
            
            # 打印指标
            logging.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_metrics["loss"]:.6f}, '
                f'Val Loss: {val_metrics["loss"]:.6f}'
            )
            
            # 保存检查点
            if epoch % MODEL_SAVE_FREQUENCY == 0:
                self.save_checkpoint(
                    epoch,
                    val_metrics['loss']
                )
            
            # 早停检查
            self.early_stopping(val_metrics['loss'])
            if self.early_stopping.early_stop:
                logging.info("早停触发")
                break
            
            # 更新最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(
                    epoch,
                    val_metrics['loss'],
                    is_best=True
                )
        
        logging.info("训练完成") 