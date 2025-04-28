import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, Any
from datetime import datetime

from config.model_config import *
from config.train_config import *
from config.data_config import *
from models.vae import ESMVAE, vae_loss
from utils.data_utils import load_sequences, create_data_loaders
from utils.model_utils import EarlyStopping, save_checkpoint

# 设置日志
def setup_logging() -> None:
    """设置日志记录"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        LOG_DIR,
        f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_epoch(
    model: ESMVAE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        
    Returns:
        训练指标
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mean, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss(
            recon_batch, data, mean, logvar, KL_WEIGHT
        )
        
        loss.backward()
        optimizer.step()
        
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
    model: ESMVAE,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        
    Returns:
        验证指标
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_batch, mean, logvar = model(data)
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

def main() -> None:
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 设置日志
    setup_logging()
    logging.info("开始训练...")
    
    # 加载数据
    logging.info("加载数据...")
    sequences = load_sequences(RAW_DATA_PATH)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        sequences,
        tokenizer,
        batch_size=VAE_BATCH_SIZE,
        train_test_split=TRAIN_TEST_SPLIT,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESMVAE().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LEARNING_RATE)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **LEARNING_RATE_SCHEDULER
    )
    
    # 早停
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(1, VAE_EPOCHS + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 打印指标
        logging.info(
            f'Epoch {epoch}: '
            f'Train Loss: {train_metrics["loss"]:.6f}, '
            f'Val Loss: {val_metrics["loss"]:.6f}'
        )
        
        # 保存检查点
        if epoch % MODEL_SAVE_FREQUENCY == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics['loss'],
                MODEL_SAVE_DIR
            )
        
        # 早停检查
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            logging.info("早停触发")
            break
        
        # 更新最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics['loss'],
                MODEL_SAVE_DIR,
                is_best=True
            )
    
    logging.info("训练完成")

if __name__ == '__main__':
    main() 