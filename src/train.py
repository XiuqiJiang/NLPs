import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os
from datetime import datetime
from typing import Tuple
import sys
import logging

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vae_token import ESMVAEToken, vae_token_loss
from src.data.dataset import create_data_loaders
from src.utils.logger import setup_logger
from config.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    KL_WEIGHT,
    SAVE_DIR,
    LOG_DIR,
    LOG_LEVEL
)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float,
    logger: logging.Logger
) -> Tuple[float, float, float]:
    """训练一个epoch
    
    Args:
        model: VAE模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        kl_weight: KL散度权重
        logger: 日志记录器
        
    Returns:
        平均总损失、重建损失和KL散度
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # 将数据移到设备上
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        recon_logits, mean, logvar = model(batch)
        
        # 计算损失
        loss, recon_loss, kl_loss = vae_token_loss(
            recon_logits,
            batch['input_ids'],
            mean,
            logvar,
            kl_weight,
            model.tokenizer.pad_token_id
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加损失
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    # 记录训练信息
    logger.info(
        f"Training - "
        f"Loss: {avg_loss:.4f} - "
        f"Recon Loss: {avg_recon_loss:.4f} - "
        f"KL Loss: {avg_kl_loss:.4f}"
    )
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(SAVE_DIR, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(LOG_DIR, level=LOG_LEVEL)
    logger.info(f"Starting training with device: {device}")
    
    # 加载数据集
    train_loader, val_loader = create_data_loaders()
    logger.info(f"Loaded {len(train_loader.dataset)} training sequences")
    logger.info(f"Loaded {len(val_loader.dataset)} validation sequences")
    
    # 初始化模型
    model = ESMVAEToken().to(device)
    logger.info("Initialized VAE model")
    
    # 初始化优化器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info(f"Initialized optimizer with learning rate: {LEARNING_RATE}")
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")
        
        # 训练一个epoch
        avg_loss, avg_recon_loss, avg_kl_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            KL_WEIGHT,
            logger
        )
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                },
                model_path
            )
            logger.info(f"Saved best model to {model_path}")
        
        # 每个epoch保存一次检查点
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            },
            checkpoint_path
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Training completed")

if __name__ == "__main__":
    main() 