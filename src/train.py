import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from pathlib import Path

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vae_token import ESMVAEToken
from src.data.dataset import create_data_loaders
from src.utils.logger import setup_logger
from config.config import (
    RANDOM_SEED,
    DEVICE,
    BATCH_SIZE,
    LATENT_DIM,
    HIDDEN_DIMS,
    LEARNING_RATE,
    NUM_EPOCHS,
    KL_WEIGHT,
    SAVE_EVERY,
    SAVE_DIR,
    LOG_DIR,
    ESM_EMBEDDING_DIM,
    ALPHABET,
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    PAD_TOKEN_ID,
    EOS_TOKEN_ID,
    ESM_MODEL_NAME,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    MODEL_SAVE_DIR,
    LOG_LEVEL
)
from src.utils.data_utils import ProteinDataset
from src.utils.trainer import VAETrainer

def vae_token_loss(
    recon_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算VAE的损失
    
    Args:
        recon_logits: 重建的logits
        input_ids: 输入的token IDs
        mean: 隐变量的均值
        logvar: 隐变量的对数方差
        kl_weight: KL散度的权重
        
    Returns:
        (总损失, 重建损失, KL损失)
    """
    # 计算重建损失（交叉熵）
    recon_loss = nn.CrossEntropyLoss(reduction='mean')(recon_logits.view(-1, recon_logits.size(-1)), input_ids.view(-1))
    
    # 计算KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 总损失
    loss = recon_loss + kl_weight * kl_loss
    
    return loss, recon_loss, kl_loss

def train_epoch(
    model: ESMVAEToken,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    kl_weight: float,
    logger: logging.Logger
) -> tuple[float, float, float]:
    """训练一个epoch
    
    Args:
        model: VAE模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        kl_weight: KL散度的权重
        logger: 日志记录器
        
    Returns:
        (平均总损失, 平均重建损失, 平均KL损失)
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        recon_logits, mean, logvar = model(batch)
        
        # 计算损失
        loss, recon_loss, kl_loss = vae_token_loss(
            recon_logits,
            batch['input_ids'],
            mean,
            logvar,
            kl_weight
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加损失
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # 记录进度
        if batch_idx % 10 == 0:
            logger.info(
                f"Batch {batch_idx}/{len(dataloader)} - "
                f"Loss: {loss.item():.4f} - "
                f"Recon Loss: {recon_loss.item():.4f} - "
                f"KL Loss: {kl_loss.item():.4f}"
            )
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def setup_logging():
    """设置日志"""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('VAETrainer')
    logger.setLevel(LOG_LEVEL)
    
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
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging()
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        data_path='data/processed/embeddings',
        batch_size=BATCH_SIZE,
        val_split=0.1,  # 使用10%的数据作为验证集
        shuffle=True
    )
    
    # 初始化模型
    logger.info("初始化模型...")
    model = ESMVAEToken(
        embedding_dim=1280,  # ESM-2 的嵌入维度
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        use_layer_norm=True
    ).to(device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 初始化训练器
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_save_dir=MODEL_SAVE_DIR,
        pad_token_id=PAD_TOKEN_ID,
        kl_weight=KL_WEIGHT
    )
    
    # 训练模型
    logger.info("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        save_every=SAVE_EVERY
    )
    
    logger.info("训练完成！")

if __name__ == '__main__':
    main() 