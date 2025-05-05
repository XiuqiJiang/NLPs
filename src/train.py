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
from tqdm import tqdm
from transformers import AutoTokenizer

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vae_token import ESMVAEToken
from src.utils.data_utils import (
    create_data_loaders,
    ProteinDataset,
    load_sequences,
    preprocess_and_embed
)
from src.utils.logger import setup_logger
from src.utils.trainer import VAETrainer
from config.config import (
    RANDOM_SEED,
    DEVICE,
    BATCH_SIZE,
    LATENT_DIM,
    HIDDEN_DIMS,
    LEARNING_RATE,
    NUM_EPOCHS,
    BETA,
    SAVE_EVERY,
    SAVE_DIR,
    LOG_DIR,
    ESM_EMBEDDING_DIM,
    MAX_SEQUENCE_LENGTH,
    ESM_MODEL_NAME,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    MODEL_SAVE_DIR,
    LOG_LEVEL,
    EMBEDDING_FILE,
    TRAIN_TEST_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
    ESM_MODEL_PATH,
    SEQUENCE_FILE,
    get_beta
)

def vae_token_loss(
    recon_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int,  # 添加epoch参数
    pad_token_id: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算VAE的损失
    
    Args:
        recon_logits: 重建的logits
        input_ids: 输入的token IDs
        mean: 隐变量的均值
        logvar: 隐变量的对数方差
        epoch: 当前epoch
        pad_token_id: padding token的ID
        
    Returns:
        (总损失, 重建损失, KL损失)
    """
    # 计算当前epoch的beta值
    beta = get_beta(epoch, max_beta=1.0, annealing_epochs=500)
    
    # 计算重建损失（交叉熵）
    recon_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_token_id)(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1)
    )
    
    # 计算KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 总损失
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss

def train_epoch(
    model: ESMVAEToken,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,  # 添加epoch参数
    logger: logging.Logger
) -> tuple[float, float, float]:
    """训练一个epoch
    
    Args:
        model: VAE模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
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
        recon_logits, mean, logvar = model(
            batch['embeddings'],
            batch['attention_mask'],
            batch['input_ids']  # 传入target_ids用于teacher forcing
        )
        
        # 计算损失
        loss, recon_loss, kl_loss = vae_token_loss(
            recon_logits,
            batch['input_ids'],
            mean,
            logvar,
            epoch,
            model.pad_token_id
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """主训练函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("开始训练...")
    
    # 检查数据文件是否存在
    if not os.path.exists(EMBEDDING_FILE):
        logger.info(f"数据文件 {EMBEDDING_FILE} 不存在，开始预处理数据...")
        # 加载原始序列
        logger.info(f"从文件 {SEQUENCE_FILE} 加载序列...")
        sequences = load_sequences(SEQUENCE_FILE)
        logger.info(f"从文件 {SEQUENCE_FILE} 中成功加载 {len(sequences)} 个序列")
        
        if len(sequences) < 10:
            logger.warning(f"加载的序列数量过少 ({len(sequences)})，请检查数据文件")
            return
        
        # 预处理数据
        logger.info("开始预处理数据...")
        preprocess_and_embed(
            sequences=sequences,
            output_path=EMBEDDING_FILE,
            model_path=ESM_MODEL_PATH,
            batch_size=BATCH_SIZE,
            max_length=MAX_SEQUENCE_LENGTH
        )
        logger.info(f"数据预处理完成，保存到 {EMBEDDING_FILE}")
    else:
        logger.info(f"使用现有的数据文件: {EMBEDDING_FILE}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        data_file=EMBEDDING_FILE,
        batch_size=BATCH_SIZE,
        train_test_split=TRAIN_TEST_SPLIT,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    )
    
    # 加载ESM tokenizer以获取词汇表大小
    logger.info("加载ESM tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    logger.info(f"ESM tokenizer词汇表大小: {vocab_size}")
    logger.info(f"ESM tokenizer pad token ID: {pad_token_id}")
    
    # 初始化模型
    logger.info("初始化模型...")
    model = ESMVAEToken(
        input_dim=ESM_EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        vocab_size=vocab_size,  # 使用ESM tokenizer的词汇表大小
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=pad_token_id,  # 使用ESM tokenizer的pad token ID
        use_layer_norm=True,
        dropout=0.1
    )
    
    # 初始化优化器
    logger.info("初始化优化器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01  # 添加L2正则化
    )
    
    # 初始化训练器
    logger.info("初始化训练器...")
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        device=DEVICE,
        pad_token_id=pad_token_id,  # 使用ESM tokenizer的pad token ID
        beta=BETA,
        max_grad_norm=1.0,
        log_dir=LOG_DIR
    )
    
    # 创建模型保存目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 训练模型
    logger.info("开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        val_loader=val_loader,
        save_dir=MODEL_SAVE_DIR,
        save_freq=5
    )
    
    logger.info("训练完成！")
    return history

if __name__ == "__main__":
    main() 