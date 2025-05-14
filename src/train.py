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
import argparse
import matplotlib.pyplot as plt

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
    get_beta,
    MAX_BETA,
    ANNEALING_EPOCHS,
    KLD_TARGET
    )
from src.utils.ring_utils import get_ring_info, analyze_ring_distribution

def vae_token_loss(
    recon_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int,
    pad_token_id: int = 1,
    max_beta: float = MAX_BETA,  # 使用配置文件中的值
    annealing_epochs: int = ANNEALING_EPOCHS,  # 使用配置文件中的值
    kld_target: float = KLD_TARGET  # 使用配置文件中的值
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算VAE的损失，使用Free Bits策略
    
    Args:
        recon_logits: 重建的logits
        input_ids: 输入的token IDs
        mean: 隐变量的均值
        logvar: 隐变量的对数方差
        epoch: 当前epoch
        pad_token_id: padding token的ID
        max_beta: KL散度的最大权重
        annealing_epochs: beta退火的周期数
        kld_target: 目标KLD下限K，单位是nats
        
    Returns:
        (总损失, 重建损失, KL损失)
    """
    # 计算当前epoch的beta值，使用传入的参数
    beta = get_beta(epoch, max_beta=max_beta, annealing_epochs=annealing_epochs)
    
    # 计算重建损失（交叉熵）
    recon_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_token_id)(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1)
    )
    
    # 计算KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 使用Free Bits策略：只有当KLD超过目标值时才计算损失
    # 公式：max(0, KLD - K)
    free_bits_kl_loss = torch.max(torch.zeros_like(kl_loss), kl_loss - kld_target)
    
    # 总损失
    loss = recon_loss + beta * free_bits_kl_loss
    
    return loss, recon_loss, kl_loss  # 返回原始kl_loss用于监控

def train_epoch(
    model: ESMVAEToken,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    logger: logging.Logger,
    max_beta: float = MAX_BETA,  # 使用配置文件中的值
    annealing_epochs: int = ANNEALING_EPOCHS,  # 使用配置文件中的值
    kld_target: float = KLD_TARGET  # 使用配置文件中的值
) -> tuple[float, float, float]:
    """训练一个epoch
    
    Args:
        model: VAE模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        logger: 日志记录器
        max_beta: KL散度的最大权重
        annealing_epochs: beta退火的周期数
        kld_target: 目标KLD下限K，单位是nats
        
    Returns:
        (平均总损失, 平均重建损失, 平均KL损失)
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    # 添加log_var的统计信息
    log_var_mean = 0
    log_var_std = 0
    log_var_min = float('inf')
    log_var_max = float('-inf')
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        recon_logits, mean, logvar = model(
            batch['embeddings'],
            batch['attention_mask'],
            batch['input_ids']  # 传入target_ids用于teacher forcing
        )
        
        # 计算损失，传入所有参数
        loss, recon_loss, kl_loss = vae_token_loss(
            recon_logits,
            batch['input_ids'],
            mean,
            logvar,
            epoch,
            model.pad_token_id,
            max_beta=max_beta,
            annealing_epochs=annealing_epochs,
            kld_target=kld_target
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加损失
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # 更新log_var统计信息
        log_var_mean += logvar.mean().item()
        log_var_std += logvar.std().item()
        log_var_min = min(log_var_min, logvar.min().item())
        log_var_max = max(log_var_max, logvar.max().item())
        
        # 记录进度
        if batch_idx % 10 == 0:
            logger.info(
                f"Batch {batch_idx}/{len(dataloader)} - "
                f"Loss: {loss.item():.4f} - "
                f"Recon Loss: {recon_loss.item():.4f} - "
                f"KL Loss: {kl_loss.item():.4f} - "
                f"Beta: {beta:.4f} - "
                f"KLD Target: {kld_target:.4f}"
            )
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def setup_logging(log_dir: str) -> None:
    """设置日志
    
    Args:
        log_dir: 日志目录
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def compute_loss(
    model: ESMVAEToken,
    x: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kl_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    """计算损失
    
    Args:
        model: VAE模型
        x: 输入序列嵌入 [batch_size, seq_len, embed_dim]
        input_ids: 输入序列的token IDs [batch_size, seq_len]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        kl_weight: KL散度权重
        
    Returns:
        包含各项损失的字典
    """
    # 前向传播
    recon_logits, mu, logvar = model(x, attention_mask, input_ids)
    
    # 重构损失
    recon_loss = nn.CrossEntropyLoss(reduction='none')(recon_logits.view(-1, recon_logits.size(-1)), input_ids.view(-1))
    recon_loss = (recon_loss * attention_mask.view(-1)).sum() / attention_mask.sum()
    
    # KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'log_var': logvar
    }

def train_epoch_new(
    model: ESMVAEToken,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    kl_weight: float
) -> Dict[str, float]:
    """训练一个epoch
    
    Args:
        model: VAE模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        kl_weight: KL散度权重
        
    Returns:
        包含各项损失的字典
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    # 添加log_var的统计信息
    log_var_mean = 0
    log_var_std = 0
    log_var_min = float('inf')
    log_var_max = float('-inf')
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # 将数据移到设备
        x = batch['embeddings'].to(device)  # [batch_size, seq_len, embed_dim]
        attention_mask = batch['attention_mask'].to(device)  # [batch_size, seq_len]
        input_ids = batch['input_ids'].to(device)  # [batch_size, seq_len]
        
        # 打印第一个batch的调试信息
        if batch_idx == 0:
            print(f"Debug - Batch shapes:")
            print(f"x shape: {x.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"input_ids shape: {input_ids.shape}")
        
        # 计算损失
        losses = compute_loss(model, x, input_ids, attention_mask, kl_weight)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        # 更新统计信息
        total_loss += losses['total_loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        
        # 更新log_var统计信息
        log_var_mean += losses['log_var'].mean().item()
        log_var_std += losses['log_var'].std().item()
        log_var_min = min(log_var_min, losses['log_var'].min().item())
        log_var_max = max(log_var_max, losses['log_var'].max().item())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'recon': f"{losses['recon_loss'].item():.4f}",
            'kl': f"{losses['kl_loss'].item():.4f}",
            'log_var_mean': f"{losses['log_var'].mean().item():.4f}"
        })
    
    # 计算平均损失
    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'log_var_stats': {
            'mean': log_var_mean / num_batches,
            'std': log_var_std / num_batches,
            'min': log_var_min,
            'max': log_var_max
        }
    }

@torch.no_grad()
def validate(
    model: ESMVAEToken,
    val_loader: DataLoader,
    device: torch.device,
    kl_weight: float
) -> Dict[str, float]:
    """验证模型
    
    Args:
        model: VAE模型
        val_loader: 验证数据加载器
        device: 设备
        kl_weight: KL散度权重
        
    Returns:
        包含各项损失的字典
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    # 添加log_var的统计信息
    log_var_mean = 0
    log_var_std = 0
    log_var_min = float('inf')
    log_var_max = float('-inf')
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
        # 将数据移到设备
        x = batch['embeddings'].to(device)  # [batch_size, seq_len, embed_dim]
        attention_mask = batch['attention_mask'].to(device)  # [batch_size, seq_len]
        input_ids = batch['input_ids'].to(device)  # [batch_size, seq_len]
        
        # 打印第一个batch的调试信息
        if batch_idx == 0:
            print(f"Debug - Validation Batch shapes:")
            print(f"x shape: {x.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"input_ids shape: {input_ids.shape}")
        
        # 计算损失
        losses = compute_loss(model, x, input_ids, attention_mask, kl_weight)
        
        # 更新统计信息
        total_loss += losses['total_loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        
        # 更新log_var统计信息
        log_var_mean += losses['log_var'].mean().item()
        log_var_std += losses['log_var'].std().item()
        log_var_min = min(log_var_min, losses['log_var'].min().item())
        log_var_max = max(log_var_max, losses['log_var'].max().item())
    
    # 计算平均损失
    num_batches = len(val_loader)
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'log_var_stats': {
            'mean': log_var_mean / num_batches,
            'std': log_var_std / num_batches,
            'min': log_var_min,
            'max': log_var_max
        }
    }

def save_checkpoint(
    model: ESMVAEToken,
    optimizer: optim.Optimizer,
    epoch: int,
    losses: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """保存检查点
    
    Args:
        model: VAE模型
        optimizer: 优化器
        epoch: 当前epoch
        losses: 损失字典
        checkpoint_dir: 检查点目录
        is_best: 是否为最佳模型
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }
    
    # 保存当前epoch的检查点
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"保存检查点到 {checkpoint_path}")
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_model_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_model_path)
        logging.info(f"保存最佳模型到 {best_model_path}")

def main(args=None):
    """主函数
    
    Args:
        args: 命令行参数，如果为None则使用默认配置
    """
    if args is None:
        # 使用配置文件中的默认参数
        args = argparse.Namespace(
            data_file=EMBEDDING_FILE,
            max_length=MAX_SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            val_split=TRAIN_TEST_SPLIT,
            num_workers=NUM_WORKERS,
            input_dim=ESM_EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIMS[0],
            latent_dim=LATENT_DIM,
            dropout=0.1,
            epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            kl_weight=MAX_BETA,
            log_dir=LOG_DIR,
            checkpoint_dir=MODEL_SAVE_DIR
        )
    
    # 设置日志记录器
    logger = setup_logger(args.log_dir)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 检查并准备数据
    if not os.path.exists(args.data_file):
        logging.info(f"未找到处理后的数据文件 {args.data_file}，开始预处理数据...")
        # 加载原始序列
        sequences = load_sequences(SEQUENCE_FILE)
        
        # 分析环数分布
        ring_distribution = analyze_ring_distribution(sequences)
        max_rings = max(ring_distribution.keys())
        logging.info(f"环数分布: {ring_distribution}")
        logging.info(f"最大环数: {max_rings}")
        
        # 预处理并生成embeddings
        preprocess_and_embed(
            sequences=sequences,
            output_path=args.data_file,
            max_length=args.max_length
        )
        logging.info(f"数据预处理完成，已保存到 {args.data_file}")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        data_file=args.data_file,
        batch_size=args.batch_size,
        train_test_split=args.val_split,
        num_workers=args.num_workers,
        pin_memory=True,
        max_sequence_length=args.max_length
    )
    
    # 加载ESM tokenizer以获取词汇表大小
    logger.info("加载ESM tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ESM_MODEL_PATH,  # 使用config.py中定义的路径
        local_files_only=True  # 强制使用本地文件
    )
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    logger.info(f"ESM tokenizer词汇表大小: {vocab_size}")
    logger.info(f"ESM tokenizer pad token ID: {pad_token_id}")
    
    # 初始化模型
    logger.info("初始化模型...")
    model = ESMVAEToken(
        input_dim=args.input_dim,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=args.latent_dim,
        vocab_size=vocab_size,
        max_sequence_length=args.max_length,
        pad_token_id=pad_token_id,
        use_layer_norm=True,
        dropout=args.dropout
    ).to(device)
    
    # 初始化优化器
    logger.info("初始化优化器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # 初始化损失历史
    train_loss_history = []
    train_recon_history = []
    train_kl_history = []
    val_loss_history = []
    val_recon_history = []
    val_kl_history = []
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_losses = train_epoch_new(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_weight=args.kl_weight
        )
        
        # 验证
        val_losses = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            kl_weight=args.kl_weight
        )
        
        # 记录损失历史
        train_loss_history.append(train_losses['loss'])
        train_recon_history.append(train_losses['recon_loss'])
        train_kl_history.append(train_losses['kl_loss'])
        val_loss_history.append(val_losses['loss'])
        val_recon_history.append(val_losses['recon_loss'])
        val_kl_history.append(val_losses['kl_loss'])
        
        # 打印损失
        print(f"[Epoch {epoch+1}] train_loss={train_losses['loss']:.4f}, train_recon={train_losses['recon_loss']:.4f}, train_kl={train_losses['kl_loss']:.4f}")
        print(f"[Epoch {epoch+1}] val_loss={val_losses['loss']:.4f}, val_recon={val_losses['recon_loss']:.4f}, val_kl={val_losses['kl_loss']:.4f}")
        
        # 记录损失和log_var统计信息
        logging.info(
            f"Train - Loss: {train_losses['loss']:.4f}, "
            f"Recon: {train_losses['recon_loss']:.4f}, "
            f"KL: {train_losses['kl_loss']:.4f}\n"
            f"Train log_var - Mean: {train_losses['log_var_stats']['mean']:.4f}, "
            f"Std: {train_losses['log_var_stats']['std']:.4f}, "
            f"Min: {train_losses['log_var_stats']['min']:.4f}, "
            f"Max: {train_losses['log_var_stats']['max']:.4f}"
        )
        logging.info(
            f"Val - Loss: {val_losses['loss']:.4f}, "
            f"Recon: {val_losses['recon_loss']:.4f}, "
            f"KL: {val_losses['kl_loss']:.4f}\n"
            f"Val log_var - Mean: {val_losses['log_var_stats']['mean']:.4f}, "
            f"Std: {val_losses['log_var_stats']['std']:.4f}, "
            f"Min: {val_losses['log_var_stats']['min']:.4f}, "
            f"Max: {val_losses['log_var_stats']['max']:.4f}"
        )
        
        # 保存最佳模型
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                losses=val_losses,
                checkpoint_dir=args.checkpoint_dir,
                is_best=True  # 标记为最佳模型
            )
            
        # 检查潜在空间是否被过度压缩
        if train_losses['log_var_stats']['mean'] < -10:  # 如果log_var均值过小
            logging.warning(
                f"警告：潜在空间可能被过度压缩！"
                f"log_var均值 ({train_losses['log_var_stats']['mean']:.4f}) 过小"
            )
        if train_losses['log_var_stats']['std'] < 0.1:  # 如果log_var标准差过小
            logging.warning(
                f"警告：潜在空间可能缺乏多样性！"
                f"log_var标准差 ({train_losses['log_var_stats']['std']:.4f}) 过小"
            )
    
    # 绘制损失曲线
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Val Loss')
    plt.plot(epochs, train_recon_history, label='Train Recon Loss', linestyle='--')
    plt.plot(epochs, val_recon_history, label='Val Recon Loss', linestyle='--')
    plt.plot(epochs, train_kl_history, label='Train KL Loss', linestyle=':')
    plt.plot(epochs, val_kl_history, label='Val KL Loss', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    print('损失曲线已保存为 loss_curve.png')

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='训练条件VAE模型')
    
    # 数据参数
    parser.add_argument('--data_file', type=str, default=EMBEDDING_FILE, help='数据文件路径')
    parser.add_argument('--max_length', type=int, default=MAX_SEQUENCE_LENGTH, help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批次大小')
    parser.add_argument('--val_split', type=float, default=TRAIN_TEST_SPLIT, help='验证集比例')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=ESM_EMBEDDING_DIM, help='输入维度（ESM嵌入维度）')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIMS[0], help='隐藏层维度')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='潜在空间维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--kl_weight', type=float, default=MAX_BETA, help='KL散度权重')
    
    # 其他参数
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='日志目录')
    parser.add_argument('--checkpoint_dir', type=str, default=MODEL_SAVE_DIR, help='检查点目录')
    
    args = parser.parse_args()
    main(args)

# 添加一个直接调用的函数
def run_training():
    """直接运行训练的函数，不需要传入参数"""
    main(None) 