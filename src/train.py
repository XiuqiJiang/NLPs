import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import matplotlib.pyplot as plt

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vae_token import ESMVAEToken, vae_token_loss
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
    WARMUP_EPOCHS,
    RNN_HIDDEN_DIM
)
from src.utils.ring_utils import get_ring_info, analyze_ring_distribution

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    pad_token_id: int,
    beta: float,
    scheduled_sampling_prob: float = 0.1
) -> Dict[str, float]:
    """训练一个epoch
    
    Args:
        model: VAE模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前训练轮次
        device: 训练设备
        pad_token_id: padding token的ID
        beta: KL散度权重
        scheduled_sampling_prob: 调度采样概率
        
    Returns:
        包含训练损失的字典
    """
    model.train()
    sum_loss = 0
    sum_recon_loss = 0
    sum_kld_loss = 0
    sum_kld_raw_mean = 0
    for batch_idx, batch in enumerate(train_loader):
        embeddings = batch['embeddings'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits, _, mu, logvar = model(
            embeddings, attention_mask, input_ids, scheduled_sampling_prob=scheduled_sampling_prob
        )
        loss, recon_loss, _, kld_loss, kld_raw_mean = vae_token_loss(
            logits, input_ids, mu, logvar, epoch, pad_token_id, beta=beta
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        sum_recon_loss += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
        sum_kld_loss += kld_loss.item() if isinstance(kld_loss, torch.Tensor) else kld_loss
        sum_kld_raw_mean += kld_raw_mean
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    avg_loss = sum_loss / len(train_loader)
    avg_recon_loss = sum_recon_loss / len(train_loader)
    avg_kld_loss = sum_kld_loss / len(train_loader)
    avg_kld_raw_mean = sum_kld_raw_mean / len(train_loader)
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'kld_loss': avg_kld_loss,
        'kld_raw_mean': avg_kld_raw_mean
    }

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    beta: float,
    scheduled_sampling_prob: float = 0.1
) -> Dict[str, float]:
    """验证模型
    
    Args:
        model: VAE模型
        val_loader: 验证数据加载器
        device: 设备
        pad_token_id: padding token的ID
        beta: KL散度权重
        scheduled_sampling_prob: 调度采样概率
        
    Returns:
        损失字典
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_kld_raw_mean = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            embeddings = batch['embeddings'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits, _, mu, logvar = model(
                embeddings, attention_mask, input_ids, scheduled_sampling_prob=scheduled_sampling_prob
            )
            loss, recon_loss, _, kld_loss, kld_raw_mean = vae_token_loss(
                logits, input_ids, mu, logvar, epoch=0, pad_token_id=pad_token_id, beta=beta
            )
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            total_recon_loss += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
            total_kld_loss += kld_loss.item() if isinstance(kld_loss, torch.Tensor) else kld_loss
            total_kld_raw_mean += kld_raw_mean
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kld_loss = total_kld_loss / num_batches
    avg_kld_raw_mean = total_kld_raw_mean / num_batches
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'kld_loss': avg_kld_loss,
        'kld_raw_mean': avg_kld_raw_mean
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
        ESM_MODEL_PATH,  # 使用config.py中定义的路径，已指向Google Drive
        local_files_only=True
    )
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    logger.info(f"ESM tokenizer词汇表大小: {vocab_size}")
    logger.info(f"ESM tokenizer pad token ID: {pad_token_id}")
    # ====== 若无bos_token_id则补充为0 ======
    if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = 0
    # ====== 断言特殊token id一致性（只检查pad/eos） ======
    assert tokenizer.pad_token_id == pad_token_id, f"pad_token_id不一致: {tokenizer.pad_token_id} vs {pad_token_id}"
    assert hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 2, f"eos_token_id不一致: {getattr(tokenizer, 'eos_token_id', None)} vs 2"
    
    scheduled_sampling_prob = 0.3  # 初始值，后续动态调整
    print(f"\n===== Scheduled Sampling概率: {scheduled_sampling_prob} =====")
    # 重新初始化模型和优化器，保证每轮独立
    model = ESMVAEToken(
        input_dim=args.input_dim,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=args.latent_dim,
        vocab_size=33,  # 词表大小可根据实际情况调整
        max_sequence_length=args.max_length,
        pad_token_id=1,  # pad_token_id可根据实际情况调整
        use_layer_norm=True,
        dropout=args.dropout,
        rnn_hidden_dim=RNN_HIDDEN_DIM
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # ====== 保证model.sos_token_id与tokenizer.bos_token_id一致 ======
    model.sos_token_id = tokenizer.bos_token_id
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 30  # 早停容忍轮数，延长到30轮
    min_delta = 1e-3  # 最小提升幅度
    for epoch in range(1000):  # 训练1000轮
        # 动态调整scheduled_sampling_prob
        if epoch < 10:
            scheduled_sampling_prob = 0.3
        elif epoch < 20:
            scheduled_sampling_prob = 0.5
        elif len(val_loss_history) > 0 and val_loss_history[-1] < 1.3:
            scheduled_sampling_prob = 0.7
        print(f"[SS={scheduled_sampling_prob}] Epoch {epoch+1}")
        beta = 0  # KL loss权重设为0，彻底关闭KL
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            pad_token_id=pad_token_id,
            beta=beta,
            scheduled_sampling_prob=scheduled_sampling_prob
        )
        val_losses = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            pad_token_id=pad_token_id,
            beta=beta,
            scheduled_sampling_prob=scheduled_sampling_prob
        )
        train_loss_history.append(train_losses['loss'])
        val_loss_history.append(val_losses['loss'])
        print(f"[SS={scheduled_sampling_prob}] Epoch {epoch+1} train_loss={train_losses['loss']:.4f}, val_loss={val_losses['loss']:.4f}, "
              f"train_kld={train_losses['kld_loss']:.4f}, val_kld={val_losses['kld_loss']:.4f}, "
              f"train_kld_raw={train_losses['kld_raw_mean']:.4f}, val_kld_raw={val_losses['kld_raw_mean']:.4f}")
        # 自动保存当前checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch+1,
            val_losses,
            args.checkpoint_dir,
            is_best=False
        )
        # 判断是否为最佳模型
        if val_losses['loss'] < best_val_loss - min_delta:
            best_val_loss = val_losses['loss']
            epochs_no_improve = 0
            save_checkpoint(
                model,
                optimizer,
                epoch+1,
                val_losses,
                args.checkpoint_dir,
                is_best=True
            )
            print(f"[EarlyStopping] 新的最佳模型已保存，val_loss={best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"[EarlyStopping] val_loss无提升，已连续{epochs_no_improve}轮")
        # 早停判断
        if epochs_no_improve >= patience:
            print(f"[EarlyStopping] 验证集loss连续{patience}轮无提升，提前停止训练。")
            break
        # 每隔10轮打印自回归生成样本
        if (epoch + 1) % 10 == 0:
            model.eval()
            batch = next(iter(train_loader))
            embeddings = batch['embeddings'][0:1].to(device)
            input_ids = batch['input_ids'][0:1].to(device)
            attention_mask = batch['attention_mask'][0:1].to(device)
            with torch.no_grad():
                mu, logvar = model.encode(embeddings, attention_mask)
                gen_ids = model.decode(mu)
                gen_seq = gen_ids[0].cpu().tolist()
                input_token_ids = input_ids[0].cpu().tolist()
                # decode为氨基酸序列
                input_tokens = tokenizer.decode(input_token_ids, skip_special_tokens=True)
                gen_tokens = tokenizer.decode(gen_seq, skip_special_tokens=True)
                # 统计有效长度（遇到eos或pad即停止）
                try:
                    eos_idx = gen_seq.index(2)
                except ValueError:
                    eos_idx = len(gen_seq)
                try:
                    pad_idx = gen_seq.index(1)
                except ValueError:
                    pad_idx = len(gen_seq)
                valid_len = min(eos_idx, pad_idx)
                print(f"[SS={scheduled_sampling_prob}] Epoch {epoch+1} 原始序列: {input_tokens}")
                print(f"[SS={scheduled_sampling_prob}] Epoch {epoch+1} 自回归生成序列: {gen_tokens} (有效长度: {valid_len})")
                # ====== 详细print每步token分布 ======
                print(f"[DEBUG] 生成token id序列: {gen_seq}")
                print(f"[DEBUG] 生成token解码: {gen_tokens}")
    # 保存loss曲线
    plt.plot(train_loss_history, label=f'Train SS={scheduled_sampling_prob}')
    plt.plot(val_loss_history, label=f'Val SS={scheduled_sampling_prob}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Scheduled Sampling={scheduled_sampling_prob}')
    plt.savefig(f'loss_curve_ss{int(scheduled_sampling_prob*100)}.png')
    plt.close()

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