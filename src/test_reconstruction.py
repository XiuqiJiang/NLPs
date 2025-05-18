import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path
import torch.nn.functional as F
import argparse

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.models.vae_token import ESMVAEToken
from src.utils.data_utils import create_data_loaders
from config.config import (
    EMBEDDING_FILE,
    ESM_MODEL_PATH,
    BATCH_SIZE,
    TRAIN_TEST_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
    MAX_SEQUENCE_LENGTH,
    MODEL_SAVE_DIR,
    DEVICE,
    ESM_EMBEDDING_DIM,
    HIDDEN_DIMS,
    LATENT_DIM
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reconstruction_test.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(model_path: str) -> tuple[ESMVAEToken, AutoTokenizer]:
    """加载模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        (模型, tokenizer)
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    
    # 初始化模型
    model = ESMVAEToken(
        input_dim=ESM_EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        use_layer_norm=True,
        dropout=0.1
    ).to(DEVICE)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def reconstruct_sequences(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    num_samples: int = 5
) -> None:
    """重构序列并比较
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        data_loader: 数据加载器
        device: 设备
        epoch: 当前epoch
        num_samples: 要重构的样本数量
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"Epoch {epoch} 的重构结果:")
    logger.info(f"{'='*50}")
    
    # 选择前num_samples个批次
    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
            
        # 将数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 获取原始序列
        original_token_ids = batch['input_ids']
        
        # 编码和解码
        with torch.no_grad():
            # 使用编码器获取mu和logvar
            mu, logvar = model.encode(batch['embeddings'])
            
            # 使用重参数化
            z = model.reparameterize(mu, logvar)
            
            # 解码（使用自回归模式）
            reconstructed_token_ids = model.decode(z, target_ids=None)
        
        # 将token ids转换回序列并显示结果
        for j in range(len(original_token_ids)):
            # 获取原始序列和重构序列
            original_seq = tokenizer.decode(original_token_ids[j], skip_special_tokens=True)
            reconstructed_seq = tokenizer.decode(reconstructed_token_ids[j], skip_special_tokens=True)
            
            # 计算序列相似度
            similarity = calculate_sequence_similarity(original_seq, reconstructed_seq)
            
            # 计算氨基酸组成
            original_aa_comp = calculate_aa_composition(original_seq)
            reconstructed_aa_comp = calculate_aa_composition(reconstructed_seq)
            
            # 记录详细信息
            logger.info(f"\n{'='*30}")
            logger.info(f"样本 {i*BATCH_SIZE + j + 1}:")
            logger.info(f"{'='*30}")
            logger.info(f"原始序列: {original_seq}")
            logger.info(f"重构序列: {reconstructed_seq}")
            logger.info(f"序列相似度: {similarity:.2%}")
            logger.info(f"原始序列长度: {len(original_seq)}")
            logger.info(f"重构序列长度: {len(reconstructed_seq)}")
            logger.info("\n氨基酸组成比较:")
            logger.info("原始序列:")
            for aa, ratio in sorted(original_aa_comp.items()):
                logger.info(f"  {aa}: {ratio:.2%}")
            logger.info("重构序列:")
            for aa, ratio in sorted(reconstructed_aa_comp.items()):
                logger.info(f"  {aa}: {ratio:.2%}")
            logger.info(f"{'='*30}\n")

def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """计算两个序列的相似度
    
    Args:
        seq1: 第一个序列
        seq2: 第二个序列
        
    Returns:
        相似度，范围[0,1]
    """
    # 移除特殊token
    seq1 = seq1.replace('<pad>', '').replace('', '').replace('', '')
    seq2 = seq2.replace('<pad>', '').replace('', '').replace('', '')
    
    if not seq1 or not seq2:
        return 0.0
    
    # 计算最长公共子序列
    lcs = longest_common_subsequence(seq1, seq2)
    return len(lcs) / max(len(seq1), len(seq2))

def longest_common_subsequence(seq1: str, seq2: str) -> str:
    """计算两个序列的最长公共子序列
    
    Args:
        seq1: 第一个序列
        seq2: 第二个序列
        
    Returns:
        最长公共子序列
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充dp表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 重建最长公共子序列
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            lcs.append(seq1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))

def calculate_aa_composition(seq: str) -> dict:
    """计算序列的氨基酸组成
    
    Args:
        seq: 氨基酸序列
        
    Returns:
        包含每个氨基酸比例的字典
    """
    # 移除特殊token
    seq = seq.replace('<pad>', '').replace('', '').replace('', '')
    
    if not seq:
        return {}
    
    # 计算每个氨基酸的出现次数
    aa_counts = {}
    for aa in seq:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    # 计算比例
    total = len(seq)
    return {aa: count/total for aa, count in aa_counts.items()}

def test_epoch(epoch: int, model_path: str, val_loader: DataLoader) -> None:
    """测试特定epoch的模型
    
    Args:
        epoch: 要测试的epoch
        model_path: 模型文件路径
        val_loader: 验证数据加载器
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"开始测试 Epoch {epoch} 的模型")
    logger.info(f"{'='*50}")
    
    # 加载模型
    model, tokenizer = load_model(model_path)
    
    # 重构序列
    reconstruct_sequences(
        model=model,
        tokenizer=tokenizer,
        data_loader=val_loader,
        device=DEVICE,
        epoch=epoch,
        num_samples=2
    )

def find_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """查找所有检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        包含(epoch, path)的列表，按epoch排序
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
        
    checkpoints = []
    for f in checkpoint_dir.glob('checkpoint_epoch_*.pt'):
        try:
            epoch = int(f.stem.split('_')[-1])
            checkpoints.append((epoch, str(f)))
        except ValueError:
            continue
            
    return sorted(checkpoints, key=lambda x: x[0])

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试VAE模型的重构能力')
    parser.add_argument('--epochs', type=str, default='all',
                      help='要测试的epoch，可以是单个数字，多个数字用逗号分隔，或"all"表示所有')
    parser.add_argument('--best_only', action='store_true',
                      help='只测试最佳模型')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("开始重构测试...")
    
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
    
    # 查找检查点
    checkpoints = find_checkpoints(MODEL_SAVE_DIR)
    if not checkpoints:
        logger.error(f"在 {MODEL_SAVE_DIR} 中没有找到任何检查点文件")
        return
        
    # 确定要测试的epoch
    if args.best_only:
        # 只测试最佳模型
        best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pt')
        if os.path.exists(best_model_path):
            test_epoch(0, best_model_path, val_loader)  # epoch设为0表示最佳模型
        else:
            logger.error(f"找不到最佳模型文件: {best_model_path}")
    else:
        if args.epochs == 'all':
            # 测试所有检查点
            for epoch, path in checkpoints:
                test_epoch(epoch, path, val_loader)
        else:
            # 测试指定的epoch
            try:
                epochs = [int(e) for e in args.epochs.split(',')]
                for epoch in epochs:
                    # 查找最接近的检查点
                    closest_checkpoint = min(checkpoints, key=lambda x: abs(x[0] - epoch))
                    if abs(closest_checkpoint[0] - epoch) <= 10:  # 允许10个epoch的误差
                        test_epoch(closest_checkpoint[0], closest_checkpoint[1], val_loader)
                    else:
                        logger.warning(f"找不到接近epoch {epoch} 的检查点")
            except ValueError:
                logger.error("epoch参数格式错误，请使用数字或逗号分隔的数字列表")
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main() 