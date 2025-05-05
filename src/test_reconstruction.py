import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path

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
    DEVICE
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

def load_model(model_path: str, device: str) -> ESMVAEToken:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        
    Returns:
        加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载tokenizer以获取词汇表大小
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    
    # 初始化模型
    model = ESMVAEToken(
        input_dim=640,  # ESM-2的嵌入维度
        hidden_dims=[512, 256],
        latent_dim=256,
        vocab_size=vocab_size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=pad_token_id,
        use_layer_norm=True,
        dropout=0.1
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
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
        original_token_ids = batch['token_ids']
        
        # 编码
        with torch.no_grad():
            # 使用编码器获取mu和logvar
            mu, logvar = model.encode(batch['embeddings'])
            
            # 直接使用mu进行重构（不使用重参数化）
            z = mu
            
            # 解码（使用自回归模式）
            reconstructed_token_ids = model.decode(z, target_ids=None)  # 直接获取生成的token IDs
        
        # 将token ids转换回序列
        for j in range(len(original_token_ids)):
            original_seq = tokenizer.decode(original_token_ids[j], skip_special_tokens=True)
            reconstructed_seq = tokenizer.decode(reconstructed_token_ids[j], skip_special_tokens=True)
            
            # 计算序列相似度
            similarity = sum(a == b for a, b in zip(original_seq, reconstructed_seq)) / len(original_seq)
            
            logger.info(f"\n样本 {i*BATCH_SIZE + j + 1}:")
            logger.info(f"原始序列: {original_seq}")
            logger.info(f"重构序列: {reconstructed_seq}")
            logger.info(f"序列相似度: {similarity:.2%}")

def generate_sequences(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    epoch: int,
    num_sequences: int = 10,
    device: str = DEVICE
) -> None:
    """生成新的序列
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        epoch: 当前epoch
        num_sequences: 要生成的序列数量
        device: 设备
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"Epoch {epoch} 的生成结果:")
    logger.info(f"{'='*50}")
    
    # 生成随机潜在向量
    z = torch.randn(num_sequences, model.latent_dim, device=device)
    
    # 解码（使用自回归模式）
    with torch.no_grad():
        generated_token_ids = model.decode(z, target_ids=None)  # 直接获取生成的token IDs
    
    # 将生成的token ids转换回序列
    for i in range(num_sequences):
        sequence = tokenizer.decode(generated_token_ids[i], skip_special_tokens=True)
        logger.info(f"\n生成的序列 {i+1}:")
        logger.info(sequence)
    
    logger.info("序列生成完成！")

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
    model, tokenizer = load_model(model_path, DEVICE)
    
    # 重构序列
    reconstruct_sequences(
        model=model,
        tokenizer=tokenizer,
        data_loader=val_loader,
        device=DEVICE,
        epoch=epoch,
        num_samples=2
    )
    
    # 生成新序列
    generate_sequences(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        num_sequences=5
    )

def main():
    """主函数"""
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
    
    # 测试不同epoch的模型
    epochs_to_test = [100, 200, 300, 400, 500]
    for epoch in epochs_to_test:
        model_path = os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch}.pt')
        if os.path.exists(model_path):
            test_epoch(epoch, model_path, val_loader)
        else:
            logger.warning(f"找不到 Epoch {epoch} 的模型文件: {model_path}")
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main() 