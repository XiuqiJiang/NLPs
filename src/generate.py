import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime
import traceback
from typing import Dict, Any, List, Optional
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vae_token import ESMVAEToken
from config.config import (
    LATENT_DIM,
    HIDDEN_DIMS,
    MAX_SEQUENCE_LENGTH,
    ESM_EMBEDDING_DIM,
    DEVICE,
    MODEL_SAVE_DIR,
    SEQUENCE_FILE
)

# 使用本地ESM模型路径
ESM_MODEL_NAME = "/content/NLPs/esm_model"  # 使用绝对路径

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def generate_sequence(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    temperature: float = 1.0,
    max_length: Optional[int] = None,
    top_k: int = 50  # 添加top-k采样
) -> str:
    """生成单个序列
    
    Args:
        model: VAE模型
        tokenizer: ESM tokenizer
        temperature: 采样温度，较低的值使生成更保守
        max_length: 最大序列长度
        top_k: 只考虑概率最高的k个token
        
    Returns:
        生成的序列
    """
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样潜在向量
        z = torch.randn(1, model.latent_dim).to(DEVICE)
        
        # 解码得到logits
        logits = model.decode(z)  # [1, max_sequence_length, vocab_size]
        
        # 应用温度缩放
        logits = logits / temperature
        
        # 应用top-k采样
        if top_k > 0:
            # 确保k不超过词汇表大小
            k = min(top_k, logits.size(-1))
            if k > 0:
                indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # 采样token IDs
        token_ids = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(1, -1)
        
        # 转换为字符序列
        sequence = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        
        # 如果达到最大长度，截断序列
        if max_length and len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        return sequence

def generate_sequences(
    model_path: str,
    num_sequences: int = 10,
    temperature: float = 0.7,  # 降低默认温度使生成更保守
    max_length: Optional[int] = None,
    top_k: int = 50
) -> List[str]:
    """生成多个序列
    
    Args:
        model_path: 模型路径
        num_sequences: 要生成的序列数量
        temperature: 采样温度
        max_length: 最大序列长度
        top_k: top-k采样参数
        
    Returns:
        生成的序列列表
    """
    logger = setup_logging()
    logger.info(f"从 {model_path} 加载模型...")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 加载ESM tokenizer以获取词汇表大小
    logger.info(f"从 {ESM_MODEL_NAME} 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    
    # 确保top_k不超过词汇表大小
    top_k = min(top_k, vocab_size)
    logger.info(f"使用top_k={top_k}（词汇表大小={vocab_size}）")
    
    # 初始化模型
    model = ESMVAEToken(
        input_dim=ESM_EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        vocab_size=vocab_size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=pad_token_id,
        use_layer_norm=True,
        dropout=0.1
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # 生成序列
    logger.info(f"生成 {num_sequences} 个序列...")
    sequences = []
    for i in range(num_sequences):
        sequence = generate_sequence(
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_length=max_length,
            top_k=top_k
        )
        sequences.append(sequence)
        logger.info(f"序列 {i+1}: {sequence}")
    
    return sequences

def main():
    """主函数"""
    logger = setup_logging() # 在 main 开始时设置日志

    # 确保模型保存目录存在，如果不存在则创建
    if not os.path.exists(MODEL_SAVE_DIR):
        logger.warning(f"模型保存目录 {MODEL_SAVE_DIR} 不存在，将尝试创建。")
        try:
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        except OSError as e:
            logger.error(f"无法创建目录 {MODEL_SAVE_DIR}: {e}")
            return # 如果无法创建目录，则无法保存，提前退出

    model_file_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pt')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_file_path):
        logger.error(f"最佳模型文件不存在: {model_file_path}")
        return

    # 生成序列
    sequences = generate_sequences(
        model_path=model_file_path,
        num_sequences=10,
        temperature=0.7,  # 使用指定的温度
        max_length=MAX_SEQUENCE_LENGTH, # 使用 config 中的最大长度
        top_k=10  # 只考虑概率最高的10个token
    )

    if not sequences:
        logger.error("生成序列失败或未生成任何序列。")
        return

    # 保存生成的序列
    output_file = os.path.join(MODEL_SAVE_DIR, 'generated_sequences.txt')
    try:
        with open(output_file, 'w') as f:
            for i, seq in enumerate(sequences, 1):
                f.write(f"序列 {i}:\n{seq}\n\n")
        logger.info(f"生成的 {len(sequences)} 个序列已保存到 {output_file}")
    except IOError as e:
        logger.error(f"无法写入生成的序列到文件 {output_file}: {e}")

if __name__ == "__main__":
    main() 