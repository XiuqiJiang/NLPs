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
from collections import Counter
import logging

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vae_token import ESMVAEToken
from config.config import (
    ESM_MODEL_NAME,
    LATENT_DIM,
    HIDDEN_DIMS,
    ALPHABET,
    DATA_PATH,
    MAX_SEQUENCE_LENGTH,
    VOCAB_SIZE,
    PAD_TOKEN_ID,
    EOS_TOKEN_ID,
    ESM_EMBEDDING_DIM,
    DEVICE,
    MODEL_SAVE_DIR
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_id_to_char_mapping() -> Dict[int, str]:
    """创建ID到字符的映射
    
    Returns:
        ID到字符的映射字典
    """
    return {i: char for i, char in enumerate(ALPHABET)}

def get_length_distribution(data_path: str) -> Dict[int, float]:
    """获取训练数据的长度分布
    
    Args:
        data_path: 训练数据文件路径
        
    Returns:
        长度分布字典，键为长度，值为概率
    """
    lengths = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                lengths.append(len(line.strip()))
    
    # 计算长度分布
    length_counts = Counter(lengths)
    total = sum(length_counts.values())
    length_probs = {length: count/total for length, count in length_counts.items()}
    
    return length_probs

def sample_length(length_probs: Dict[int, float]) -> int:
    """从长度分布中采样一个长度
    
    Args:
        length_probs: 长度分布字典
        
    Returns:
        采样的长度
    """
    lengths = list(length_probs.keys())
    probs = list(length_probs.values())
    return np.random.choice(lengths, p=probs)

def generate_sequence(
    model: ESMVAEToken,
    temperature: float = 1.0,
    max_length: Optional[int] = None
) -> str:
    """生成单个序列
    
    Args:
        model: VAE模型
        temperature: 采样温度
        max_length: 最大序列长度
        
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
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # 采样token IDs
        token_ids = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(1, -1)
        
        # 转换为字符序列
        id_to_char = create_id_to_char_mapping()
        sequence = []
        
        # 处理生成的token IDs
        for token_id in token_ids[0]:
            token_id = token_id.item()
            
            # 如果遇到EOS token，停止生成
            if token_id == EOS_TOKEN_ID:
                break
                
            # 跳过PAD token
            if token_id == PAD_TOKEN_ID:
                continue
                
            # 添加字符到序列
            sequence.append(id_to_char[token_id])
            
            # 如果达到最大长度，停止生成
            if max_length and len(sequence) >= max_length:
                break
        
        return ''.join(sequence)

def generate_sequences(
    model_path: str,
    num_sequences: int = 10,
    temperature: float = 1.0,
    max_length: Optional[int] = None
) -> List[str]:
    """生成多个序列
    
    Args:
        model_path: 模型路径
        num_sequences: 要生成的序列数量
        temperature: 采样温度
        max_length: 最大序列长度
        
    Returns:
        生成的序列列表
    """
    logger = setup_logging()
    logger.info(f"从 {model_path} 加载模型...")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 初始化模型
    model = ESMVAEToken(
        input_dim=ESM_EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=PAD_TOKEN_ID,
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
            temperature=temperature,
            max_length=max_length
        )
        sequences.append(sequence)
        logger.info(f"序列 {i+1}: {sequence}")
    
    return sequences

def main():
    """主函数"""
    # 生成序列
    sequences = generate_sequences(
        model_path=os.path.join(MODEL_SAVE_DIR, 'best_model.pt'),
        num_sequences=10,
        temperature=0.8,
        max_length=MAX_SEQUENCE_LENGTH
    )
    
    # 保存生成的序列
    output_file = os.path.join(MODEL_SAVE_DIR, 'generated_sequences.txt')
    with open(output_file, 'w') as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f"序列 {i}:\n{seq}\n\n")
    
    print(f"生成的序列已保存到 {output_file}")

if __name__ == "__main__":
    main() 