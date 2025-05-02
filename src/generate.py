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
    ESM_EMBEDDING_DIM
)

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
    id_to_char: Dict[int, str],
    temperature: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """生成单个蛋白质序列
    
    Args:
        model: VAE模型
        id_to_char: ID到字符的映射
        temperature: 采样温度，控制生成的多样性
        device: 设备
        
    Returns:
        生成的序列
    """
    model.eval()
    
    with torch.no_grad():
        # 从标准正态分布采样隐变量
        z = torch.randn(1, model.latent_dim).to(device)
        
        # 解码生成logits
        logits = model.decode(z)  # shape: (1, max_length, vocab_size)
        
        # 应用温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 使用softmax获取概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 从概率分布中采样token序列
        sampled_ids = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(1, -1)
        
        # 将token IDs转换为字符序列
        sequence = []
        for token_id in sampled_ids[0]:
            if token_id.item() == EOS_TOKEN_ID:
                break
            if token_id.item() != PAD_TOKEN_ID:
                sequence.append(id_to_char[token_id.item()])
        
        return ''.join(sequence)

def generate_sequences(
    model: ESMVAEToken,
    num_sequences: int,
    temperature: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[str]:
    """生成多个蛋白质序列
    
    Args:
        model: VAE模型
        num_sequences: 要生成的序列数量
        temperature: 采样温度
        device: 设备
        
    Returns:
        生成的序列列表
    """
    id_to_char = create_id_to_char_mapping()
    sequences = []
    
    for _ in tqdm(range(num_sequences), desc="生成序列"):
        sequence = generate_sequence(model, id_to_char, temperature, device)
        if sequence:  # 只添加非空序列
            sequences.append(sequence)
    
    return sequences

def main():
    parser = argparse.ArgumentParser(description="使用VAE生成蛋白质序列")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--num_sequences", type=int, default=10, help="要生成的序列数量")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度，控制生成的多样性")
    parser.add_argument("--output_file", type=str, default="generated_sequences.txt", help="输出文件路径")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = ESMVAEToken(
        embedding_dim=ESM_EMBEDDING_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        use_layer_norm=True
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"从 {args.model_path} 加载模型")
    
    # 生成序列
    sequences = generate_sequences(
        model,
        args.num_sequences,
        args.temperature,
        device
    )
    
    # 保存序列
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f">Generated_sequence_{i}\n")
            f.write(f"{seq}\n")
    
    print(f"生成 {len(sequences)} 个序列并保存到 {output_path}")
    
    # 打印生成的序列长度分布
    generated_lengths = [len(seq) for seq in sequences]
    print("\n生成序列的长度分布:")
    for length, count in Counter(generated_lengths).items():
        print(f"长度 {length}: {count} 个序列")

if __name__ == "__main__":
    main() 