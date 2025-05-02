import os
import sys
import torch
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
    HIDDEN_DIM,
    ALPHABET,
    DATA_PATH,
    MAX_SEQUENCE_LENGTH
)

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

def generate_sequences(
    model: ESMVAEToken,
    num_sequences: int,
    length_probs: Dict[int, float],
    temperature: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[str]:
    """生成蛋白质序列
    
    Args:
        model: VAE模型
        num_sequences: 要生成的序列数量
        length_probs: 长度分布字典
        temperature: 采样温度，控制生成的多样性
            - temperature > 1.0: 增加多样性
            - temperature < 1.0: 减少多样性
            - temperature = 1.0: 标准采样
        device: 设备
        
    Returns:
        生成的序列列表
    """
    model.eval()
    sequences = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_sequences), desc="Generating sequences"):
            # 从长度分布中采样目标长度
            target_length = sample_length(length_probs)
            
            # 从标准正态分布采样隐变量
            z = torch.randn(1, model.latent_dim).to(device)
            
            # 解码生成整个序列的logits
            # shape: (1, max_length, vocab_size)
            logits = model.decode(z)
            
            # 应用温度缩放
            if temperature != 1.0:
                logits = logits / temperature
            
            # 使用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)
            
            # 从概率分布中采样token序列
            # shape: (1, max_length)
            token_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)
            
            # 截取到目标长度
            token_ids = token_ids[:, :target_length]
            
            # 解码序列，跳过特殊token
            sequence = model.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
            
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
    print(f"Using device: {device}")
    
    # 获取训练数据的长度分布
    length_probs = get_length_distribution(DATA_PATH)
    print(f"Loaded length distribution from training data")
    print(f"Length range: {min(length_probs.keys())} - {max(length_probs.keys())}")
    
    # 加载模型
    model = ESMVAEToken().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # 生成序列
    sequences = generate_sequences(
        model,
        args.num_sequences,
        length_probs,
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
    
    print(f"Generated {len(sequences)} sequences and saved to {output_path}")
    
    # 打印生成的序列长度分布
    generated_lengths = [len(seq) for seq in sequences]
    print("\nGenerated sequences length distribution:")
    for length, count in Counter(generated_lengths).items():
        print(f"Length {length}: {count} sequences")

if __name__ == "__main__":
    main() 