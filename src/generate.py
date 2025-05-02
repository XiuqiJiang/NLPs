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

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

try:
    from config.config import *
    print("成功导入配置文件")
    print(f"从配置文件导入的LATENT_DIM: {LATENT_DIM}")
except ImportError:
    print("Warning: Could not import from config file. Using default values.")
    # 默认配置值
    LATENT_DIM = 64
    MAX_SEQUENCE_LENGTH = 64
    NUM_SAMPLES = 10
    MODEL_WEIGHTS_PATH = '/content/NLPs/results/models/vae/weights/checkpoint_epoch_95.pth'
    GENERATED_SEQUENCES_PATH = 'results/generated/sequences.txt'
    ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
    RANDOM_SEED = 42

from src.models.vae_token import ESMVAEToken
from src.utils.data_utils import load_sequences, create_data_loaders

def print_system_info():
    """打印系统信息"""
    print("系统信息:")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n当前配置:")
    print(f"NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"LATENT_DIM: {LATENT_DIM}")
    print(f"MAX_SEQUENCE_LENGTH: {MAX_SEQUENCE_LENGTH}")
    print(f"MODEL_WEIGHTS_PATH: {MODEL_WEIGHTS_PATH}")
    print(f"GENERATED_SEQUENCES_PATH: {GENERATED_SEQUENCES_PATH}")
    print(f"ESM_MODEL_NAME: {ESM_MODEL_NAME}")

def generate_sequences(
    model: ESMVAEToken,
    num_sequences: int,
    max_length: int = 100,
    temperature: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[str]:
    """生成蛋白质序列
    
    Args:
        model: VAE模型
        num_sequences: 要生成的序列数量
        max_length: 最大序列长度
        temperature: 采样温度
        device: 设备
        
    Returns:
        生成的序列列表
    """
    model.eval()
    sequences = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_sequences), desc="Generating sequences"):
            # 从标准正态分布采样隐变量
            z = torch.randn(1, model.latent_dim).to(device)
            
            # 解码生成logits
            logits = model.decode(z)  # shape: (1, vocab_size)
            
            # 应用温度缩放
            logits = logits / temperature
            
            # 使用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)
            
            # 从概率分布中采样token
            token_id = torch.multinomial(probs, num_samples=1).item()
            
            # 解码token为氨基酸序列
            sequence = model.tokenizer.decode([token_id])
            
            # 移除特殊token
            sequence = sequence.replace("<pad>", "").replace("<cls>", "").replace("<eos>", "").strip()
            
            if sequence:  # 只添加非空序列
                sequences.append(sequence)
    
    return sequences

def main():
    parser = argparse.ArgumentParser(description="使用VAE生成蛋白质序列")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--num_sequences", type=int, default=10, help="要生成的序列数量")
    parser.add_argument("--max_length", type=int, default=100, help="最大序列长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--output_file", type=str, default="generated_sequences.txt", help="输出文件路径")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model = ESMVAEToken().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # 生成序列
    sequences = generate_sequences(
        model,
        args.num_sequences,
        args.max_length,
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

if __name__ == "__main__":
    main() 