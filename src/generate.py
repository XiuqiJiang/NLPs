import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime
import traceback
from typing import Dict, Any, List

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 默认配置值
NUM_SAMPLES = 10
LATENT_DIM = 50
MAX_SEQUENCE_LENGTH = 64
MODEL_WEIGHTS_PATH = 'results/models/vae/weights/epoch_100.pth'
GENERATED_SEQUENCES_PATH = 'results/generated/sequences.txt'
ESM_MODEL_NAME = "esm_model"
RANDOM_SEED = 42

try:
    from config.config import *
except ImportError:
    print("Warning: Could not import from config.config. Using default values.")

from src.models.vae import ESMVAE
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
    model: ESMVAE,
    tokenizer: AutoTokenizer,
    device: torch.device,
    num_samples: int = NUM_SAMPLES
) -> Dict[str, Any]:
    """生成序列
    
    Args:
        model: VAE模型
        tokenizer: 分词器
        device: 设备
        num_samples: 生成序列数量
        
    Returns:
        生成结果和指标
    """
    print(f"开始生成序列，使用设备: {device}")
    print(f"生成参数: num_samples={num_samples}")
    
    model.eval()
    sequences = []
    total_time = 0
    unique_sequences = set()
    
    with torch.no_grad():
        for i in range(num_samples):
            start_time = datetime.now()
            
            # 从先验分布采样
            z = torch.randn(1, LATENT_DIM).to(device)
            
            # 生成序列
            sequence = model.decode(z)
            
            # 解码序列
            sequence = tokenizer.decode(sequence[0].tolist())
            sequences.append(sequence)
            unique_sequences.add(sequence)
            
            # 计算耗时
            batch_time = (datetime.now() - start_time).total_seconds()
            total_time += batch_time
            
            print(f"生成第 {i+1}/{num_samples} 个序列:")
            print(f"序列长度: {len(sequence)}")
            print(f"生成耗时: {batch_time:.2f}秒")
            if torch.cuda.is_available():
                print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    
    # 计算指标
    metrics = {
        'num_sequences': len(sequences),
        'avg_length': sum(len(s) for s in sequences) / len(sequences),
        'unique_sequences': len(unique_sequences)
    }
    
    print("\n生成完成:")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个序列耗时: {total_time/num_samples:.2f}秒")
    print(f"生成序列总数: {metrics['num_sequences']}")
    print(f"平均序列长度: {metrics['avg_length']:.2f}")
    print(f"唯一序列数: {metrics['unique_sequences']}")
    
    return sequences, metrics

def main() -> None:
    """主函数"""
    try:
        print("开始序列生成任务")
        
        # 设置随机种子
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        print(f"设置随机种子: {RANDOM_SEED}")
        
        # 打印系统信息
        print_system_info()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        print(f"加载本地ESM模型: {ESM_MODEL_NAME}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 初始化模型
        model = ESMVAE().to(device)
        
        # 加载模型权重
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            print(f"找不到模型权重文件: {MODEL_WEIGHTS_PATH}")
            print("使用随机初始化的权重")
        else:
            checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型权重: {MODEL_WEIGHTS_PATH}")
        
        # 生成序列
        sequences, metrics = generate_sequences(
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_samples=NUM_SAMPLES
        )
        
        # 保存生成的序列
        os.makedirs(os.path.dirname(GENERATED_SEQUENCES_PATH), exist_ok=True)
        with open(GENERATED_SEQUENCES_PATH, 'w') as f:
            for sequence in sequences:
                f.write(sequence + '\n')
        
        print(f"保存生成的序列到: {GENERATED_SEQUENCES_PATH}")
        print(f"成功生成 {len(sequences)} 个序列，保存至 {GENERATED_SEQUENCES_PATH}")
        print("序列生成任务完成")
        
    except Exception as e:
        print(f"序列生成任务失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误位置: {traceback.format_exc()}")

if __name__ == '__main__':
    main() 