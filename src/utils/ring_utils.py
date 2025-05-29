import torch
from typing import List, Dict, Tuple
import numpy as np

def count_cysteines(sequence: str) -> int:
    """计算序列中半胱氨酸(C)的数量
    
    Args:
        sequence: 氨基酸序列
        
    Returns:
        半胱氨酸的数量
    """
    return sequence.count('C')

def get_ring_info(sequence: str) -> int:
    """计算序列中的环数标签（半胱氨酸数量-1，保证标签从0开始）
    
    Args:
        sequence: 蛋白质序列
        
    Returns:
        环数标签（半胱氨酸数量-1，最小为0）
    """
    return max(sequence.count('C') - 1, 0)

def analyze_ring_distribution(sequences: List[str]) -> Dict[str, int]:
    """分析数据集中环数的分布
    
    Args:
        sequences: 序列列表
        
    Returns:
        环数分布字典
    """
    ring_counts = {}
    for seq in sequences:
        rings = get_ring_info(seq)
        ring_counts[rings] = ring_counts.get(rings, 0) + 1
    return ring_counts

def get_ring_embedding(ring_info: torch.Tensor, embedding_dim: int = 32) -> torch.Tensor:
    """将环数信息转换为嵌入向量
    
    Args:
        ring_info: 环数张量 [batch_size]
        embedding_dim: 嵌入维度
        
    Returns:
        环数嵌入 [batch_size, embedding_dim]
    """
    # 使用正弦位置编码
    position = ring_info.float().unsqueeze(-1)  # [batch_size, 1]
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
    pos_encoding = torch.zeros(ring_info.size(0), embedding_dim, device=ring_info.device)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding 