import os
import numpy as np
import torch
import random
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from config.data_config import (
    MAX_SEQUENCE_LENGTH,
    ALPHABET,
    NUM_WORKERS,
    PIN_MEMORY
)

class ProteinDataset(Dataset):
    """蛋白质序列数据集"""
    
    def __init__(
        self,
        sequences: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = MAX_SEQUENCE_LENGTH
    ) -> None:
        """初始化数据集
        
        Args:
            sequences: 蛋白质序列列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            编码后的序列和原始序列
        """
        sequence = self.sequences[idx]
        
        # 编码序列
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded['input_ids'].squeeze(0), sequence

def load_sequences(file_path: str) -> List[str]:
    """加载蛋白质序列
    
    Args:
        file_path: 序列文件路径
        
    Returns:
        序列列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"序列文件不存在: {file_path}")
    
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                sequences.append(line)
    
    return sequences

def create_data_loaders(
    sequences: List[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    train_test_split: float = 0.15,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器
    
    Args:
        sequences: 序列列表
        tokenizer: 分词器
        batch_size: 批次大小
        train_test_split: 训练集比例
        num_workers: 数据加载线程数
        pin_memory: 是否使用固定内存
        
    Returns:
        训练和验证数据加载器
    """
    # 划分训练集和验证集
    np.random.shuffle(sequences)
    split_idx = int(len(sequences) * (1 - train_test_split))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    # 创建数据集
    train_dataset = ProteinDataset(train_sequences, tokenizer)
    val_dataset = ProteinDataset(val_sequences, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def sequence_to_onehot(sequence: str) -> torch.Tensor:
    """将序列转换为one-hot编码
    
    Args:
        sequence: 蛋白质序列
        
    Returns:
        one-hot编码张量
    """
    onehot = torch.zeros(len(sequence), len(ALPHABET))
    for i, aa in enumerate(sequence):
        if aa in ALPHABET:
            onehot[i, ALPHABET.index(aa)] = 1
    return onehot

def onehot_to_sequence(onehot: torch.Tensor) -> str:
    """将one-hot编码转换为序列
    
    Args:
        onehot: one-hot编码张量
        
    Returns:
        蛋白质序列
    """
    indices = torch.argmax(onehot, dim=1)
    sequence = ''.join([ALPHABET[i] for i in indices])
    return sequence 

