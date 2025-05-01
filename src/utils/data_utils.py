import os
import numpy as np
import torch
import random
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from config.data_config import (
    MAX_SEQUENCE_LENGTH,
    ALPHABET,
    NUM_WORKERS,
    PIN_MEMORY
)
from config.config import *
import pandas as pd

class EmbeddingDataset(Dataset):
    """ESM Embeddings数据集"""
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        sequences: List[str],
        max_length: int = 64
    ) -> None:
        """初始化数据集
        
        Args:
            embeddings: 预计算的ESM embeddings
            sequences: 对应的序列列表
            max_length: 最大序列长度
        """
        # 确保embeddings在CPU上
        self.embeddings = embeddings
        self.sequences = sequences
        self.max_length = max_length
        
        # 打印embeddings的形状以进行调试
        print(f"Embeddings shape in dataset: {self.embeddings.shape}")
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            包含embeddings和序列的字典
        """
        # 获取embeddings
        embedding = self.embeddings[idx]
        
        # 将序列转换为张量并填充到固定长度
        sequence = self.sequences[idx]
        sequence_tensor = torch.zeros(self.max_length, dtype=torch.long)  # 创建全零张量
        sequence_encoded = torch.tensor([ord(c) for c in sequence], dtype=torch.long)  # 编码序列
        sequence_tensor[:min(len(sequence_encoded), self.max_length)] = sequence_encoded[:self.max_length]  # 填充序列
        
        # 打印单个样本的形状以进行调试
        if idx == 0:
            print(f"Sample {idx} - Embedding shape: {embedding.shape}, Sequence shape: {sequence_tensor.shape}")
        
        return {
            'embeddings': embedding,  # 直接返回预计算的embedding
            'sequence': sequence_tensor
        }

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
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):
                    sequences.append(line)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        raise
    
    if not sequences:
        print(f"警告: 文件 {file_path} 中没有找到任何序列")
    
    print(f"从文件 {file_path} 中成功加载 {len(sequences)} 个序列")
    return sequences

def create_data_loaders(
    embeddings: torch.Tensor,
    sequences: List[str],
    batch_size: int,
    train_test_split: float = 0.15,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    max_sequence_length: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器
    
    Args:
        embeddings: 预计算的ESM embeddings
        sequences: 对应的序列列表
        batch_size: 批次大小
        train_test_split: 训练集比例
        num_workers: 数据加载线程数
        pin_memory: 是否使用固定内存
        max_sequence_length: 最大序列长度
        
    Returns:
        训练和验证数据加载器
    """
    # 打印输入embeddings的形状以进行调试
    print(f"Input embeddings shape: {embeddings.shape}")
    
    # 划分训练集和验证集
    indices = np.arange(len(embeddings))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - train_test_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # 创建数据集
    train_dataset = EmbeddingDataset(
        embeddings=embeddings[train_indices],
        sequences=[sequences[i] for i in train_indices],
        max_length=max_sequence_length
    )
    val_dataset = EmbeddingDataset(
        embeddings=embeddings[val_indices],
        sequences=[sequences[i] for i in val_indices],
        max_length=max_sequence_length
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃不完整的批次
    )
    
    return train_loader, val_loader

def get_esm_embeddings(
    sequences: List[str],
    batch_size: int = 32
) -> torch.Tensor:
    """使用ESM模型获取序列的embeddings
    
    Args:
        sequences: 氨基酸序列列表
        batch_size: 批处理大小，用于避免OOM错误
        
    Returns:
        序列的embeddings张量，形状为 [num_sequences, embedding_dim]
    """
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME)
    
    # 将模型移到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 初始化存储所有embeddings的列表
    all_embeddings = []
    
    # 分批处理序列
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        # 对序列进行编码
        encoded = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        # 将输入移到设备
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # 获取embeddings
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # 获取最后一层的hidden states
            hidden_states = outputs.last_hidden_state
            
            # 使用attention_mask进行平均池化
            # 将attention_mask扩展到最后两个维度
            attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            # 将padding位置的hidden states置为0
            hidden_states = hidden_states * attention_mask
            # 计算每个序列的有效长度（非padding位置的数量）
            seq_lengths = attention_mask.sum(dim=1)
            # 计算平均池化
            batch_embeddings = hidden_states.sum(dim=1) / seq_lengths
            
            all_embeddings.append(batch_embeddings)
    
    # 将所有批次的embeddings拼接起来并移到CPU
    embeddings = torch.cat(all_embeddings, dim=0).cpu()
    
    return embeddings

def prepare_VAE(data: str | List[str], csv: bool = False) -> Tuple[torch.Tensor, List[str]]:
    """准备VAE的输入数据
    
    Args:
        data: 序列数据（可以是文件路径或序列列表）
        csv: 是否为CSV文件
        
    Returns:
        序列的embeddings张量和原始序列列表
    """
    if isinstance(data, str):
        if csv:
            # 如果是CSV文件，读取序列
            try:
                df = pd.read_csv(data)
                if 'sequence' not in df.columns:
                    raise ValueError("CSV文件必须包含'sequence'列")
                sequences = df['sequence'].tolist()
            except Exception as e:
                print(f"读取CSV文件时出错: {str(e)}")
                raise
        else:
            # 如果是文本文件，直接读取序列
            try:
                sequences = load_sequences(data)
            except Exception as e:
                print(f"读取文本文件时出错: {str(e)}")
                raise
    else:
        # 如果已经是序列列表，直接使用
        sequences = data
    
    if not sequences:
        raise ValueError("没有找到任何序列数据")
    
    print(f"成功加载 {len(sequences)} 个序列")
    
    # 使用ESM模型获取embeddings
    try:
        embeddings = get_esm_embeddings(sequences)
        print(f"成功生成embeddings，形状: {embeddings.shape}")
    except Exception as e:
        print(f"生成embeddings时出错: {str(e)}")
        raise
    
    return embeddings, sequences

